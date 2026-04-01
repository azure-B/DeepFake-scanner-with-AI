"""
Deepfake Detection — 3-Stream Training Pipeline
데이터셋 : processed/celebdf/  (real 80K  /  fake 500K)
모델     : RGB(EfficientNetB4) + DCT(CNN) + LM(MLP) → Late Fusion
불균형   : Focal Loss (α=0.75, γ=2) + fake 언더샘플(3×real)
분할     : 영상 ID 단위 7:1:2  →  data leakage 방지
"""

import os, sys, re, time, logging
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision

# ── 로깅 설정
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"train_{_ts}.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── GPU
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
log.info(f"GPU {len(gpus)}개 감지")

# Mixed precision (Ampere/Turing 이상 권장)
mixed_precision.set_global_policy("mixed_float16")
log.info(f"compute dtype: {mixed_precision.global_policy().compute_dtype}")

# ═══════════════════════════════════════════════════════════════
# 0. CONFIG
# ═══════════════════════════════════════════════════════════════
CFG = {
    # ── 데이터
    "processed_root"   : "./processed",
    "dataset"          : "celebdf",
    "lm_suffix"        : None,       # None → 자동 감지 (_lm.npy 또는 _im.npy)

    # ── 불균형
    "undersample_ratio": 3.0,        # fake = real * ratio  (None → 사용 안 함)

    # ── 이미지
    "img_size"         : 224,

    # ── 모델
    "backbone"         : "EfficientNetB4",
    "freeze_backbone"  : True,       # True: backbone frozen (feature extractor 모드)
    "unfreeze_epoch"   : 10,         # 이 epoch부터 backbone 전체 unfreeze
    "rgb_embed_dim"    : 256,
    "dct_embed_dim"    : 128,
    "lm_embed_dim"     : 64,
    "fusion_hidden"    : [512, 256],
    "dropout_rate"     : 0.4,

    # ── 학습
    "batch_size"       : 32,
    "epochs"           : 50,
    "lr_init"          : 1e-3,
    "lr_min"           : 1e-6,
    "warmup_epochs"    : 3,
    "weight_decay"     : 1e-4,

    # ── Focal Loss
    "focal_alpha"      : 0.75,       # real(소수 클래스) 가중치
    "focal_gamma"      : 2.0,

    # ── 분할 (영상 ID 단위)
    "split_ratio"      : (0.7, 0.1, 0.2),
    "seed"             : 42,

    # ── 저장
    "ckpt_dir"         : "./checkpoints",
    "log_dir"          : "./logs",
}

MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


# ═══════════════════════════════════════════════════════════════
# 1. 데이터 수집
# ═══════════════════════════════════════════════════════════════

def _detect_lm_suffix(label_dir: Path) -> str:
    """_lm.npy 또는 _im.npy 중 실제 존재하는 suffix 자동 감지"""
    for suffix in ("_lm.npy", "_im.npy"):
        if next(label_dir.glob(f"*{suffix}"), None) is not None:
            log.info(f"랜드마크 suffix 감지: {suffix}")
            return suffix
    raise FileNotFoundError(f"{label_dir} 에서 _lm.npy/_im.npy 파일을 찾을 수 없음")


def collect_samples(root: str, dataset: str, lm_suffix: str = None) -> list:
    """
    processed/{dataset}/{0,1}/ 스캔 → (face, lm, dct, label) 리스트 반환
    lm_suffix=None 이면 자동 감지
    """
    ds_root = Path(root) / dataset
    samples = []
    detected_suffix = lm_suffix

    for label_str in ("0", "1"):
        label_dir = ds_root / label_str
        if not label_dir.exists():
            log.warning(f"폴더 없음: {label_dir}")
            continue

        label = int(label_str)
        if detected_suffix is None:
            detected_suffix = _detect_lm_suffix(label_dir)

        for face_path in sorted(label_dir.glob("*_face.jpg")):
            stem     = face_path.stem.replace("_face", "")
            lm_path  = label_dir / f"{stem}{detected_suffix}"
            dct_path = label_dir / f"{stem}_dct.npy"

            if lm_path.exists() and dct_path.exists():
                samples.append((str(face_path), str(lm_path), str(dct_path), label))

    n_real = sum(1 for *_, l in samples if l == 0)
    n_fake = sum(1 for *_, l in samples if l == 1)
    log.info(f"전체 수집: real={n_real:,}  fake={n_fake:,}  total={len(samples):,}")
    return samples, detected_suffix


def _video_id(face_path: str) -> str:
    """
    파일명에서 영상 ID 추출.
    전처리 코드 기준: {video_stem}_f{idx:03d}_face.jpg
    → _f000, _f010 ... 제거하면 영상 ID
    """
    stem = Path(face_path).stem.replace("_face", "")
    # _f000 ~ _f999 패턴 제거
    video_id = re.sub(r"_f\d{3}$", "", stem)
    return video_id


def video_level_split(samples: list, ratio: tuple, seed: int) -> dict:
    """
    영상 ID 단위로 train/val/test 분할.
    같은 영상의 프레임이 서로 다른 split에 들어가지 않도록 보장 (leakage 방지).
    """
    # 영상 ID 수집
    video_ids = sorted(set(_video_id(fp) for fp, *_ in samples))
    rng = np.random.default_rng(seed)
    rng.shuffle(video_ids)

    n  = len(video_ids)
    t1 = int(n * ratio[0])
    t2 = t1 + int(n * ratio[1])
    split_map = {}
    for vid in video_ids[:t1]:   split_map[vid] = "train"
    for vid in video_ids[t1:t2]: split_map[vid] = "val"
    for vid in video_ids[t2:]:   split_map[vid] = "test"

    result = {"train": [], "val": [], "test": []}
    for s in samples:
        vid = _video_id(s[0])
        result[split_map[vid]].append(s)

    for split, lst in result.items():
        nr = sum(1 for *_, l in lst if l == 0)
        nf = sum(1 for *_, l in lst if l == 1)
        log.info(f"[{split}] real={nr:,}  fake={nf:,}  total={len(lst):,}")

    return result


def undersample_fake(samples: list, ratio: float, seed: int) -> list:
    """
    fake를 real * ratio 개수로 줄임.
    ratio=3.0 → real 80K, fake 240K → 총 320K
    """
    real = [s for s in samples if s[3] == 0]
    fake = [s for s in samples if s[3] == 1]
    target = int(len(real) * ratio)

    if target >= len(fake):
        log.info("fake 언더샘플 불필요 (target >= 실제 fake 수)")
        return samples

    rng = np.random.default_rng(seed)
    fake_sub = rng.choice(len(fake), size=target, replace=False)
    fake_sampled = [fake[i] for i in fake_sub]

    result = real + fake_sampled
    log.info(
        f"언더샘플 완료: real={len(real):,}  fake={len(fake_sampled):,}"
        f"  (원래 {len(fake):,}개에서 {len(fake_sampled):,}개로)"
    )
    return result


# ═══════════════════════════════════════════════════════════════
# 2. tf.data 파이프라인
# ═══════════════════════════════════════════════════════════════

def _load_npy(path_tensor, shape):
    arr = np.load(path_tensor.numpy().decode()).astype(np.float32)
    return arr


def augment_face(face: tf.Tensor) -> tf.Tensor:
    """train 전용 증강. 입출력: float32 [0,1]"""
    face = tf.image.random_flip_left_right(face)
    face = tf.image.random_brightness(face, max_delta=0.15)
    face = tf.image.random_contrast(face, lower=0.85, upper=1.15)
    face = tf.image.random_saturation(face, lower=0.9, upper=1.1)
    # JPEG 압축 시뮬레이션 (소셜미디어 업로드 환경 모사)
    face = tf.cast(face * 255, tf.uint8)
    face = tf.image.encode_jpeg(
        face, quality=tf.random.uniform([], 70, 100, dtype=tf.int32)
    )
    face = tf.cast(tf.image.decode_jpeg(face, channels=3), tf.float32) / 255.0
    face = tf.clip_by_value(face, 0.0, 1.0)
    return face


def make_load_fn(split: str):
    do_aug = (split == "train")

    def load_fn(face_p, lm_p, dct_p, label):
        # ── 얼굴 이미지
        raw  = tf.io.read_file(face_p)
        face = tf.image.decode_jpeg(raw, channels=3)
        face = tf.cast(face, tf.float32) / 255.0
        if do_aug:
            face = augment_face(face)
        face = (face - MEAN) / STD
        face.set_shape([224, 224, 3])

        # ── DCT 맵
        dct = tf.py_function(
            lambda p: _load_npy(p, [224, 224, 3]), [dct_p], tf.float32
        )
        dct.set_shape([224, 224, 3])

        # ── 랜드마크 (픽셀 좌표 → [0,1] 정규화)
        lm = tf.py_function(
            lambda p: (_load_npy(p, [68, 2]) / 224.0), [lm_p], tf.float32
        )
        lm.set_shape([68, 2])

        return (
            {"face": face, "dct": dct, "lm": lm},
            tf.cast(label, tf.int32),
        )

    return load_fn


def build_tf_dataset(samples: list, split: str, batch_size: int, seed: int) -> tf.data.Dataset:
    face_paths = [s[0] for s in samples]
    lm_paths   = [s[1] for s in samples]
    dct_paths  = [s[2] for s in samples]
    labels     = [s[3] for s in samples]

    ds = tf.data.Dataset.from_tensor_slices(
        (face_paths, lm_paths, dct_paths, labels)
    )
    load_fn = make_load_fn(split)
    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if split == "train":
        ds = ds.shuffle(buffer_size=4096, seed=seed, reshuffle_each_iteration=True)

    drop = (split == "train")
    ds = ds.batch(batch_size, drop_remainder=drop).prefetch(tf.data.AUTOTUNE)
    return ds


# ═══════════════════════════════════════════════════════════════
# 3. 모델 아키텍처
# ═══════════════════════════════════════════════════════════════

def build_rgb_branch(img_size: int, embed_dim: int, freeze: bool) -> keras.Model:
    """
    EfficientNetB4 backbone → GlobalAveragePooling → Dense(embed_dim)
    freeze=True: backbone 가중치 고정 (feature extractor)
    """
    base = keras.applications.EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = not freeze

    inp  = keras.Input(shape=(img_size, img_size, 3), name="face")
    x    = base(inp, training=False)     # BN은 항상 inference 모드
    x    = layers.GlobalAveragePooling2D()(x)
    x    = layers.Dense(embed_dim, use_bias=False)(x)
    x    = layers.BatchNormalization()(x)
    x    = layers.Activation("relu")(x)
    return keras.Model(inp, x, name="rgb_branch")


def build_dct_branch(img_size: int, embed_dim: int) -> keras.Model:
    """
    DCT 맵 전용 경량 CNN.
    ImageNet pretrain 없음 — 주파수 도메인은 자연 이미지와 다름.
    3개 Conv block → GAP → Dense(embed_dim)
    """
    def conv_block(x, filters, stride=1):
        x = layers.Conv2D(filters, 3, stride, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    inp = keras.Input(shape=(img_size, img_size, 3), name="dct")
    x   = conv_block(inp, 32, stride=2)   # → 112
    x   = conv_block(x,  64, stride=2)   # → 56
    x   = conv_block(x, 128, stride=2)   # → 28
    x   = conv_block(x, 256, stride=2)   # → 14
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(embed_dim, use_bias=False)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    return keras.Model(inp, x, name="dct_branch")


def build_lm_branch(embed_dim: int, dropout: float) -> keras.Model:
    """
    랜드마크 68점 (68,2) → Flatten(136) → MLP → Dense(embed_dim)
    기하학적 비일관성(눈·코·입 비율, 대칭성 위반) 검출
    """
    inp = keras.Input(shape=(68, 2), name="lm")
    x   = layers.Flatten()(inp)                        # (136,)
    x   = layers.Dense(256, use_bias=False)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    x   = layers.Dropout(dropout)(x)
    x   = layers.Dense(128, use_bias=False)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    x   = layers.Dropout(dropout * 0.5)(x)
    x   = layers.Dense(embed_dim, use_bias=False)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    return keras.Model(inp, x, name="lm_branch")


def build_model(cfg: dict) -> keras.Model:
    """
    3-stream 모델:
        rgb_embed (256) + dct_embed (128) + lm_embed (64)
        → Concat (448) → FC (512→256) → sigmoid
    """
    img_size = cfg["img_size"]

    rgb_branch = build_rgb_branch(img_size, cfg["rgb_embed_dim"], cfg["freeze_backbone"])
    dct_branch = build_dct_branch(img_size, cfg["dct_embed_dim"])
    lm_branch  = build_lm_branch(cfg["lm_embed_dim"], cfg["dropout_rate"])

    # ── 입력
    inp_face = keras.Input(shape=(img_size, img_size, 3), name="face")
    inp_dct  = keras.Input(shape=(img_size, img_size, 3), name="dct")
    inp_lm   = keras.Input(shape=(68, 2),                 name="lm")

    # ── 각 스트림 통과
    rgb_feat = rgb_branch(inp_face)
    dct_feat = dct_branch(inp_dct)
    lm_feat  = lm_branch(inp_lm)

    # ── Late Fusion
    x = layers.Concatenate(name="concat_embeddings")([rgb_feat, dct_feat, lm_feat])

    for units in cfg["fusion_hidden"]:
        x = layers.Dense(units, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(cfg["dropout_rate"])(x)

    # dtype=float32 강제 (mixed_precision 사용 시 logit은 float32 필요)
    out = layers.Dense(1, name="logit")(x)
    out = layers.Activation("sigmoid", dtype="float32", name="prob")(out)

    model = keras.Model(
        inputs={"face": inp_face, "dct": inp_dct, "lm": inp_lm},
        outputs=out,
        name="deepfake_3stream",
    )
    model.summary(line_length=100)
    return model


# ═══════════════════════════════════════════════════════════════
# 4. Focal Loss
# ═══════════════════════════════════════════════════════════════

class FocalLoss(keras.losses.Loss):
    """
    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    α: real(소수 클래스) 가중치  → fake 가중치 = 1 - α
    γ: hard example 집중도 (γ=0 → 일반 BCE, γ=2 → 논문 기본값)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # α_t: real(y=0)이면 self.alpha, fake(y=1)이면 1-self.alpha
        # 주의: real=0 (소수) 에 높은 alpha 부여
        alpha_t = y_true * (1.0 - self.alpha) + (1.0 - y_true) * self.alpha

        # p_t
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        fl = -alpha_t * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)
        return tf.reduce_mean(fl)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "gamma": self.gamma})
        return cfg


# ═══════════════════════════════════════════════════════════════
# 5. 스케줄러 — Warmup + Cosine Decay
# ═══════════════════════════════════════════════════════════════

class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup → Cosine annealing
    warmup_steps 동안 0 → lr_init 선형 증가
    이후 lr_init → lr_min 코사인 감소
    """
    def __init__(self, lr_init, lr_min, warmup_steps, total_steps):
        super().__init__()
        self.lr_init      = float(lr_init)
        self.lr_min       = float(lr_min)
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)

    def __call__(self, step):
        step     = tf.cast(step, tf.float32)
        warmup   = self.lr_init * (step / self.warmup_steps)
        cosine   = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (
            1.0 + tf.cos(
                np.pi * (step - self.warmup_steps) /
                (self.total_steps - self.warmup_steps)
            )
        )
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {
            "lr_init"      : self.lr_init,
            "lr_min"       : self.lr_min,
            "warmup_steps" : self.warmup_steps,
            "total_steps"  : self.total_steps,
        }


# ═══════════════════════════════════════════════════════════════
# 6. Callbacks
# ═══════════════════════════════════════════════════════════════

class BackboneUnfreezeCallback(keras.callbacks.Callback):
    """
    unfreeze_epoch에 도달하면 rgb_branch 내 EfficientNet backbone unfreeze.
    이 시점에 학습률도 1/10으로 낮춤 (fine-tuning용).
    """
    def __init__(self, unfreeze_epoch: int, lr_scale: float = 0.1):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.lr_scale       = lr_scale
        self._unfrozen      = False

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.unfreeze_epoch and not self._unfrozen:
            for layer in self.model.layers:
                if hasattr(layer, "trainable"):
                    layer.trainable = True
            # 현재 lr 가져와서 축소
            old_lr = float(self.model.optimizer.learning_rate)
            new_lr = old_lr * self.lr_scale
            self.model.optimizer.learning_rate.assign(new_lr)
            self._unfrozen = True
            log.info(
                f"Epoch {epoch}: backbone unfreeze. "
                f"lr {old_lr:.2e} → {new_lr:.2e}"
            )


def build_callbacks(cfg: dict, steps_per_epoch: int) -> list:
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    ckpt_path = os.path.join(
        cfg["ckpt_dir"],
        "best_auc_epoch{epoch:03d}_val{val_auc:.4f}.keras"
    )

    callbacks = [
        # ── 최고 val_auc 기준 저장
        keras.callbacks.ModelCheckpoint(
            filepath         = ckpt_path,
            monitor          = "val_auc",
            mode             = "max",
            save_best_only   = True,
            save_weights_only= False,
            verbose          = 1,
        ),
        # ── 조기 종료 (val_auc 기준, patience=8)
        keras.callbacks.EarlyStopping(
            monitor              = "val_auc",
            mode                 = "max",
            patience             = 8,
            restore_best_weights = True,
            verbose              = 1,
        ),
        # ── TensorBoard
        keras.callbacks.TensorBoard(
            log_dir          = cfg["log_dir"],
            histogram_freq   = 0,
            update_freq      = "epoch",
        ),
        # ── CSV 로거
        keras.callbacks.CSVLogger(
            filename = f"training_log_{_ts}.csv",
            append   = False,
        ),
        # ── Backbone Unfreeze
        BackboneUnfreezeCallback(
            unfreeze_epoch = cfg["unfreeze_epoch"],
            lr_scale       = 0.1,
        ),
        # ── LR 로깅 (디버깅용)
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: log.info(
                f"Epoch {epoch+1:3d} │ "
                f"loss={logs.get('loss',0):.4f}  "
                f"auc={logs.get('auc',0):.4f}  "
                f"val_loss={logs.get('val_loss',0):.4f}  "
                f"val_auc={logs.get('val_auc',0):.4f}  "
                f"lr={float(model.optimizer.learning_rate):.2e}"
            )
        ),
    ]
    return callbacks


# ═══════════════════════════════════════════════════════════════
# 7. 평가 — 상세 지표
# ═══════════════════════════════════════════════════════════════

def evaluate_model(model: keras.Model, test_ds: tf.data.Dataset):
    """
    테스트셋에서 상세 지표 출력:
    AUC, Accuracy, Precision, Recall, F1, Confusion Matrix
    """
    y_true_all, y_pred_all = [], []

    for batch_inputs, batch_labels in test_ds:
        preds = model(batch_inputs, training=False)
        y_pred_all.append(preds.numpy().flatten())
        y_true_all.append(batch_labels.numpy().flatten())

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_pred_all)
    y_pred = (y_prob >= 0.5).astype(int)

    # sklearn 없이 TF metric으로 계산
    auc_m = keras.metrics.AUC()
    auc_m.update_state(y_true, y_prob)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    acc       = (tp + tn) / len(y_true)

    log.info("=" * 60)
    log.info("  테스트셋 평가 결과")
    log.info("=" * 60)
    log.info(f"  AUC-ROC   : {auc_m.result().numpy():.4f}")
    log.info(f"  Accuracy  : {acc:.4f}")
    log.info(f"  Precision : {precision:.4f}  (fake 탐지 정밀도)")
    log.info(f"  Recall    : {recall:.4f}  (fake 탐지 재현율)")
    log.info(f"  F1-Score  : {f1:.4f}")
    log.info(f"  Confusion Matrix:")
    log.info(f"            Pred Real  Pred Fake")
    log.info(f"  True Real  {tn:8,}   {fp:8,}")
    log.info(f"  True Fake  {fn:8,}   {tp:8,}")
    log.info("=" * 60)

    return {
        "auc": auc_m.result().numpy(),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


# ═══════════════════════════════════════════════════════════════
# 8. 데이터 로더 Sanity Check
# ═══════════════════════════════════════════════════════════════

def sanity_check(ds: tf.data.Dataset):
    """배치 1개 뽑아서 shape·dtype·값 범위 확인"""
    log.info("─" * 50)
    log.info("Sanity check: 배치 1개 로딩 중...")
    for inputs, labels in ds.take(1):
        face = inputs["face"]
        dct  = inputs["dct"]
        lm   = inputs["lm"]
        log.info(f"  face     shape={face.shape}  dtype={face.dtype}  "
                 f"min={float(tf.reduce_min(face)):.3f}  max={float(tf.reduce_max(face)):.3f}")
        log.info(f"  dct      shape={dct.shape}   dtype={dct.dtype}   "
                 f"min={float(tf.reduce_min(dct)):.3f}  max={float(tf.reduce_max(dct)):.3f}")
        log.info(f"  lm       shape={lm.shape}    dtype={lm.dtype}    "
                 f"min={float(tf.reduce_min(lm)):.4f}  max={float(tf.reduce_max(lm)):.4f}")
        log.info(f"  labels   shape={labels.shape}  values={labels.numpy()}")
        real_cnt = int(tf.reduce_sum(tf.cast(labels == 0, tf.int32)))
        fake_cnt = int(tf.reduce_sum(tf.cast(labels == 1, tf.int32)))
        log.info(f"  배치 내 real={real_cnt}  fake={fake_cnt}")
    log.info("Sanity check 완료")
    log.info("─" * 50)


# ═══════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    cfg  = CFG
    seed = cfg["seed"]
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # ── 1. 데이터 수집
    log.info("=" * 60)
    log.info("STEP 1: 데이터 수집")
    log.info("=" * 60)
    samples, lm_suffix = collect_samples(
        cfg["processed_root"], cfg["dataset"], cfg["lm_suffix"]
    )

    # ── 2. 영상 ID 단위 분할
    log.info("=" * 60)
    log.info("STEP 2: 영상 ID 단위 분할 (leakage 방지)")
    log.info("=" * 60)
    splits = video_level_split(samples, cfg["split_ratio"], seed)

    # ── 3. 불균형 처리 (train만 undersample)
    log.info("=" * 60)
    log.info("STEP 3: 클래스 불균형 처리")
    log.info("=" * 60)
    if cfg["undersample_ratio"] is not None:
        splits["train"] = undersample_fake(
            splits["train"], cfg["undersample_ratio"], seed
        )

    # ── 4. tf.data 빌드
    log.info("=" * 60)
    log.info("STEP 4: tf.data 파이프라인 빌드")
    log.info("=" * 60)
    bs = cfg["batch_size"]
    train_ds = build_tf_dataset(splits["train"], "train", bs, seed)
    val_ds   = build_tf_dataset(splits["val"],   "val",   bs, seed)
    test_ds  = build_tf_dataset(splits["test"],  "test",  bs, seed)

    # Sanity check
    sanity_check(train_ds)

    # ── 5. 모델 빌드
    log.info("=" * 60)
    log.info("STEP 5: 모델 빌드")
    log.info("=" * 60)
    global model
    model = build_model(cfg)

    # ── 6. 옵티마이저 & 컴파일
    log.info("=" * 60)
    log.info("STEP 6: 컴파일")
    log.info("=" * 60)
    steps_per_epoch = len(splits["train"]) // bs
    total_steps     = steps_per_epoch * cfg["epochs"]
    warmup_steps    = steps_per_epoch * cfg["warmup_epochs"]

    schedule = WarmupCosineDecay(
        lr_init      = cfg["lr_init"],
        lr_min       = cfg["lr_min"],
        warmup_steps = warmup_steps,
        total_steps  = total_steps,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate  = schedule,
        weight_decay   = cfg["weight_decay"],
    )

    model.compile(
        optimizer = optimizer,
        loss      = FocalLoss(alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"]),
        metrics   = [
            keras.metrics.AUC(name="auc"),
            keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
            keras.metrics.Precision(name="precision", thresholds=0.5),
            keras.metrics.Recall(name="recall", thresholds=0.5),
        ],
    )

    log.info(f"steps_per_epoch = {steps_per_epoch:,}")
    log.info(f"total_steps     = {total_steps:,}")
    log.info(f"warmup_steps    = {warmup_steps:,}")

    # ── 7. 학습
    log.info("=" * 60)
    log.info("STEP 7: 학습 시작")
    log.info("=" * 60)
    t0 = time.time()

    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = cfg["epochs"],
        callbacks       = build_callbacks(cfg, steps_per_epoch),
        verbose         = 1,
    )

    elapsed = time.time() - t0
    log.info(f"학습 완료 — {elapsed/3600:.1f}h ({elapsed:.0f}s)")

    # ── 8. 테스트 평가
    log.info("=" * 60)
    log.info("STEP 8: 테스트셋 최종 평가")
    log.info("=" * 60)
    results = evaluate_model(model, test_ds)

    # ── 9. 최종 모델 저장
    final_path = os.path.join(cfg["ckpt_dir"], f"final_model_{_ts}.keras")
    model.save(final_path)
    log.info(f"최종 모델 저장: {final_path}")

    return history, results


if __name__ == "__main__":
    history, results = main()