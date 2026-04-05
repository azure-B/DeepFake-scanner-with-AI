"""
Deepfake Detection v3 — 7-Head Cross-Attention + Uncertainty Weighting
═══════════════════════════════════════════════════════════════════════
Architecture:
  RGB  : CLIP ViT-B/16  → 256-dim embedding
  DCT  : FAD (6ch)      → 128-dim embedding
  LM   : Geometric MLP  →  32-dim embedding

  7 Fusion Heads (Shared Branch Weights):
    h_rgb     : RGB only
    h_dct     : DCT only
    h_lm      : LM  only
    h_rgb_dct : Cross-Attention(RGB ↔ DCT)
    h_rgb_lm  : Cross-Attention(RGB ↔ LM)
    h_dct_lm  : Cross-Attention(DCT ↔ LM)
    h_all ★  : Cross-Attention(RGB ↔ DCT ↔ LM)  ← 메인

  Loss : Uncertainty-Weighted Focal Loss (Kendall & Gal 2017)
         L = Σ_i [ 0.5 * exp(-σ_i) * FL_i + 0.5 * σ_i ]
"""

import os, sys, re, time, logging
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision

# ── 로깅 ────────────────────────────────────────────────────────
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

# ── CLIP 임포트 (없으면 EfficientNetB4 fallback) ────────────────
try:
    from transformers import TFCLIPVisionModel
    CLIP_AVAILABLE = True
    log.info("CLIP ViT-B/16 사용 가능")
except ImportError:
    CLIP_AVAILABLE = False
    log.warning("transformers 미설치 → EfficientNetB4 fallback")
    log.warning("설치: pip install transformers --break-system-packages")

# ── GPU ─────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
log.info(f"GPU {len(gpus)}개 감지")

mixed_precision.set_global_policy("mixed_float16")
log.info(f"compute dtype: {mixed_precision.global_policy().compute_dtype}")

# DCT 저주파 마스크 전역 상수 (매 샘플마다 생성 방지)
def _create_dct_mask():
    mask = np.zeros((224, 224), dtype=np.float32)
    for i in range(224):
        for j in range(224):
            if i + j < 224:
                mask[i, j] = 1.0
    return mask

# 지연 초기화를 버리고, 코드 실행 즉시 전역 상수로 쾅 박아버립니다.
_DCT_LOW_MASK = tf.constant(_create_dct_mask())


# ═══════════════════════════════════════════════════════════════
# 0. CONFIG
# ═══════════════════════════════════════════════════════════════
CFG = {
    # ── 데이터
    "processed_root"   : "./processed",
    "train_datasets"   : ["ffpp", "dff", "hidf"],
    "test_datasets"    : ["celebdf", "redface"],
    "undersample_ratio": 1.0,

    # ── 이미지
    "img_size"         : 224,

    # ── 모델
    "freeze_backbone"  : True,
    "unfreeze_epoch"   : 15,
    "rgb_embed_dim"    : 256,
    "dct_embed_dim"    : 128,
    "lm_embed_dim"     : 32,
    "attn_dim"         : 128,       # Cross-Attention 공통 차원
    "head_hidden"      : 64,        # 각 Head Dense 크기
    "dropout_rate"     : 0.4,

    # ── 학습
    "batch_size"       : 16,        # CLIP ViT 메모리 때문에 16
    "epochs"           : 50,
    "lr_init"          : 3e-4,
    "lr_min"           : 1e-7,
    "warmup_epochs"    : 5,
    "weight_decay"     : 3e-4,
    "aux_weight": 0.4,

    # ── Focal Loss
    "focal_alpha"      : 0.5,
    "focal_gamma"      : 2.0,

    # ── 분할
    "split_ratio"      : (0.7, 0.1, 0.2),
    "seed"             : 42,

    # ── 저장
    "ckpt_dir"         : "./checkpoints",
}

# CLIP 전용 정규화 (ImageNet과 다름)
MEAN = tf.constant([0.48145466, 0.4578275,  0.40821073], dtype=tf.float32)
STD  = tf.constant([0.26862954, 0.26130258, 0.27577711], dtype=tf.float32)

LM_FEAT_DIM = 34
HEAD_NAMES  = ["rgb", "dct", "lm", "rgb_dct", "rgb_lm", "dct_lm", "all"]
AUX_HEAD_NAMES = ["aux_lf", "aux_hf"]


# ═══════════════════════════════════════════════════════════════
# 1. 기하학적 랜드마크 피처 (34차원)
# ═══════════════════════════════════════════════════════════════

def extract_geometric_features(lm: np.ndarray) -> np.ndarray:
    """
    (68, 2) 절대좌표 → (34,) 기하학적 비율 피처
    크기/위치 불변 → 도메인 일반화에 유리
    """
    feats = []
    face_w = float(np.linalg.norm(lm[16] - lm[0])) + 1e-8
    face_h = float(np.linalg.norm(lm[8]  - lm[27])) + 1e-8

    # 1. 얼굴 종횡비
    feats.append(face_h / face_w)

    # 2. 눈 관련 (6개)
    eye_dist = float(np.linalg.norm(lm[42] - lm[36]))
    eye_l_h  = float(np.linalg.norm(lm[41] - lm[37]))
    eye_r_h  = float(np.linalg.norm(lm[47] - lm[43]))
    eye_l_w  = float(np.linalg.norm(lm[39] - lm[36]))
    eye_r_w  = float(np.linalg.norm(lm[45] - lm[42]))
    feats += [eye_dist/face_w, eye_l_h/(eye_r_h+1e-8),
              eye_l_w/(eye_r_w+1e-8), eye_l_h/(eye_l_w+1e-8),
              eye_r_h/(eye_r_w+1e-8), eye_dist/face_h]

    # 3. 눈썹 관련 (4개)
    brow_l_h = float(np.linalg.norm(lm[19] - lm[38]))
    brow_r_h = float(np.linalg.norm(lm[24] - lm[44]))
    brow_l_w = float(np.linalg.norm(lm[21] - lm[17]))
    brow_r_w = float(np.linalg.norm(lm[26] - lm[22]))
    feats += [brow_l_h/face_h, brow_r_h/face_h,
              brow_l_h/(brow_r_h+1e-8), brow_l_w/(brow_r_w+1e-8)]

    # 4. 코 관련 (4개)
    nose_h   = float(np.linalg.norm(lm[33] - lm[27]))
    nose_w   = float(np.linalg.norm(lm[35] - lm[31]))
    nose_tip = float(np.linalg.norm(lm[33] - lm[30]))
    feats += [nose_h/face_h, nose_w/face_w,
              nose_h/(nose_w+1e-8), nose_tip/(nose_h+1e-8)]

    # 5. 입 관련 (6개)
    mouth_w = float(np.linalg.norm(lm[54] - lm[48]))
    mouth_h = float(np.linalg.norm(lm[57] - lm[51]))
    upper_h = float(np.linalg.norm(lm[51] - lm[62]))
    lower_h = float(np.linalg.norm(lm[66] - lm[57]))
    mouth_l = float(np.linalg.norm(lm[48] - lm[8]))
    mouth_r = float(np.linalg.norm(lm[54] - lm[8]))
    feats += [mouth_w/face_w, mouth_h/face_h,
              mouth_w/(mouth_h+1e-8), upper_h/(lower_h+1e-8),
              mouth_l/(mouth_r+1e-8), mouth_w/(eye_dist+1e-8)]

    # 6. 턱선 대칭 (4개)
    jaw_l  = float(np.linalg.norm(lm[4]  - lm[0]))
    jaw_r  = float(np.linalg.norm(lm[12] - lm[16]))
    chin_l = float(np.linalg.norm(lm[8]  - lm[4]))
    chin_r = float(np.linalg.norm(lm[8]  - lm[12]))
    feats += [jaw_l/(jaw_r+1e-8), chin_l/(chin_r+1e-8),
              (jaw_l+jaw_r)/face_w, (chin_l+chin_r)/face_h]

    # 7. 눈-코-입 삼각형 (4개)
    eye_mid     = (lm[39] + lm[45]) / 2
    nose_tip_pt = lm[33]
    mouth_mid   = (lm[48] + lm[54]) / 2
    tri_h   = float(np.linalg.norm(nose_tip_pt - eye_mid))
    tri_b   = float(np.linalg.norm(mouth_mid   - nose_tip_pt))
    tri_asy = float(np.linalg.norm(eye_mid      - nose_tip_pt))
    feats += [tri_h/face_h, tri_b/face_h,
              tri_h/(tri_b+1e-8), tri_asy/face_w]

    # 8. 좌우 전체 대칭 (5개)
    for l, r in [(0,16),(1,15),(2,14),(3,13),(4,12)]:
        cx   = (lm[0][0] + lm[16][0]) / 2
        diff = abs(abs(lm[l][0]-cx) - abs(lm[r][0]-cx))
        feats.append(diff / face_w)

    return np.array(feats[:LM_FEAT_DIM], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# 2. 데이터 수집 & 분할
# ═══════════════════════════════════════════════════════════════

def collect_samples(processed_root: str, dataset_keys: list) -> list:
    root    = Path(processed_root)
    samples = []
    for ds_name in dataset_keys:
        ds_dir = root / ds_name
        if not ds_dir.exists():
            log.warning(f"폴더 없음: {ds_dir} — 스킵")
            continue
        ds_samples = []
        for label_str in ("0", "1"):
            label_dir = ds_dir / label_str
            if not label_dir.exists():
                continue
            label     = int(label_str)
            lm_suffix = None
            for s in ("_lm.npy", "_im.npy"):
                if next(label_dir.glob(f"*{s}"), None):
                    lm_suffix = s
                    break
            if lm_suffix is None:
                log.warning(f"랜드마크 없음: {label_dir}")
                continue
            for face_path in sorted(label_dir.glob("*_face.jpg")):
                stem    = face_path.stem.replace("_face", "")
                lm_path = label_dir / f"{stem}{lm_suffix}"
                if lm_path.exists():
                    ds_samples.append((str(face_path), str(lm_path), label))
        nr = sum(1 for *_, l in ds_samples if l == 0)
        nf = sum(1 for *_, l in ds_samples if l == 1)
        log.info(f"[{ds_name}] real={nr:,}  fake={nf:,}")
        samples.extend(ds_samples)
    nr = sum(1 for *_, l in samples if l == 0)
    nf = sum(1 for *_, l in samples if l == 1)
    log.info(f"전체: real={nr:,}  fake={nf:,}  total={len(samples):,}")
    return samples


def _video_id(face_path: str) -> str:
    stem = Path(face_path).stem.replace("_face", "")
    return re.sub(r"_f\d{3}$", "", stem)


def video_level_split(samples, ratio, seed):
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
        result[split_map[_video_id(s[0])]].append(s)
    for sp, lst in result.items():
        nr = sum(1 for *_, l in lst if l == 0)
        nf = sum(1 for *_, l in lst if l == 1)
        log.info(f"[{sp}] real={nr:,}  fake={nf:,}  total={len(lst):,}")
    return result


def undersample_fake(samples, ratio, seed):
    if ratio is None:  # ← 추가
        log.info("언더샘플 스킵 (ratio=None)")
        return samples

    real   = [s for s in samples if s[2] == 0]
    fake   = [s for s in samples if s[2] == 1]
    target = int(len(real) * ratio)
    if target >= len(fake):
        log.info("fake 언더샘플 불필요")
        return samples
    rng          = np.random.default_rng(seed)
    fake_sampled = [fake[i] for i in rng.choice(len(fake), target, replace=False)]
    log.info(f"언더샘플: real={len(real):,}  fake={len(fake_sampled):,}")
    combined = real + fake_sampled
    rng.shuffle(combined)
    return combined


# ═══════════════════════════════════════════════════════════════
# 3. tf.data 파이프라인
# ═══════════════════════════════════════════════════════════════

def _load_lm_geom(path_tensor, flipped=False) -> np.ndarray:
    lm = np.load(path_tensor.numpy().decode()).astype(np.float32)
    if flipped:
        img_w    = lm[:, 0].max() + lm[:, 0].min()
        lm       = lm.copy()
        lm[:, 0] = img_w - lm[:, 0]
    return extract_geometric_features(lm)


def compute_dct_tf(face: tf.Tensor) -> tf.Tensor:
    channels = tf.unstack(face, axis=-1)
    low_chs, high_chs = [], []
    mask = _DCT_LOW_MASK  # ← 전역 상수 재사용
    for ch in channels:
        d     = tf.signal.dct(ch, type=2, norm="ortho")
        d     = tf.signal.dct(tf.transpose(d), type=2, norm="ortho")
        d     = tf.transpose(d)
        d     = tf.math.log(tf.abs(d) + 1e-8)
        d_min = tf.reduce_min(d)
        d_max = tf.reduce_max(d)
        d     = (d - d_min) / (d_max - d_min + 1e-8)
        low   = d * mask
        high  = d - low
        low_chs.append(low)
        high_chs.append(high)
    return tf.concat([
        tf.stack(low_chs,  axis=-1),
        tf.stack(high_chs, axis=-1),
    ], axis=-1)

def augment_face_no_flip(face: tf.Tensor) -> tf.Tensor:
    """flip을 제외한 증강 — flip은 LM과 동기화하여 make_load_fn에서 처리"""
    face = tf.image.random_brightness(face, max_delta=0.2)
    face = tf.image.random_contrast(face, lower=0.7, upper=1.3)
    face = tf.image.random_saturation(face, lower=0.7, upper=1.3)
    face = tf.cast(face * 255, tf.uint8)

    def jpeg_compress(img):
        q       = int(np.random.randint(75, 95))
        encoded = tf.image.encode_jpeg(img, quality=q)
        return tf.image.decode_jpeg(encoded, channels=3)

    face = tf.py_function(jpeg_compress, [face], tf.uint8)
    face.set_shape([224, 224, 3])
    face = tf.cast(face, tf.float32) / 255.0

    def random_blur(img):
        if np.random.rand() < 0.3:
            img = tf.expand_dims(img, 0)
            img = tf.nn.avg_pool2d(img, ksize=3, strides=1, padding="SAME")
            img = tf.squeeze(img, 0)
        return img

    face = tf.py_function(random_blur, [face], tf.float32)
    face.set_shape([224, 224, 3])
    return tf.clip_by_value(face, 0.0, 1.0)


def make_load_fn(split: str):
    do_aug = (split == "train")

    def load_fn(face_p, lm_p, label):
        face = tf.image.decode_jpeg(tf.io.read_file(face_p), channels=3)
        face = tf.cast(face, tf.float32) / 255.0

        # flip_flag: 0.0 or 1.0
        flip_flag = (
            tf.cast(tf.random.uniform([]) > 0.5, tf.float32)
            if do_aug else tf.constant(0.0)
        )

        if do_aug:
            face = augment_face_no_flip(face)
            face = tf.cond(
                flip_flag > 0.5,
                lambda: tf.image.flip_left_right(face),
                lambda: face,
            )

        dct = compute_dct_tf(face)
        dct.set_shape([224, 224, 6])

        face = (face - MEAN) / STD
        face.set_shape([224, 224, 3])

        lm = tf.py_function(
            lambda p, f: _load_lm_geom(p, flipped=bool(f.numpy() > 0.5)),
            [lm_p, flip_flag],
            tf.float32,
        )
        lm.set_shape([LM_FEAT_DIM])

        return (
            {"face": face, "dct": dct, "lm": lm},
            tf.expand_dims(tf.cast(label, tf.float32), axis=-1),
        )

    return load_fn


def mixup_batch(inputs, labels, alpha=0.2):
    bs  = tf.shape(inputs["face"])[0]
    lam = tf.random.uniform([], 0.0, alpha)
    lam = tf.maximum(lam, 1.0 - lam)
    idx = tf.random.shuffle(tf.range(bs))

    mixed = {
        "face": lam * inputs["face"] + (1-lam) * tf.gather(inputs["face"], idx),
        "dct" : lam * inputs["dct"]  + (1-lam) * tf.gather(inputs["dct"],  idx),
        "lm"  : inputs["lm"],   # ← LM은 기하학적 피처라 MixUp 안 함
    }
    labels_f   = tf.cast(labels, tf.float32)
    labels_mix = lam * labels_f + (1-lam) * tf.cast(tf.gather(labels, idx), tf.float32)
    return mixed, labels_mix

def build_tf_dataset(samples, split, batch_size, seed):
    face_paths = [s[0] for s in samples]
    lm_paths   = [s[1] for s in samples]
    labels     = [s[2] for s in samples]

    ds = tf.data.Dataset.from_tensor_slices((face_paths, lm_paths, labels))
    if split == "train":
        ds = ds.shuffle(buffer_size=10000, seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(make_load_fn(split), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=(split == "train"))

    if split == "train":
        ds = ds.map(mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


# ═══════════════════════════════════════════════════════════════
# 4. 브랜치 아키텍처
# ═══════════════════════════════════════════════════════════════

# ── 4-A. CLIP ViT-B/16 (RGB Branch) ────────────────────────────

if CLIP_AVAILABLE:
    class CLIPVisionLayer(keras.layers.Layer):
        def __init__(self, freeze: bool = True, **kwargs):
            super().__init__(**kwargs)
            self.clip = TFCLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch16",
                from_pt = True,
            )
            self.clip.trainable = not freeze

        def call(self, inputs, training=False):
            pv  = tf.transpose(inputs, [0, 3, 1, 2])
            out = self.clip(pixel_values=pv, training=training)
            return out.pooler_output

        def set_freeze(self, freeze: bool):
            self.clip.trainable = not freeze
            self.trainable      = not freeze

        def get_config(self):
            return super().get_config()

def build_rgb_branch(img_size: int, embed_dim: int, freeze: bool) -> keras.Model:
    inp       = keras.Input(shape=(img_size, img_size, 3), name="face")
    clip_feat = CLIPVisionLayer(freeze=freeze, name="clip_layer")(inp)  # (B,768)
    x         = layers.Dense(embed_dim, use_bias=False, name="rgb_proj")(clip_feat)
    x         = layers.LayerNormalization(name="rgb_ln")(x)
    x         = layers.Activation("relu", name="rgb_relu")(x)
    return keras.Model(inp, x, name="rgb_branch")


def build_rgb_branch_fallback(img_size: int, embed_dim: int, freeze: bool) -> keras.Model:
    """CLIP 미설치 시 EfficientNetB4 fallback"""
    base = keras.applications.EfficientNetB4(
        include_top=False, weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = not freeze
    inp = keras.Input(shape=(img_size, img_size, 3), name="face")
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(embed_dim, use_bias=False)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    return keras.Model(inp, x, name="rgb_branch")


# ── 4-B. FAD (Frequency-Aware Discriminator, DCT Branch) ───────

def build_fad_branch(img_size: int, embed_dim: int) -> keras.Model:
    def cb(x, f, s, name):
        x = layers.Conv2D(f, 3, s, padding="same", use_bias=False, name=f"{name}_c")(x)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        return layers.Activation("relu", name=f"{name}_r")(x)

    inp     = keras.Input(shape=(img_size, img_size, 6), name="dct")
    low_inp = inp[:, :, :, :3]
    hig_inp = inp[:, :, :, 3:]

    lf = cb(low_inp, 32,  2, "lf1")
    lf = cb(lf,      64,  2, "lf2")
    lf = cb(lf,      128, 2, "lf3")
    lf_gap = layers.GlobalAveragePooling2D(name="lf_gap")(lf)  # (B,128)

    hf = cb(hig_inp, 32,  2, "hf1")
    hf = cb(hf,      64,  2, "hf2")
    hf = cb(hf,      128, 2, "hf3")
    hf = cb(hf,      256, 2, "hf4")
    hf = cb(hf,      256, 1, "hf5")
    hf_gap = layers.GlobalAveragePooling2D(name="hf_gap")(hf)  # (B,256)

    x = layers.Concatenate(name="fad_cat")([lf_gap, hf_gap])
    x = layers.Dense(embed_dim, use_bias=False, name="fad_proj")(x)
    x = layers.BatchNormalization(name="fad_bn")(x)
    x = layers.Activation("relu", name="fad_relu")(x)

    # ← 수정: 출력 3개 (메인 임베딩 + 중간 피처 2개)
    return keras.Model(inp, [x, lf_gap, hf_gap], name="fad_branch")

# ── 4-C. Geometric Landmark MLP ────────────────────────────────

def build_lm_branch(feat_dim: int, embed_dim: int, dropout: float) -> keras.Model:
    inp = keras.Input(shape=(feat_dim,), name="lm")
    x   = layers.Dense(128, use_bias=False)(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    x   = layers.Dropout(dropout)(x)
    x   = layers.Dense(64, use_bias=False)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    x   = layers.Dropout(dropout * 0.5)(x)
    x   = layers.Dense(embed_dim, use_bias=False)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    return keras.Model(inp, x, name="lm_branch")


# ═══════════════════════════════════════════════════════════════
# 5. Cross-Attention Fusion
# ═══════════════════════════════════════════════════════════════

def project_feat(feat: tf.Tensor, dim: int, name: str) -> tf.Tensor:
    """각 브랜치를 공통 ATTN_DIM으로 프로젝션 + LayerNorm"""
    x = layers.Dense(dim, use_bias=False, name=f"{name}_proj")(feat)
    x = layers.LayerNormalization(name=f"{name}_pln")(x)
    return x


def cross_attention_2(feat_a, feat_b, dim, name):
    """
    2-way Cross-Attention
    A → B 방향 & B → A 방향 각각 계산 후 잔차 연결 + 합산

    핵심 수식:
      Attn(Q,K,V) = softmax(QKᵀ / √d) · V
    """
    # (B,dim) → (B,1,dim) : 시퀀스 차원 추가
    a = tf.expand_dims(feat_a, 1)
    b = tf.expand_dims(feat_b, 1)

    # A가 B를 참조
    a_cross = layers.MultiHeadAttention(
        num_heads=4, key_dim=dim//4, name=f"{name}_mha_ab"
    )(query=a, key=b, value=b)             # (B,1,dim)

    # B가 A를 참조
    b_cross = layers.MultiHeadAttention(
        num_heads=4, key_dim=dim//4, name=f"{name}_mha_ba"
    )(query=b, key=a, value=a)             # (B,1,dim)

    # 잔차 + LayerNorm
    a_out = layers.LayerNormalization(name=f"{name}_lna")(
        feat_a + tf.squeeze(a_cross, 1)
    )
    b_out = layers.LayerNormalization(name=f"{name}_lnb")(
        feat_b + tf.squeeze(b_cross, 1)
    )
    fused = layers.Add(name=f"{name}_add")([a_out, b_out])
    fused = layers.Dense(dim, use_bias=False, name=f"{name}_out")(fused)
    fused = layers.LayerNormalization(name=f"{name}_oln")(fused)
    return fused                            # (B,dim)


class CrossAttention3Way(keras.layers.Layer):
    def __init__(self, dim, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        _n = name or "ca3"
        self.mha = layers.MultiHeadAttention(num_heads=4, key_dim=dim // 4, name=f"{_n}_mha")
        self.ln1 = layers.LayerNormalization(name=f"{_n}_ln1")  # ← name → _n
        self.dense = layers.Dense(dim, use_bias=False, name=f"{_n}_out")  # ← name → _n
        self.ln2 = layers.LayerNormalization(name=f"{_n}_ln2")  # ← name → _n

    def build(self, input_shape):  # ⬅️ 오타 수정: (self, input_shape) 로 닫기
        # 3개의 모달리티(RGB, DCT, LM)를 구분하는 학습 가능한 이름표(Embedding)
        self.modality_emb = self.add_weight(
            shape=(1, 3, self.dim),
            initializer="random_normal",
            trainable=True,
            name="modality_embedding"
        )
        super().build(input_shape)

    def call(self, inputs):  # ⬅️ Keras 규칙에 맞게 하나의 리스트(inputs)로 받음
        feat_r, feat_d, feat_l = inputs

        # 1. 3개 토큰 스택: (B, 3, dim)
        seq = tf.stack([feat_r, feat_d, feat_l], axis=1)

        # 2. 이름표(Modality Embedding) 부착! (성능 상승의 핵심)
        seq_emb = seq + self.modality_emb

        # 3. Self-Attention
        attn = self.mha(query=seq_emb, key=seq_emb, value=seq)
        out = self.ln1(seq + attn)

        # 4. 평균 풀링 및 최종 출력
        fused = tf.reduce_mean(out, axis=1)
        fused = self.dense(fused)
        return self.ln2(fused)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

def build_head(feat, name, hidden, dropout, reg):
    """퓨전 피처 → 확률값 (단일 바이너리 출력)"""
    x = layers.Dropout(dropout,   name=f"{name}_drop")(feat)
    x = layers.Dense(hidden, use_bias=False,
                     kernel_regularizer=reg, name=f"{name}_d1")(x)
    x = layers.LayerNormalization(name=f"{name}_ln")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    x = layers.Dense(1, kernel_regularizer=reg, name=f"{name}_logit")(x)
    return layers.Activation("sigmoid", dtype="float32", name=f"{name}_prob")(x)


# ═══════════════════════════════════════════════════════════════
# 6. Uncertainty Weights Layer
# ═══════════════════════════════════════════════════════════════

class UncertaintyWeights(keras.layers.Layer):
    """
    Kendall & Gal (2017) Homoscedastic Uncertainty
    각 헤드별 학습 가능한 log(σ²) 파라미터 7개
    → 약한 브랜치는 σ가 커져 자동으로 loss 기여 감소
    → add_weight()로 생성 → trainable_variables에 자동 등록
    """
    def __init__(self, head_names, **kwargs):
        super().__init__(name="uncertainty_weights", **kwargs)
        self.head_names = head_names
        self.log_vars   = {
            name: self.add_weight(
                name=f"log_var_{name}",
                shape=(),
                initializer="zeros",
                trainable=True,
                dtype=tf.float32,
            )
            for name in head_names
        }

    def call(self, inputs):
        return inputs   # 통과만 함 (가중치 보관 목적)

    def get_log_var(self, name):
        return self.log_vars[name]

    def get_config(self):
        cfg = super().get_config()
        cfg["head_names"] = self.head_names
        return cfg


# ═══════════════════════════════════════════════════════════════
# 7. Inner Model (Functional API, 7 Dict Outputs)
# ═══════════════════════════════════════════════════════════════
def build_aux_head(feat, name, reg):
    x   = layers.Dense(32, use_bias=False, kernel_regularizer=reg,
                        name=f"{name}_d")(feat)
    x   = layers.BatchNormalization(name=f"{name}_bn")(x)
    x   = layers.Activation("relu", name=f"{name}_relu")(x)
    x   = layers.Dense(1, kernel_regularizer=reg, name=f"{name}_logit")(x)
    return layers.Activation("sigmoid", dtype="float32", name=f"{name}_prob")(x)

def build_inner_model(cfg: dict) -> keras.Model:
    img_size = cfg["img_size"]
    dropout  = cfg["dropout_rate"]
    wd       = cfg["weight_decay"]
    reg      = keras.regularizers.l2(wd)
    hidden   = cfg["head_hidden"]
    attn_dim = cfg["attn_dim"]   # ← 전역 ATTN_DIM 대신 지역변수 사용

    if CLIP_AVAILABLE:
        rgb_ext = build_rgb_branch(img_size, cfg["rgb_embed_dim"],
                                    cfg["freeze_backbone"])
    else:
        rgb_ext = build_rgb_branch_fallback(img_size, cfg["rgb_embed_dim"],
                                             cfg["freeze_backbone"])
    dct_ext = build_fad_branch(img_size, cfg["dct_embed_dim"])
    lm_ext  = build_lm_branch(LM_FEAT_DIM, cfg["lm_embed_dim"], dropout)

    inp_face = keras.Input(shape=(img_size, img_size, 3), name="face")
    inp_dct  = keras.Input(shape=(img_size, img_size, 6), name="dct")
    inp_lm   = keras.Input(shape=(LM_FEAT_DIM,),          name="lm")

    rgb_raw = rgb_ext(inp_face)
    dct_raw, lf_raw, hf_raw = dct_ext(inp_dct)
    lm_raw  = lm_ext(inp_lm)

    # ← attn_dim 지역변수로 교체
    rgb_p = project_feat(rgb_raw, attn_dim, "rgb")
    dct_p = project_feat(dct_raw, attn_dim, "dct")
    lm_p  = project_feat(lm_raw,  attn_dim, "lm")

    outputs = {}
    outputs["rgb"]     = build_head(rgb_p, "h_rgb",     hidden, dropout, reg)
    outputs["dct"]     = build_head(dct_p, "h_dct",     hidden, dropout, reg)
    outputs["lm"]      = build_head(lm_p,  "h_lm",      hidden, dropout, reg)

    rd = cross_attention_2(rgb_p, dct_p, attn_dim, "ca_rd")
    outputs["rgb_dct"] = build_head(rd,   "h_rgb_dct",  hidden, dropout, reg)

    rl = cross_attention_2(rgb_p, lm_p,  attn_dim, "ca_rl")
    outputs["rgb_lm"]  = build_head(rl,   "h_rgb_lm",   hidden, dropout, reg)

    dl = cross_attention_2(dct_p, lm_p,  attn_dim, "ca_dl")
    outputs["dct_lm"]  = build_head(dl,   "h_dct_lm",   hidden, dropout, reg)

    all_f = CrossAttention3Way(attn_dim, name="ca_all")([rgb_p, dct_p, lm_p])
    outputs["all"]     = build_head(all_f, "h_all",      hidden, dropout, reg)

    outputs["aux_lf"]  = build_aux_head(lf_raw, "aux_lf", reg)
    outputs["aux_hf"]  = build_aux_head(hf_raw, "aux_hf", reg)

    return keras.Model(
        inputs  = {"face": inp_face, "dct": inp_dct, "lm": inp_lm},
        outputs = outputs,
        name    = "inner_7head",
    )

# ═══════════════════════════════════════════════════════════════
# 8. DeepfakeDetector (Custom train_step + Uncertainty Loss)
# ═══════════════════════════════════════════════════════════════

class DeepfakeDetector(keras.Model):
    def __init__(self, inner_model, focal_alpha=0.5, focal_gamma=2.0,
                 aux_weight=0.4, **kwargs):
        super().__init__(**kwargs)
        self.inner_model  = inner_model
        self.uw_layer     = UncertaintyWeights(HEAD_NAMES)
        self.focal_alpha  = focal_alpha
        self.focal_gamma  = focal_gamma
        self.aux_weight   = aux_weight

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.auc_metric   = keras.metrics.AUC(name="auc")
        self.acc_metric   = keras.metrics.BinaryAccuracy(name="acc")
        self.precision_metric = keras.metrics.Precision(name="precision")
        self.recall_metric = keras.metrics.Recall(name="recall")

    @property
    def metrics(self):
        return [
            self.loss_tracker, self.auc_metric, self.acc_metric,
            self.precision_metric, self.recall_metric
        ]

    def call(self, inputs, training=False):
        return self.inner_model(inputs, training=training)

    def _focal(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        a      = y_true * self.focal_alpha + (1-y_true) * (1-self.focal_alpha)
        p      = y_true * y_pred + (1-y_true) * (1-y_pred)
        return tf.reduce_mean(
            -a * tf.pow(1.0 - p, self.focal_gamma) * tf.math.log(p)
        )

    def _uncertainty_loss(self, y_true, y_pred_dict):
        total = tf.constant(0.0, dtype=tf.float32)

        # 메인 7 헤드: 불확실성 가중
        for name in HEAD_NAMES:
            l_i   = self._focal(y_true, y_pred_dict[name])
            lv    = self.uw_layer.get_log_var(name)
            total += 0.5 * (tf.exp(-lv) * l_i + lv)

        # Auxiliary 헤드: 고정 가중치
        for name in AUX_HEAD_NAMES:
            total += self.aux_weight * self._focal(y_true, y_pred_dict[name])

        return total

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss   = self._uncertainty_loss(y, y_pred)
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        scaled_grads = tape.gradient(scaled_loss, self.trainable_variables)
        grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # 소프트 레이블 → 하드 (메트릭용)
        y_hard = tf.cast(y >= 0.5, tf.float32)
        self.loss_tracker.update_state(loss)
        self.auc_metric.update_state(y_hard, y_pred["all"])
        self.acc_metric.update_state(y_hard, y_pred["all"])
        self.precision_metric.update_state(y_hard, y_pred["all"])
        self.recall_metric.update_state(y_hard, y_pred["all"])

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y   = data
        y_pred = self(x, training=False)
        loss   = self._uncertainty_loss(y, y_pred)

        self.loss_tracker.update_state(loss)
        self.auc_metric.update_state(y, y_pred["all"])
        self.acc_metric.update_state(y, y_pred["all"])
        self.precision_metric.update_state(y, y_pred["all"])
        self.recall_metric.update_state(y, y_pred["all"])

        return {m.name: m.result() for m in self.metrics}

# ═══════════════════════════════════════════════════════════════
# 9. 학습률 스케줄러
# ═══════════════════════════════════════════════════════════════

@keras.utils.register_keras_serializable(package="deepfake")
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_init, lr_min, warmup_steps, total_steps):
        super().__init__()
        self.lr_init      = float(lr_init)
        self.lr_min       = float(lr_min)
        self.warmup_steps = int(warmup_steps)
        self.total_steps  = int(total_steps)

    # 수정
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.lr_init * (step / self.warmup_steps)
        denom = float(max(self.total_steps - self.warmup_steps, 1))  # ← Python float로 계산
        cosine = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (
                1.0 + tf.cos(
            np.pi * (step - self.warmup_steps) / denom  # float32 / float → 정상
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
# 10. Callbacks
# ═══════════════════════════════════════════════════════════════

def _unfreeze_all(m):
    """재귀적으로 모든 레이어 unfreeze (CLIP 포함)"""
    m.trainable = True
    for layer in getattr(m, "layers", []):
        layer.trainable = True
        if hasattr(layer, "clip"):
            layer.clip.trainable = True
        _unfreeze_all(layer)


class BackboneUnfreezeCallback(keras.callbacks.Callback):
    def __init__(self, unfreeze_epoch: int, lr_scale: float = 0.01):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.lr_scale       = lr_scale
        self._unfrozen      = False

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.unfreeze_epoch and not self._unfrozen:
            _unfreeze_all(self.model)

            current_step = self.model.optimizer.iterations
            sched        = self.model.optimizer.learning_rate
            current_lr   = float(sched(current_step)) if callable(sched) else float(sched)
            new_lr       = current_lr * self.lr_scale

            # 스케줄러를 고정값으로 교체 (unfreeze 후엔 낮은 lr로 fine-tune)
            self.model.optimizer.learning_rate = new_lr
            self._unfrozen = True
            log.info(f"Epoch {epoch}: backbone unfreeze | "
                     f"lr {current_lr:.2e} → {new_lr:.2e}")


class LogUncertaintyCallback(keras.callbacks.Callback):
    """에폭 끝마다 각 헤드의 log_var(불확실성) 출력"""
    def on_epoch_end(self, epoch, logs=None):
        lv = {
            name: float(self.model.uw_layer.get_log_var(name))
            for name in HEAD_NAMES
        }
        log.info(
            f"Epoch {epoch+1} log_vars | "
            + "  ".join(f"{k}={v:+.3f}" for k, v in lv.items())
        )


def _get_lr(model):
    sched = model.optimizer.learning_rate
    if callable(sched):
        return float(sched(model.optimizer.iterations))
    return float(sched)


def build_callbacks(cfg: dict, model) -> list:
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath          = os.path.join(
                cfg["ckpt_dir"],
                "best_epoch{epoch:03d}_valauc{val_auc:.4f}",
            ),
            monitor           = "val_auc",
            mode              = "max",
            save_best_only    = True,
            save_weights_only = True,
            verbose           = 1,
        ),
        keras.callbacks.EarlyStopping(
            monitor              = "val_auc",
            mode                 = "max",
            patience             = 12,
            restore_best_weights = True,
            verbose              = 1,
            min_delta            = 0.001,
        ),
        BackboneUnfreezeCallback(
            unfreeze_epoch = cfg["unfreeze_epoch"],
            lr_scale       = 0.01,  # CLIP fine-tune은 lr 매우 작게
        ),
        LogUncertaintyCallback(),
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: log.info(
                f"Epoch {epoch+1:3d} │ "
                f"loss={logs.get('loss',0):.4f}  "
                f"auc={logs.get('auc',0):.4f}  "
                f"val_loss={logs.get('val_loss',0):.4f}  "
                f"val_auc={logs.get('val_auc',0):.4f}  "
                f"lr={_get_lr(model):.2e}"
            )
        ),
    ]


# ═══════════════════════════════════════════════════════════════
# 11. 평가 — Ablation Table 자동 생성
# ═══════════════════════════════════════════════════════════════

# HEAD_NAMES, AUX_HEAD_NAMES 아래에 추가
ALL_EVAL_HEADS = HEAD_NAMES + AUX_HEAD_NAMES


def evaluate_model(model: DeepfakeDetector, test_ds, tag="Test"):
    preds_all  = {name: [] for name in ALL_EVAL_HEADS}  # ← 수정
    labels_all = []

    for inputs, labels in test_ds:
        batch_out = model(inputs, training=False)
        for name in ALL_EVAL_HEADS:                       # ← 수정
            preds_all[name].append(batch_out[name].numpy().flatten())
        labels_all.append(labels.numpy().flatten())

    y_true = np.concatenate(labels_all)

    log.info(f"\n{'='*62}")
    log.info(f"  [{tag}]  Ablation Table")
    log.info(f"{'='*62}")
    log.info(f"  {'Head':<12}  {'AUC':>6}  {'F1':>6}  {'ACC':>6}  {'Thr':>5}")
    log.info(f"  {'-'*46}")

    results = {}
    for name in ALL_EVAL_HEADS:                           # ← 수정
        y_prob = np.concatenate(preds_all[name])
        auc_m  = keras.metrics.AUC()
        auc_m.update_state(y_true, y_prob)
        auc    = float(auc_m.result())

        best_f1, best_thr = 0.0, 0.5
        for thr in np.arange(0.1, 0.9, 0.05):
            yp = (y_prob >= thr).astype(int)
            tp = np.sum((yp == 1) & (y_true == 1))
            fp = np.sum((yp == 1) & (y_true == 0))
            fn = np.sum((yp == 0) & (y_true == 1))
            p  = tp / (tp + fp + 1e-8)
            r  = tp / (tp + fn + 1e-8)
            f1 = 2 * p * r / (p + r + 1e-8)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        y_pred = (y_prob >= best_thr).astype(int)
        acc    = np.mean(y_pred == y_true.astype(int))

        # ← 마커 구분
        if name == "all":
            marker = "  ★ MAIN"
        elif name in AUX_HEAD_NAMES:
            marker = "  (aux)"
        else:
            marker = ""

        log.info(f"  {name:<12}  {auc:.4f}  {best_f1:.4f}  {acc:.4f}  "
                 f"{best_thr:.2f}{marker}")
        results[name] = {"auc": auc, "f1": best_f1, "acc": acc, "thr": best_thr}

    log.info(f"{'='*62}")
    return results

# ═══════════════════════════════════════════════════════════════
# 12. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    cfg  = CFG
    seed = cfg["seed"]
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # STEP 1: 데이터 수집
    log.info("=" * 60)
    log.info("STEP 1: Train 데이터 수집")
    train_samples = collect_samples(cfg["processed_root"], cfg["train_datasets"])

    # STEP 2: 영상 단위 분할
    log.info("=" * 60)
    log.info("STEP 2: 영상 ID 단위 분할")
    splits = video_level_split(train_samples, cfg["split_ratio"], seed)

    # STEP 3: 불균형 처리
    log.info("=" * 60)
    log.info("STEP 3: 클래스 불균형 처리 (Undersample)")
    splits["train"] = undersample_fake(splits["train"], cfg["undersample_ratio"], seed)

    # STEP 4: External Test 수집
    log.info("=" * 60)
    log.info("STEP 4: External Test 데이터 수집")
    ext_test_samples = collect_samples(cfg["processed_root"], cfg["test_datasets"])

    # STEP 5: tf.data 빌드
    log.info("=" * 60)
    log.info("STEP 5: tf.data 파이프라인 빌드")
    bs = cfg["batch_size"]
    train_ds    = build_tf_dataset(splits["train"],     "train", bs, seed)
    val_ds      = build_tf_dataset(splits["val"],       "val",   bs, seed)
    int_test_ds = build_tf_dataset(splits["test"],      "test",  bs, seed)
    ext_test_ds = build_tf_dataset(ext_test_samples,    "test",  bs, seed)

    # STEP 6: 모델 빌드
    log.info("=" * 60)
    log.info("STEP 6: 모델 빌드 (7-Head + Cross-Attention)")
    inner   = build_inner_model(cfg)
    model   = DeepfakeDetector(
        inner_model  = inner,
        focal_alpha  = cfg["focal_alpha"],
        focal_gamma  = cfg["focal_gamma"],
        aux_weight=cfg["aux_weight"],
        name         = "deepfake_v3",
    )
    log.info(f"Inner model params: {inner.count_params():,}")

    # STEP 7: 컴파일
    log.info("=" * 60)
    log.info("STEP 7: 컴파일")
    steps_per_epoch = len(splits["train"]) // bs
    total_steps     = steps_per_epoch * cfg["epochs"]
    warmup_steps    = steps_per_epoch * cfg["warmup_epochs"]

    schedule  = WarmupCosineDecay(
        cfg["lr_init"], cfg["lr_min"],
        int(warmup_steps), int(total_steps)
    )
    optimizer = keras.optimizers.Adam(learning_rate=schedule, clipnorm=1.0)
    model.compile(optimizer=optimizer)

    log.info(f"steps/epoch={steps_per_epoch:,}  "
             f"total={total_steps:,}  warmup={warmup_steps:,}")

    # STEP 8: 학습
    log.info("=" * 60)
    log.info("STEP 8: 학습 시작")
    t0      = time.time()
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = cfg["epochs"],
        callbacks       = build_callbacks(cfg, model),
        verbose         = 1,
    )
    log.info(f"학습 완료 — {(time.time()-t0)/3600:.2f}h")

    # STEP 9: 평가
    log.info("=" * 60)
    log.info("STEP 9: 최종 평가")
    int_res = evaluate_model(model, int_test_ds, "Internal (FF++/DFF/HIDF)")
    ext_res = evaluate_model(model, ext_test_ds, "External (CelebDF/redface)")

    # STEP 10: 저장
    final_path = os.path.join(cfg["ckpt_dir"], f"final_{_ts}")
    model.save_weights(final_path)
    log.info(f"저장 완료: {final_path}")

    return history, int_res, ext_res


if __name__ == "__main__":
    history, int_res, ext_res = main()