"""
Deepfake Detection — 3-Stream Training Pipeline (v2)
─────────────────────────────────────────────────────
변경사항:
  1. LM Branch: (68,2) 절대좌표 → (34,) 기하학적 비율 피처
     - 눈/코/입 비율, 좌우 대칭성, 얼굴 종횡비 등
     - HIDF/redface 도메인에서도 의미 있는 신호 추출 가능
  2. 데이터셋 역할 분리
     - Train : FF++ + DFF + HIDF
     - Test  : CelebDF + redface + Deepfake-Eval-2024 (학습 절대 불포함)
  3. 도메인 핑거프린트 방지
     - 완전 랜덤 셔플 (buffer_size=10000)
  4. LM 임베딩 64 → 32차원으로 축소

모델: RGB(EfficientNetB4) + DCT(CNN) + LM(MLP) → Late Fusion
불균형: Focal Loss (α=0.75, γ=2) + fake 언더샘플(3×real)
분할: 영상 ID 단위 7:1:2 → data leakage 방지
"""

import os, sys, re, time, logging
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision

# ── 로깅
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

mixed_precision.set_global_policy("mixed_float16")
log.info(f"compute dtype: {mixed_precision.global_policy().compute_dtype}")


# ═══════════════════════════════════════════════════════════════
# 0. CONFIG
# ═══════════════════════════════════════════════════════════════
CFG = {
    # ── 데이터
    "processed_root"   : "./processed",

    # ── 역할 분리
    # Train/Val 에만 사용 (도메인 핑거프린트 방지)
    "train_datasets"   : ["ffpp", "dff", "hidf"],
    # Test 전용 (학습에 절대 포함 X)
    "test_datasets"    : ["celebdf", "redface"],

    # ── 불균형
    "undersample_ratio": 1.0,        # fake = real * ratio

    # ── 이미지
    "img_size"         : 224,

    # ── 모델
    "freeze_backbone"  : True,
    "unfreeze_epoch"   : 15,
    "rgb_embed_dim"    : 256,
    "dct_embed_dim"    : 128,
    "lm_embed_dim"     : 32,         # 64 → 32로 축소 (기하학적 피처 34차원에 맞게)
    "fusion_hidden"    : [512, 256],
    "dropout_rate"     : 0.5,

    # ── 학습
    "batch_size"       : 32,
    "epochs"           : 50,
    "lr_init"          : 3e-4,
    "lr_min"           : 1e-7,
    "warmup_epochs"    : 5,
    "weight_decay"     : 3e-4,

    # ── Focal Loss
    "focal_alpha"      : 0.5,
    "focal_gamma"      : 2.0,

    # ── 분할
    "split_ratio"      : (0.7, 0.1, 0.2),
    "seed"             : 42,

    # ── 저장
    "ckpt_dir"         : "./checkpoints",
    # "log_dir"          : "./logs",
}

MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

# LM 기하학적 피처 차원 (변경 시 build_lm_branch도 함께 수정)
LM_FEAT_DIM = 34


# ═══════════════════════════════════════════════════════════════
# 1. 기하학적 랜드마크 피처 추출
# ═══════════════════════════════════════════════════════════════

def extract_geometric_features(lm: np.ndarray) -> np.ndarray:
    """
    (68, 2) 절대 픽셀 좌표 → (34,) 기하학적 비율 피처

    절대좌표 대신 비율/대칭성을 사용하므로:
    - 얼굴 크기·위치 불변
    - HIDF/RedFace처럼 랜드마크가 자연스러운 경우에도
      미세한 기하학적 비일관성 검출 가능

    dlib 68점 인덱스 기준:
      0-16 : 얼굴 외곽
      17-21: 왼쪽 눈썹 / 22-26: 오른쪽 눈썹
      27-35: 코 / 36-41: 왼쪽 눈 / 42-47: 오른쪽 눈
      48-67: 입
    """
    feats = []

    # ── 기준값: 얼굴 너비 (외곽 0번 ↔ 16번)
    face_w = float(np.linalg.norm(lm[16] - lm[0])) + 1e-8
    # 얼굴 높이 (턱 끝 8번 ↔ 미간 27번)
    face_h = float(np.linalg.norm(lm[8] - lm[27])) + 1e-8

    # ── 1. 얼굴 종횡비
    feats.append(face_h / face_w)                                   # 1

    # ── 2. 눈 관련 (6개)
    eye_dist  = float(np.linalg.norm(lm[42] - lm[36]))             # 눈 간격
    eye_l_h   = float(np.linalg.norm(lm[41] - lm[37]))             # 왼눈 높이
    eye_r_h   = float(np.linalg.norm(lm[47] - lm[43]))             # 오른눈 높이
    eye_l_w   = float(np.linalg.norm(lm[39] - lm[36]))             # 왼눈 너비
    eye_r_w   = float(np.linalg.norm(lm[45] - lm[42]))             # 오른눈 너비
    feats.append(eye_dist / face_w)                                 # 2
    feats.append(eye_l_h / (eye_r_h + 1e-8))                       # 3 대칭성
    feats.append(eye_l_w / (eye_r_w + 1e-8))                       # 4 대칭성
    feats.append(eye_l_h / (eye_l_w + 1e-8))                       # 5 왼눈 종횡비
    feats.append(eye_r_h / (eye_r_w + 1e-8))                       # 6 오른눈 종횡비
    feats.append(eye_dist / face_h)                                 # 7

    # ── 3. 눈썹 관련 (4개)
    brow_l_h = float(np.linalg.norm(lm[19] - lm[38]))              # 왼눈썹-눈 거리
    brow_r_h = float(np.linalg.norm(lm[24] - lm[44]))              # 오른눈썹-눈 거리
    brow_l_w = float(np.linalg.norm(lm[21] - lm[17]))              # 왼눈썹 너비
    brow_r_w = float(np.linalg.norm(lm[26] - lm[22]))              # 오른눈썹 너비
    feats.append(brow_l_h / face_h)                                 # 8
    feats.append(brow_r_h / face_h)                                 # 9
    feats.append(brow_l_h / (brow_r_h + 1e-8))                     # 10 대칭성
    feats.append(brow_l_w / (brow_r_w + 1e-8))                     # 11 대칭성

    # ── 4. 코 관련 (4개)
    nose_h   = float(np.linalg.norm(lm[33] - lm[27]))              # 코 길이
    nose_w   = float(np.linalg.norm(lm[35] - lm[31]))              # 코 너비
    nose_tip = float(np.linalg.norm(lm[33] - lm[30]))              # 코끝 처짐
    feats.append(nose_h / face_h)                                   # 12
    feats.append(nose_w / face_w)                                   # 13
    feats.append(nose_h / (nose_w + 1e-8))                         # 14 종횡비
    feats.append(nose_tip / nose_h)                                 # 15

    # ── 5. 입 관련 (6개)
    mouth_w  = float(np.linalg.norm(lm[54] - lm[48]))              # 입 너비
    mouth_h  = float(np.linalg.norm(lm[57] - lm[51]))              # 입 높이
    upper_h  = float(np.linalg.norm(lm[51] - lm[62]))              # 윗입술 두께
    lower_h  = float(np.linalg.norm(lm[66] - lm[57]))              # 아랫입술 두께
    mouth_l  = float(np.linalg.norm(lm[48] - lm[8]))               # 왼쪽 입꼬리-턱
    mouth_r  = float(np.linalg.norm(lm[54] - lm[8]))               # 오른쪽 입꼬리-턱
    feats.append(mouth_w / face_w)                                  # 16
    feats.append(mouth_h / face_h)                                  # 17
    feats.append(mouth_w / (mouth_h + 1e-8))                       # 18 종횡비
    feats.append(upper_h / (lower_h + 1e-8))                       # 19 입술 대칭
    feats.append(mouth_l / (mouth_r + 1e-8))                       # 20 좌우 대칭
    feats.append(mouth_w / (eye_dist + 1e-8))                      # 21 입/눈 비율

    # ── 6. 얼굴 전체 비율 (4개)
    jaw_l    = float(np.linalg.norm(lm[4]  - lm[0]))               # 왼쪽 턱선
    jaw_r    = float(np.linalg.norm(lm[12] - lm[16]))              # 오른쪽 턱선
    chin_l   = float(np.linalg.norm(lm[8]  - lm[4]))               # 왼쪽 턱 하단
    chin_r   = float(np.linalg.norm(lm[8]  - lm[12]))              # 오른쪽 턱 하단
    feats.append(jaw_l / (jaw_r + 1e-8))                           # 22 대칭성
    feats.append(chin_l / (chin_r + 1e-8))                         # 23 대칭성
    feats.append((jaw_l + jaw_r) / face_w)                         # 24
    feats.append((chin_l + chin_r) / face_h)                       # 25

    # ── 7. 눈-입-코 삼각형 비율 (4개)
    eye_mid  = (lm[39] + lm[45]) / 2                               # 두 눈 중심
    nose_tip_pt = lm[33]
    mouth_mid   = (lm[48] + lm[54]) / 2
    tri_h    = float(np.linalg.norm(nose_tip_pt - eye_mid))
    tri_b    = float(np.linalg.norm(mouth_mid  - nose_tip_pt))
    tri_asym = float(np.linalg.norm(eye_mid    - nose_tip_pt))
    feats.append(tri_h  / face_h)                                  # 26
    feats.append(tri_b  / face_h)                                  # 27
    feats.append(tri_h  / (tri_b + 1e-8))                         # 28
    feats.append(tri_asym / face_w)                                # 29

    # ── 8. 좌우 전체 대칭 점수 (5개)
    # 좌우 대응점 거리 평균 (대칭이면 0에 가까움)
    sym_pairs = [(0,16),(1,15),(2,14),(3,13),(4,12)]
    for l, r in sym_pairs:
        # 얼굴 중심선 기준 좌우 x 거리 비율
        center_x = (lm[0][0] + lm[16][0]) / 2
        diff = abs(abs(lm[l][0] - center_x) - abs(lm[r][0] - center_x))
        feats.append(diff / face_w)                                # 30~34

    arr = np.array(feats[:LM_FEAT_DIM], dtype=np.float32)
    return arr


# ═══════════════════════════════════════════════════════════════
# 2. 데이터 수집
# ═══════════════════════════════════════════════════════════════

def collect_samples(processed_root: str, dataset_keys: list) -> list:
    """
    processed/{dataset}/{0,1}/ 스캔 → (face, lm, label) 리스트
    dataset_keys: 포함할 데이터셋 이름 리스트
    """
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
            label = int(label_str)

            # _lm.npy suffix 자동 감지
            lm_suffix = None
            for s in ("_lm.npy", "_im.npy"):
                if next(label_dir.glob(f"*{s}"), None):
                    lm_suffix = s
                    break
            if lm_suffix is None:
                log.warning(f"랜드마크 파일 없음: {label_dir}")
                continue

            for face_path in sorted(label_dir.glob("*_face.jpg")):
                stem    = face_path.stem.replace("_face", "")
                lm_path = label_dir / f"{stem}{lm_suffix}"
                if lm_path.exists():
                    ds_samples.append((str(face_path), str(lm_path), label))

        nr = sum(1 for *_, l in ds_samples if l == 0)
        nf = sum(1 for *_, l in ds_samples if l == 1)
        log.info(f"[{ds_name}] real={nr:,}  fake={nf:,}  total={len(ds_samples):,}")
        samples.extend(ds_samples)

    nr = sum(1 for *_, l in samples if l == 0)
    nf = sum(1 for *_, l in samples if l == 1)
    log.info(f"전체 수집: real={nr:,}  fake={nf:,}  total={len(samples):,}")
    return samples


def _video_id(face_path: str) -> str:
    stem = Path(face_path).stem.replace("_face", "")
    return re.sub(r"_f\d{3}$", "", stem)


def video_level_split(samples: list, ratio: tuple, seed: int) -> dict:
    """영상 ID 단위 분할 — data leakage 방지"""
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


def undersample_fake(samples: list, ratio: float, seed: int) -> list:
    real   = [s for s in samples if s[2] == 0]
    fake   = [s for s in samples if s[2] == 1]
    target = int(len(real) * ratio)

    if target >= len(fake):
        log.info("fake 언더샘플 불필요")
        return samples

    rng          = np.random.default_rng(seed)
    fake_idx     = rng.choice(len(fake), size=target, replace=False)
    fake_sampled = [fake[i] for i in fake_idx]

    log.info(
        f"언더샘플: real={len(real):,}  fake={len(fake_sampled):,}"
        f"  (원래 {len(fake):,} → {len(fake_sampled):,})"
    )
    combined = real + fake_sampled
    rng.shuffle(combined)

    return combined


# ═══════════════════════════════════════════════════════════════
# 3. tf.data 파이프라인
# ═══════════════════════════════════════════════════════════════

def _load_lm_geom(path_tensor) -> np.ndarray:
    """
    .npy (68,2) 로드 → 기하학적 피처 (34,) 추출
    """
    lm = np.load(path_tensor.numpy().decode()).astype(np.float32)
    return extract_geometric_features(lm)

def mixup_batch(inputs, labels, alpha=0.2):
    """배치 내 MixUp — 도메인 일반화에 유효"""
    batch_size = tf.shape(inputs["face"])[0]
    lam = tf.random.uniform([], 0.0, alpha)   # 약한 MixUp
    lam = tf.maximum(lam, 1.0 - lam)          # lam >= 0.5 보장

    # 셔플 인덱스
    idx = tf.random.shuffle(tf.range(batch_size))

    mixed = {}
    for key in ["face", "dct"]:
        mixed[key] = lam * inputs[key] + (1 - lam) * tf.gather(inputs[key], idx)
    mixed["lm"] = inputs["lm"]  # LM은 mix 안 함

    # 소프트 레이블
    labels_f   = tf.cast(labels, tf.float32)
    labels_mix = lam * labels_f + (1 - lam) * tf.cast(tf.gather(labels, idx), tf.float32)

    return mixed, labels_mix


def augment_face(face: tf.Tensor) -> tf.Tensor:
    face = tf.image.random_flip_left_right(face)
    face = tf.image.random_brightness(face, max_delta=0.2)
    face = tf.image.random_contrast(face, lower=0.7, upper=1.3)
    face = tf.image.random_saturation(face, lower=0.7, upper=1.3)
    # face = tf.image.random_hue(face, max_delta=0.05)          # 추가

    # Gaussian noise 추가 (도메인 핑거프린트 희석)
    # noise = tf.random.normal(tf.shape(face), stddev=0.02)
    # face = tf.clip_by_value(face + noise, 0.0, 1.0)

    face = tf.cast(face * 255, tf.uint8)

    def jpeg_compress(img):
        # JPEG 품질 범위를 더 넓게 (50~95) → 압축 아티팩트 다양화
        q = int(np.random.randint(75, 95))
        encoded = tf.image.encode_jpeg(img, quality=q)
        return tf.image.decode_jpeg(encoded, channels=3)

    face = tf.py_function(jpeg_compress, [face], tf.uint8)
    face.set_shape([224, 224, 3])
    face = tf.cast(face, tf.float32) / 255.0

    # Random Gaussian blur (도메인 일반화)
    def random_blur(img):
        if np.random.rand() < 0.3:
            sigma = np.random.uniform(0.5, 1.5)
            img = tf.expand_dims(img, 0)
            img = tf.nn.avg_pool2d(img, ksize=3, strides=1, padding='SAME')
            img = tf.squeeze(img, 0)
        return img

    face = tf.py_function(random_blur, [face], tf.float32)
    face.set_shape([224, 224, 3])
    return tf.clip_by_value(face, 0.0, 1.0)

def compute_dct_tf(face: tf.Tensor) -> tf.Tensor:
    """(224,224,3) [0,1] → DCT log-normalized (224,224,3)"""
    channels     = tf.unstack(face, axis=-1)
    dct_channels = []
    for ch in channels:
        d     = tf.signal.dct(ch, type=2, norm="ortho")
        d     = tf.signal.dct(tf.transpose(d), type=2, norm="ortho")
        d     = tf.transpose(d)
        d     = tf.math.log(tf.abs(d) + 1e-8)
        d_min = tf.reduce_min(d)
        d_max = tf.reduce_max(d)
        d     = (d - d_min) / (d_max - d_min + 1e-8)
        dct_channels.append(d)
    return tf.stack(dct_channels, axis=-1)


def make_load_fn(split: str):
    do_aug = (split == "train")

    def load_fn(face_p, lm_p, label):
        # ── 얼굴 이미지
        face = tf.image.decode_jpeg(tf.io.read_file(face_p), channels=3)
        face = tf.cast(face, tf.float32) / 255.0

        if do_aug:
            face = augment_face(face)

        # ── DCT (정규화 전 픽셀로 계산)
        dct = compute_dct_tf(face)
        dct.set_shape([224, 224, 3])

        # ── ImageNet 정규화
        face = (face - MEAN) / STD
        face.set_shape([224, 224, 3])

        # ── 랜드마크 → 기하학적 피처 (34,)
        lm = tf.py_function(_load_lm_geom, [lm_p], tf.float32)
        lm.set_shape([LM_FEAT_DIM])

        return (
            {"face": face, "dct": dct, "lm": lm},
            tf.cast(label, tf.int32),
        )

    return load_fn


def build_tf_dataset(samples: list, split: str,
                     batch_size: int, seed: int) -> tf.data.Dataset:
    face_paths = [s[0] for s in samples]
    lm_paths   = [s[1] for s in samples]
    labels     = [s[2] for s in samples]

    ds = tf.data.Dataset.from_tensor_slices((face_paths, lm_paths, labels))

    # ← 셔플을 map 이전으로 이동 (파일 경로 단계에서 섞는 게 더 효율적)
    if split == "train":
        ds = ds.shuffle(buffer_size=10000, seed=seed,
                        reshuffle_each_iteration=True)

    ds = ds.map(make_load_fn(split), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=(split == "train"))

    # ← MixUp은 반드시 batch() 이후에 (배치 단위 연산이기 때문)
    # if split == "train":
    #     ds = ds.map(mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


# ═══════════════════════════════════════════════════════════════
# 4. 모델 아키텍처
# ═══════════════════════════════════════════════════════════════

def build_rgb_branch(img_size: int, embed_dim: int,
                     freeze: bool) -> keras.Model:
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


def build_dct_branch(img_size: int, embed_dim: int) -> keras.Model:
    def conv_block(x, filters, stride=1):
        x = layers.Conv2D(filters, 3, stride, padding="same",
                          use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation("relu")(x)

    inp = keras.Input(shape=(img_size, img_size, 3), name="dct")
    x   = conv_block(inp, 32,  stride=2)
    x   = conv_block(x,   64,  stride=2)
    x   = conv_block(x,  128,  stride=2)
    x   = conv_block(x,  256,  stride=2)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(embed_dim, use_bias=False)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    return keras.Model(inp, x, name="dct_branch")


def build_lm_branch(feat_dim: int, embed_dim: int,
                    dropout: float) -> keras.Model:
    """
    입력: (34,) 기하학적 비율 피처
    출력: (embed_dim,) 임베딩
    """
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


def build_model(cfg: dict) -> keras.Model:
    img_size   = cfg["img_size"]
    rgb_branch = build_rgb_branch(img_size, cfg["rgb_embed_dim"],
                                  cfg["freeze_backbone"])
    dct_branch = build_dct_branch(img_size, cfg["dct_embed_dim"])
    lm_branch  = build_lm_branch(LM_FEAT_DIM, cfg["lm_embed_dim"],
                                  cfg["dropout_rate"])

    inp_face = keras.Input(shape=(img_size, img_size, 3), name="face")
    inp_dct  = keras.Input(shape=(img_size, img_size, 3), name="dct")
    inp_lm   = keras.Input(shape=(LM_FEAT_DIM,),          name="lm")

    rgb_feat = rgb_branch(inp_face)
    dct_feat = dct_branch(inp_dct)
    lm_feat  = lm_branch(inp_lm)

    x = layers.Concatenate(name="concat")([rgb_feat, dct_feat, lm_feat])

    wd = cfg["weight_decay"]
    reg = keras.regularizers.l2(wd)
    for units in cfg["fusion_hidden"]:
        x = layers.Dense(units, use_bias=False, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(cfg["dropout_rate"])(x)

    out = layers.Dense(1, name="logit", kernel_regularizer=reg)(x)
    out = layers.Activation("sigmoid", dtype="float32", name="prob")(out)

    model = keras.Model(
        inputs={"face": inp_face, "dct": inp_dct, "lm": inp_lm},
        outputs=out,
        name="deepfake_3stream_v2",
    )
    return model


# ═══════════════════════════════════════════════════════════════
# 5. Focal Loss
# ═══════════════════════════════════════════════════════════════
@keras.utils.register_keras_serializable(package="deepfake")
class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.75, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def call(self, y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.cast(y_pred, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        fl      = -alpha_t * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)
        return tf.reduce_mean(fl)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": float(self.alpha), "gamma": float(self.gamma)})
        return cfg


# ═══════════════════════════════════════════════════════════════
# 6. 스케줄러
# ═══════════════════════════════════════════════════════════════

@keras.utils.register_keras_serializable(package="deepfake")
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_init, lr_min, warmup_steps, total_steps):
        super().__init__()
        self.lr_init      = float(lr_init)
        self.lr_min       = float(lr_min)
        self.warmup_steps = int(warmup_steps)
        self.total_steps  = int(total_steps)

    def __call__(self, step):
        step   = tf.cast(step, tf.float32)
        warmup = self.lr_init * (step / self.warmup_steps)
        cosine = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (
            1.0 + tf.cos(
                np.pi * (step - self.warmup_steps) /
                (self.total_steps - self.warmup_steps)
            )
        )
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {
            "lr_init"      : float(self.lr_init),
            "lr_min"       : float(self.lr_min),
            "warmup_steps" : int(self.warmup_steps),
            "total_steps"  : int(self.total_steps),
        }


# ═══════════════════════════════════════════════════════════════
# 7. Callbacks
# ═══════════════════════════════════════════════════════════════

class BackboneUnfreezeCallback(keras.callbacks.Callback):
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

            current_step = self.model.optimizer.iterations
            current_lr   = self.model.optimizer.learning_rate(current_step)
            new_lr       = float(current_lr) * self.lr_scale
            self.model.optimizer.learning_rate = new_lr

            self._unfrozen = True
            log.info(f"Epoch {epoch}: backbone unfreeze. "
                     f"lr {float(current_lr):.2e} → {new_lr:.2e}")

def _get_lr():
    lr = model.optimizer.learning_rate
    if callable(lr):
        return float(lr(model.optimizer.iterations).numpy())
    return float(lr)

def build_callbacks(cfg: dict) -> list:
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    # os.makedirs(cfg["log_dir"], exist_ok=True)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath          = os.path.join(
                cfg["ckpt_dir"],
                "best_auc_epoch{epoch:03d}_val{val_auc:.4f}"
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
        # keras.callbacks.TensorBoard(
        #     log_dir      = cfg["log_dir"],
        #     histogram_freq = 0,
        #     update_freq  = "epoch",
        # ),
        # keras.callbacks.CSVLogger(
        #     filename = f"training_log_{_ts}.csv",
        #     append   = False,
        # ),
        BackboneUnfreezeCallback(
            unfreeze_epoch = cfg["unfreeze_epoch"],
            lr_scale       = 0.1,
        ),
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: log.info(
                f"Epoch {epoch+1:3d} │ "
                f"loss={logs.get('loss',0):.4f}  "
                f"auc={logs.get('auc',0):.4f}  "
                f"val_loss={logs.get('val_loss',0):.4f}  "
                f"val_auc={logs.get('val_auc',0):.4f}  "
                f"lr={_get_lr():.2e}"
            )
        ),
    ]


# ═══════════════════════════════════════════════════════════════
# 8. 평가
# ═══════════════════════════════════════════════════════════════

def evaluate_model(model: keras.Model, test_ds: tf.data.Dataset,
                   tag: str = "Test"):
    y_true_all, y_pred_all = [], []
    for batch_inputs, batch_labels in test_ds:
        preds = model(batch_inputs, training=False)
        y_pred_all.append(preds.numpy().flatten())
        y_true_all.append(batch_labels.numpy().flatten())

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_pred_all)
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        yp = (y_prob >= thr).astype(int)
        tp = np.sum((yp == 1) & (y_true == 1))
        fp = np.sum((yp == 1) & (y_true == 0))
        fn = np.sum((yp == 0) & (y_true == 1))
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    log.info(f"  최적 임계값: {best_thr:.2f}  (F1={best_f1:.4f})")
    y_pred = (y_prob >= best_thr).astype(int)

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
    log.info(f"  [{tag}] 평가 결과")
    log.info("=" * 60)
    log.info(f"  AUC-ROC   : {auc_m.result().numpy():.4f}")
    log.info(f"  Accuracy  : {acc:.4f}")
    log.info(f"  Precision : {precision:.4f}")
    log.info(f"  Recall    : {recall:.4f}")
    log.info(f"  F1-Score  : {f1:.4f}")
    log.info(f"  TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")
    log.info("=" * 60)

    return {"auc": auc_m.result().numpy(), "acc": acc,
            "precision": precision, "recall": recall, "f1": f1}


# def sanity_check(ds: tf.data.Dataset):
#     log.info("─" * 50)
#     for inputs, labels in ds.take(1):
#         face = inputs["face"]
#         dct  = inputs["dct"]
#         lm   = inputs["lm"]
#         log.info(f"  face  {face.shape}  [{float(tf.reduce_min(face)):.2f}, {float(tf.reduce_max(face)):.2f}]")
#         log.info(f"  dct   {dct.shape}   [{float(tf.reduce_min(dct)):.2f}, {float(tf.reduce_max(dct)):.2f}]")
#         log.info(f"  lm    {lm.shape}    [{float(tf.reduce_min(lm)):.4f}, {float(tf.reduce_max(lm)):.4f}]")
#         log.info(f"  labels {labels.numpy()}")
#     log.info("Sanity check 완료")
#     log.info("─" * 50)


# ═══════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    cfg  = CFG
    seed = cfg["seed"]
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # ── STEP 1: Train용 데이터 수집 (FF++ + DFF + HIDF)
    log.info("=" * 60)
    log.info("STEP 1: Train 데이터 수집")
    log.info(f"  대상: {cfg['train_datasets']}")
    log.info("=" * 60)
    train_samples = collect_samples(cfg["processed_root"],
                                    cfg["train_datasets"])

    # ── STEP 2: 영상 ID 단위 분할
    log.info("=" * 60)
    log.info("STEP 2: 영상 ID 단위 분할 (leakage 방지)")
    log.info("=" * 60)
    splits = video_level_split(train_samples, cfg["split_ratio"], seed)

    # ── STEP 3: 불균형 처리 (train만)
    log.info("=" * 60)
    log.info("STEP 3: 클래스 불균형 처리")
    log.info("=" * 60)
    if cfg["undersample_ratio"] is not None:
        splits["train"] = undersample_fake(
            splits["train"], cfg["undersample_ratio"], seed
        )

    # ── STEP 4: Test용 데이터 수집 (CelebDF + redface )
    log.info("=" * 60)
    log.info("STEP 4: Test 데이터 수집 (학습 미사용)")
    log.info(f"  대상: {cfg['test_datasets']}")
    log.info("=" * 60)
    ext_test_samples = collect_samples(cfg["processed_root"],
                                       cfg["test_datasets"])

    # ── STEP 5: tf.data 빌드
    log.info("=" * 60)
    log.info("STEP 5: tf.data 파이프라인 빌드")
    log.info("=" * 60)
    bs       = cfg["batch_size"]

    train_ds = build_tf_dataset(splits["train"],    "train", bs, seed)
    val_ds   = build_tf_dataset(splits["val"],      "val",   bs, seed)
    # 내부 test (train 데이터셋 출신)
    int_test_ds = build_tf_dataset(splits["test"],  "test",  bs, seed)
    # 외부 test (CelebDF / redface / Eval-2024)
    ext_test_ds = build_tf_dataset(ext_test_samples,"test",  bs, seed)

    # sanity_check(train_ds)
    def sanity_check(ds, model, tag=""):
        for inputs, labels in ds.take(1):
            preds = model(inputs, training=False)
            log.info(
                f"[{tag}] pred min={float(tf.reduce_min(preds)):.3f} max={float(tf.reduce_max(preds)):.3f} mean={float(tf.reduce_mean(preds)):.3f}")
            log.info(f"[{tag}] labels {labels.numpy()[:8]}")

    # ── STEP 6: 모델 빌드
    log.info("=" * 60)
    log.info("STEP 6: 모델 빌드")
    log.info("=" * 60)
    global model
    model = build_model(cfg)

    sanity_check(train_ds, model, "train")
    sanity_check(val_ds, model, "val")

    # ── STEP 7: 컴파일
    log.info("=" * 60)
    log.info("STEP 7: 컴파일")
    log.info("=" * 60)
    steps_per_epoch = len(splits["train"]) // bs
    total_steps     = steps_per_epoch * cfg["epochs"]
    warmup_steps    = steps_per_epoch * cfg["warmup_epochs"]

    schedule  = WarmupCosineDecay(cfg["lr_init"], cfg["lr_min"],
                                   int(warmup_steps), int(total_steps))
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=schedule,
        clipnorm=1.0,
    )
    model.compile(
        optimizer = optimizer,
        # loss      = FocalLoss(cfg["focal_alpha"], cfg["focal_gamma"]),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics   = [
            keras.metrics.AUC(name="auc"),
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    log.info(f"steps_per_epoch={steps_per_epoch:,}  "
             f"total_steps={total_steps:,}  warmup={warmup_steps:,}")

    # ── STEP 8: 학습
    log.info("=" * 60)
    log.info("STEP 8: 학습 시작")
    log.info("=" * 60)
    t0      = time.time()
    history = model.fit(
        # 디버깅용
        # train_ds.take(1000),
        # validation_data=val_ds.take(1000),
        train_ds,
        validation_data = val_ds,
        epochs          = cfg["epochs"],
        callbacks       = build_callbacks(cfg),
        class_weight={0: 1.0, 1: 1.0},
        verbose         = 1,
    )
    log.info(f"학습 완료 — {(time.time()-t0)/3600:.1f}h")

    # ── STEP 9: 평가 (내부 + 외부)
    log.info("=" * 60)
    log.info("STEP 9: 최종 평가")
    log.info("=" * 60)
    int_results = evaluate_model(model, int_test_ds,
                                 tag="Internal Test (FF++/DFF/HIDF)")
    ext_results = evaluate_model(model, ext_test_ds,
                                 tag="External Test (CelebDF/redface/Eval-2024)")

    # ── STEP 10: 저장
    final_path = os.path.join(cfg["ckpt_dir"], f"final_model_{_ts}")
    model.save_weights(final_path)
    log.info(f"최종 모델 저장: {final_path}")

    return history, int_results, ext_results


if __name__ == "__main__":
    history, int_res, ext_res = main()