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
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pywt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision

# ── 로깅 ────────────────────────────────────────────────────────
os.makedirs("log", exist_ok=True)

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"log/train_{_ts}.log", encoding="utf-8"),
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
    i = np.arange(224).reshape(224, 1)
    j = np.arange(224).reshape(1, 224)
    return (i + j < 224).astype(np.float32)

N_BANDS    = 10
BAND_NAMES = ["LL3","LH3","HL3","HH3","LH2","HL2","HH2","LH1","HL1","HH1"]

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
    "unfreeze_epoch"   : 8,
    "rgb_embed_dim"    : 256,
    "wav_embed_dim": 256,  # ← 추가
    "dct_embed_dim"    : 128,
    "lm_embed_dim"     : 32,
    "attn_dim"         : 128,       # Cross-Attention 공통 차원
    "head_hidden"      : 64,        # 각 Head Dense 크기
    "dropout_rate"     : 0.4,
    "wav_num_heads": 8,  # ← 추가
    "wav_num_layers": 4,

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

LM_FEAT_DIM = 68
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

def extract_geometric_features_v2(lm: np.ndarray) -> np.ndarray:
    """
    기존 34차원 → 68차원으로 확장
    절대 좌표 기반 미세 변형 감지 추가
    """
    # 기존 34차원 비율 피처
    ratio_feats = extract_geometric_features(lm)  # (34,)

    # ── 추가 1: 인접 랜드마크 간 거리 변화 (국소 변형) ──────────
    # deepfake는 특정 부위(눈, 입 주변)에서 미세한 비연속성 발생
    local_dists = []
    # 눈 주변 6점 간 거리
    for i in range(36, 41):
        local_dists.append(float(np.linalg.norm(lm[i] - lm[i+1])))
    for i in range(42, 47):
        local_dists.append(float(np.linalg.norm(lm[i] - lm[i+1])))
    # 입 주변 12점 간 거리
    for i in range(48, 59):
        local_dists.append(float(np.linalg.norm(lm[i] - lm[i+1])))

    # 정규화
    face_w = float(np.linalg.norm(lm[16] - lm[0])) + 1e-8
    local_dists = np.array(local_dists, dtype=np.float32) / face_w

    # ── 추가 2: 랜드마크 곡률 (부드러움 정도) ────────────────────
    # deepfake는 랜드마크 경계가 부자연스럽게 꺾임
    curvatures = []
    for i in range(1, 18):  # 턱선
        v1 = lm[i]   - lm[i-1]
        v2 = lm[i+1] - lm[i]
        cos_sim = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        )
        curvatures.append(float(cos_sim))
    curvatures = np.array(curvatures, dtype=np.float32)

    feats = np.concatenate([ratio_feats, local_dists[:17], curvatures[:17]])
    return feats[:68].astype(np.float32)  # 68차원

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
    return extract_geometric_features_v2(lm)


# ── 기존 compute_dct_tf 삭제 후 아래로 교체 ──────────────────────

def _decompose_wavelet_np(img_np: np.ndarray) -> np.ndarray:
    """
    (224, 224, 3) float32 [0,1] → (10, 14, 14, 3) float32

    3단계 Haar 분해 → 각 서브밴드를 14×14로 리사이즈.
    14×14: CLIP ViT-B/16 내부 패치 그리드와 동일 해상도.
    공간-주파수 지역성 동시 보존 (DCT는 전역 변환이라 위치 정보 소실).
    """
    result = []
    for band_idx in range(N_BANDS):
        ch_list = []
        for c in range(3):
            ch = img_np[:, :, c]
            LL,  (LH1, HL1, HH1) = pywt.dwt2(ch,  "haar")
            LL2, (LH2, HL2, HH2) = pywt.dwt2(LL,  "haar")
            LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, "haar")
            bands = [LL3, LH3, HL3, HH3, LH2, HL2, HH2, LH1, HL1, HH1]

            b = bands[band_idx].astype(np.float32)
            # 14×14 리사이즈
            b_t = tf.image.resize(
                b[..., np.newaxis], [14, 14], method="bilinear"
            )
            b_norm = tf.squeeze(b_t, -1).numpy()
            # 서브밴드별 정규화 (스케일이 제각각이므로)
            b_min, b_max = b_norm.min(), b_norm.max()
            b_norm = (b_norm - b_min) / (b_max - b_min + 1e-8)
            ch_list.append(b_norm)

        result.append(np.stack(ch_list, axis=-1))   # (14, 14, 3)

    return np.stack(result, axis=0)                  # (10, 14, 14, 3)


def _load_wavelet(img_path_tensor, flip: bool = False) -> np.ndarray:
    """tf.py_function 래퍼 — 이미지 경로 → 웨이블릿 서브밴드"""
    path = img_path_tensor.numpy().decode()
    img  = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    img  = tf.cast(img, tf.float32) / 255.0
    if flip:
        img = tf.image.flip_left_right(img)
    return _decompose_wavelet_np(img.numpy())

def augment_face_no_flip(face: tf.Tensor) -> tf.Tensor:
    face = tf.image.random_brightness(face, max_delta=0.2)
    face = tf.image.random_contrast(face, lower=0.7, upper=1.3)
    face = tf.image.random_saturation(face, lower=0.7, upper=1.3)
    face = tf.cast(face * 255, tf.uint8)

    def jpeg_compress(img):
        q       = int(np.random.randint(50, 95))  # 75→50으로 하한 낮춤
        encoded = tf.image.encode_jpeg(img, quality=q)
        return tf.image.decode_jpeg(encoded, channels=3)

    face = tf.py_function(jpeg_compress, [face], tf.uint8)
    face.set_shape([224, 224, 3])
    face = tf.cast(face, tf.float32) / 255.0

    # 블러
    do_blur = tf.random.uniform([]) < 0.3
    blurred = tf.squeeze(
        tf.nn.avg_pool2d(tf.expand_dims(face, 0), ksize=3, strides=1, padding="SAME"), 0
    )
    face = tf.cond(do_blur, lambda: blurred, lambda: face)
    face.set_shape([224, 224, 3])

    # ── 추가: 다운샘플 → 업샘플 (압축 아티팩트 시뮬레이션) ──
    do_resize = tf.random.uniform([]) < 0.4
    small  = tf.image.resize(face, [112, 112])
    small  = tf.image.resize(small, [224, 224])
    face   = tf.cond(do_resize, lambda: small, lambda: face)
    face.set_shape([224, 224, 3])

    # ── 추가: 가우시안 노이즈 ──
    do_noise = tf.random.uniform([]) < 0.3
    noise  = tf.random.normal([224, 224, 3], mean=0.0, stddev=0.03)
    noisy  = tf.clip_by_value(face + noise, 0.0, 1.0)
    face   = tf.cond(do_noise, lambda: noisy, lambda: face)
    face.set_shape([224, 224, 3])

    # ── 추가: 랜덤 그레이스케일 (색상 의존성 감소) ──
    do_gray = tf.random.uniform([]) < 0.1
    gray   = tf.image.rgb_to_grayscale(face)
    gray   = tf.repeat(gray, 3, axis=-1)
    face   = tf.cond(do_gray, lambda: gray, lambda: face)
    face.set_shape([224, 224, 3])

    return tf.clip_by_value(face, 0.0, 1.0)

def make_load_fn(split: str):
    do_aug = (split == "train")

    def load_fn(face_p, lm_p, label):
        face = tf.image.decode_jpeg(tf.io.read_file(face_p), channels=3)
        face = tf.cast(face, tf.float32) / 255.0

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

        # ── 웨이블릿 (DCT 대체) ───────────────────────────────────
        # py_function: numpy 연산이라 CPU에서 실행, VRAM 영향 없음
        flipped = flip_flag > 0.5
        wav = tf.py_function(
            lambda p, f: _load_wavelet(p, flip=bool(f.numpy() > 0.5)),
            [face_p, flip_flag],
            tf.float32,
        )
        wav.set_shape([N_BANDS, 14, 14, 3])             # (10, 14, 14, 3)

        face = (face - MEAN) / STD
        face.set_shape([224, 224, 3])

        lm = tf.py_function(
            lambda p, f: _load_lm_geom(p, flipped=bool(f.numpy() > 0.5)),
            [lm_p, flip_flag],
            tf.float32,
        )
        lm.set_shape([LM_FEAT_DIM])

        return (
            {"face": face, "wav": wav, "lm": lm},       # dct → wav
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
        "wav" : lam * inputs["wav"]  + (1-lam) * tf.gather(inputs["wav"],  idx),  # dct→wav
        "lm"  : inputs["lm"],
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

# ── 4-B. WaveletSubbandViT (FAD 완전 대체) ─────────────────────

class BandPositionEmbedding(keras.layers.Layer):
    """
    서브밴드별 학습 가능한 위치 임베딩.
    LL3(저주파)와 HH1(고주파)에 다른 임베딩 부여
    → Transformer가 어떤 주파수 대역 토큰인지 인식
    """
    def __init__(self, n_bands: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.n_bands   = n_bands
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            shape=(1, self.n_bands, self.embed_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="band_position_embedding",
        )
        super().build(input_shape)

    def call(self, x):
        return tf.broadcast_to(self.pos_emb, tf.shape(x))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_bands": self.n_bands, "embed_dim": self.embed_dim})
        return cfg


class WavSubbandBlock(keras.layers.Layer):
    """
    Pre-Norm Transformer Block.
    서브밴드 간 self-attention:
      GAN   → HH 서브밴드에 체커보드 패턴 집중
      Diffusion → 모든 서브밴드에 균일 분산
    이 차이를 attention 가중치로 포착.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_r = dropout

    def build(self, input_shape):
        D, H = self.embed_dim, self.num_heads
        self.ln1   = layers.LayerNormalization(epsilon=1e-6)
        self.mha   = layers.MultiHeadAttention(
            num_heads=H, key_dim=D // H, dropout=self.dropout_r
        )
        self.drop1 = layers.Dropout(self.dropout_r)
        self.ln2   = layers.LayerNormalization(epsilon=1e-6)
        self.ffn1  = layers.Dense(D * 4, activation="gelu")
        self.ffn2  = layers.Dense(D)
        self.drop2 = layers.Dropout(self.dropout_r)
        super().build(input_shape)

    def call(self, x, training=False):
        h = self.ln1(x)
        h = self.mha(h, h, training=training)
        h = self.drop1(h, training=training)
        x = x + h
        h = self.ln2(x)
        h = self.ffn2(self.ffn1(h))
        h = self.drop2(h, training=training)
        return x + h

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim,
                    "num_heads": self.num_heads,
                    "dropout"  : self.dropout_r})
        return cfg


def build_wsv_branch(
    n_bands    : int   = N_BANDS,
    embed_dim  : int   = 256,
    num_heads  : int   = 8,
    num_layers : int   = 4,
    dropout    : float = 0.1,
) -> keras.Model:
    """
    입력: (B, 10, 14, 14, 3)
    출력: [(B, embed_dim), (B, 128), (B, 128)]
          메인 임베딩 + 저주파 aux + 고주파 aux
          (aux 출력 유지로 aux_lf / aux_hf 헤드와 호환)
    """
    patch_dim = 14 * 14 * 3    # 588

    inp = keras.Input(shape=(n_bands, 14, 14, 3), name="wav")

    # 1. Flatten: (B, 10, 588)
    x = layers.Reshape((n_bands, patch_dim), name="wav_flat")(inp)

    # 2. Linear 임베딩
    x = layers.TimeDistributed(
        layers.Dense(embed_dim, use_bias=False), name="wav_proj"
    )(x)
    x = layers.LayerNormalization(name="wav_proj_ln")(x)   # (B, 10, embed_dim)

    # 3. 밴드 위치 임베딩
    pos = BandPositionEmbedding(n_bands, embed_dim, name="band_pos")(x)
    x   = layers.Add(name="wav_pos_add")([x, pos])

    # 4. Transformer 블록 (서브밴드 간 attention)
    for i in range(num_layers):
        x = WavSubbandBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            name=f"wsv_block_{i}",
        )(x)

    # 5. 저주파(LL3~HH3, 0:4) / 고주파(HH2~HH1, 6:10) 분리 → aux용
    lf_feat = layers.GlobalAveragePooling1D(name="wsv_lf_gap")(x[:, :4, :])   # (B,embed_dim)
    hf_feat = layers.GlobalAveragePooling1D(name="wsv_hf_gap")(x[:, 6:, :])   # (B,embed_dim)

    # 6. 전체 평균 풀링 → 메인 임베딩
    out = layers.GlobalAveragePooling1D(name="wav_gap")(x)
    out = layers.LayerNormalization(name="wav_out_ln")(out)   # (B, embed_dim)

    return keras.Model(inp, [out, lf_feat, hf_feat], name="ws_vit")

# ── 4-C. Geometric Landmark MLP ────────────────────────────────

def build_lm_branch(feat_dim: int, embed_dim: int, dropout: float) -> keras.Model:
    """
    기존 MLP → Residual MLP
    얕은 구조의 표현력 한계 해소
    """
    inp = keras.Input(shape=(feat_dim,), name="lm")

    # 입력 정규화
    x = layers.LayerNormalization()(inp)

    # Block 1
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(dropout)(x)

    # Block 2 + Residual
    skip = layers.Dense(128, use_bias=False)(x)
    x    = layers.Dense(128, use_bias=False)(x)
    x    = layers.BatchNormalization()(x)
    x    = layers.Activation("gelu")(x)
    x    = layers.Dropout(dropout * 0.5)(x)
    x    = layers.Add()([x, skip])

    # Block 3 + Residual
    skip = x
    x    = layers.Dense(128, use_bias=False)(x)
    x    = layers.BatchNormalization()(x)
    x    = layers.Activation("gelu")(x)
    x    = layers.Add()([x, skip])

    # 출력
    x = layers.Dense(embed_dim, use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("gelu")(x)

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
    """
    기존 CrossAttention3Way 개선:
    각 모달리티의 신뢰도를 게이트로 학습
    → 약한 브랜치(DCT/LM)가 노이즈로 작용하는 것을 방지
    """
    def __init__(self, dim, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        _n = name or "gca3"
        self.mha   = layers.MultiHeadAttention(
            num_heads=8, key_dim=dim // 8, name=f"{_n}_mha"  # 헤드 수 4→8
        )
        self.ln1   = layers.LayerNormalization(name=f"{_n}_ln1")
        self.dense = layers.Dense(dim, use_bias=False, name=f"{_n}_out")
        self.ln2   = layers.LayerNormalization(name=f"{_n}_ln2")
        self.pool_query = layers.Dense(1, use_bias=False, name=f"{_n}_pool_q")

        # ── 추가: 모달리티별 게이트 ──────────────────────────────
        # 각 브랜치가 현재 샘플에서 얼마나 신뢰할 수 있는지 학습
        self.gate_rgb = layers.Dense(1, activation="sigmoid", name=f"{_n}_g_rgb")
        self.gate_dct = layers.Dense(1, activation="sigmoid", name=f"{_n}_g_dct")
        self.gate_lm  = layers.Dense(1, activation="sigmoid", name=f"{_n}_g_lm")

        # FFN (기존엔 없었음)
        self.ffn1 = layers.Dense(dim * 2, activation="gelu", name=f"{_n}_ffn1")
        self.ffn2 = layers.Dense(dim,     use_bias=False,    name=f"{_n}_ffn2")
        self.ln3  = layers.LayerNormalization(name=f"{_n}_ln3")

    def build(self, input_shape):
        self.modality_emb = self.add_weight(
            shape=(1, 3, self.dim),
            initializer="random_normal",
            trainable=True,
            name="modality_embedding",
        )
        super().build(input_shape)

    def call(self, inputs):
        feat_r, feat_d, feat_l = inputs

        # 1. 게이트로 각 브랜치 신뢰도 조절
        g_r = self.gate_rgb(feat_r)  # (B, 1)
        g_d = self.gate_dct(feat_d)
        g_l = self.gate_lm(feat_l)

        feat_r = feat_r * g_r
        feat_d = feat_d * g_d
        feat_l = feat_l * g_l

        # 2. 스택 + Modality Embedding
        seq     = tf.stack([feat_r, feat_d, feat_l], axis=1)  # (B,3,dim)
        seq_emb = seq + self.modality_emb

        # 3. Self-Attention
        attn = self.mha(query=seq_emb, key=seq_emb, value=seq)
        out  = self.ln1(seq + attn)

        # 4. FFN (기존 대비 추가)
        ffn  = self.ffn2(self.ffn1(out))
        out  = self.ln3(out + ffn)

        # 5. 평균 풀링
        attn_score = tf.nn.softmax(
            self.pool_query(out), axis=1
        )
        fused = tf.reduce_sum(
            out * attn_score, axis=1
        )
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
                initializer=keras.initializers.Constant(0.5),
                trainable=True,
                dtype=tf.float32,
            )
            for name in head_names
        }

    def call(self, inputs):
        return inputs   # 통과만 함 (가중치 보관 목적)

    def get_log_var(self, name):
        return tf.clip_by_value(self.log_vars[name], -2.0, 2.0)

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
    attn_dim = cfg["attn_dim"]

    if CLIP_AVAILABLE:
        rgb_ext = build_rgb_branch(img_size, cfg["rgb_embed_dim"],
                                   cfg["freeze_backbone"])
    else:
        rgb_ext = build_rgb_branch_fallback(img_size, cfg["rgb_embed_dim"],
                                            cfg["freeze_backbone"])

    # FAD → WS-ViT
    wav_ext = build_wsv_branch(
        n_bands   = N_BANDS,
        embed_dim = cfg["wav_embed_dim"],
        num_heads = cfg["wav_num_heads"],
        num_layers= cfg["wav_num_layers"],
    )
    lm_ext = build_lm_branch(LM_FEAT_DIM, cfg["lm_embed_dim"], dropout)

    inp_face = keras.Input(shape=(img_size, img_size, 3), name="face")
    inp_wav  = keras.Input(shape=(N_BANDS, 14, 14, 3),   name="wav")  # ← dct→wav
    inp_lm   = keras.Input(shape=(LM_FEAT_DIM,),          name="lm")

    rgb_raw             = rgb_ext(inp_face)
    wav_raw, lf_raw, hf_raw = wav_ext(inp_wav)            # ← 3 outputs
    lm_raw              = lm_ext(inp_lm)

    rgb_p = project_feat(rgb_raw, attn_dim, "rgb")
    wav_p = project_feat(wav_raw, attn_dim, "wav")        # ← dct_p → wav_p
    lm_p  = project_feat(lm_raw,  attn_dim, "lm")

    outputs = {}
    outputs["rgb"]     = build_head(rgb_p, "h_rgb",     hidden, dropout, reg)
    outputs["wav"]     = build_head(wav_p, "h_wav",     hidden, dropout, reg)
    outputs["lm"]      = build_head(lm_p,  "h_lm",      hidden, dropout, reg)

    rw = cross_attention_2(rgb_p, wav_p, attn_dim, "ca_rw")
    outputs["rgb_wav"] = build_head(rw, "h_rgb_wav", hidden, dropout, reg)

    rl = cross_attention_2(rgb_p, lm_p, attn_dim, "ca_rl")
    outputs["rgb_lm"]  = build_head(rl, "h_rgb_lm",  hidden, dropout, reg)

    wl = cross_attention_2(wav_p, lm_p, attn_dim, "ca_wl")
    outputs["wav_lm"]  = build_head(wl, "h_wav_lm",  hidden, dropout, reg)

    # Hierarchical: rgb+wav 먼저 융합 → lm과 최종 융합
    rw_fused    = cross_attention_2(rgb_p, wav_p, attn_dim, "ca_hier_rw")
    rwl_fused   = cross_attention_2(rw_fused, lm_p, attn_dim, "ca_hier_rwl")
    all_f       = CrossAttention3Way(attn_dim, name="ca_all")(
        [rgb_p, rwl_fused, lm_p]
    )
    outputs["all"] = build_head(all_f, "h_all", hidden, dropout, reg)

    outputs["aux_lf"] = build_aux_head(lf_raw, "aux_lf", reg)
    outputs["aux_hf"] = build_aux_head(hf_raw, "aux_hf", reg)

    return keras.Model(
        inputs  = {"face": inp_face, "wav": inp_wav, "lm": inp_lm},
        outputs = outputs,
        name    = "inner_wsv",
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
            loss = self._uncertainty_loss(y, y_pred)
            scaled_loss = self.optimizer.get_scaled_loss(loss)  # ← 이제 정상 작동

        scaled_grads = tape.gradient(scaled_loss, self.trainable_variables)
        grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

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
        y_hard = tf.cast(y >= 0.5, tf.float32)

        self.loss_tracker.update_state(loss)
        self.auc_metric.update_state(y_hard, y_pred["all"])
        self.acc_metric.update_state(y_hard, y_pred["all"])
        self.precision_metric.update_state(y_hard, y_pred["all"])
        self.recall_metric.update_state(y_hard, y_pred["all"])

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


class SmartEarlyStopping(keras.callbacks.EarlyStopping):
    """Resume 시 이전 best val_auc를 복원하는 EarlyStopping"""
    def __init__(self, initial_best=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_best = initial_best

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        if self.initial_best is not None:
            self.best = self.initial_best
            log.info(f"EarlyStopping best 복원: {self.best:.4f}")

def parse_best_auc_from_ckpt(ckpt_path: str) -> float:
    """체크포인트 파일명에서 val_auc 파싱"""
    m = re.search(r"valauc([\d.]+?)(?=\.index|$)", ckpt_path)
    return float(m.group(1)) if m else None

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

            # LossScaleOptimizer 대응
            opt = self.model.optimizer
            inner_opt = opt.inner_optimizer if hasattr(opt, "inner_optimizer") else opt

            current_lr = float(inner_opt.learning_rate(inner_opt.iterations)
                               if callable(inner_opt.learning_rate)
                               else inner_opt.learning_rate)
            new_lr = current_lr * self.lr_scale
            inner_opt.learning_rate = new_lr

            self._unfrozen = True
            log.info(f"Epoch {epoch}: backbone unfreeze | "
                     f"lr {current_lr:.2e} → {new_lr:.2e}")


class LogUncertaintyCallback(keras.callbacks.Callback):
    """에폭 끝마다 각 헤드의 log_var(불확실성) 출력"""
    def on_epoch_end(self, epoch, logs=None):
        lv = {
            name: float(self.model.uw_layer.get_log_var(name).numpy())
            for name in HEAD_NAMES
        }
        log.info(
            f"Epoch {epoch+1} log_vars | "
            + "  ".join(f"{k}={v:+.3f}" for k, v in lv.items())
        )


def _get_lr(model):
    opt = model.optimizer
    # LossScaleOptimizer 내부의 실제 optimizer 꺼내기
    if hasattr(opt, "inner_optimizer"):
        opt = opt.inner_optimizer
    sched = opt.learning_rate
    if callable(sched):
        return float(sched(opt.iterations))
    return float(sched)


def build_callbacks(cfg: dict, model, initial_best_auc=None) -> list:
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                cfg["ckpt_dir"],
                "best_epoch{epoch:03d}_valauc{val_auc:.4f}",
            ),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        SmartEarlyStopping(
            initial_best=initial_best_auc,
            monitor="val_auc",
            mode="max",
            patience=25,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001,
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

HEAD_NAMES     = ["rgb", "wav", "lm", "rgb_wav", "rgb_lm", "wav_lm", "all"]
AUX_HEAD_NAMES = ["aux_lf", "aux_hf"]
ALL_EVAL_HEADS = HEAD_NAMES + AUX_HEAD_NAMES

def evaluate_model(model: DeepfakeDetector, test_ds, tag="Test", n_tta=8):
    preds_all  = {name: [] for name in ALL_EVAL_HEADS}
    labels_all = []

    for inputs, labels in test_ds:
        tta_preds = {name: [] for name in ALL_EVAL_HEADS}

        for i in range(n_tta):
            # 짝수 회차: flip, 홀수 회차: 원본
            do_flip   = (i % 2 == 0)
            aug_face  = tf.image.flip_left_right(inputs["face"]) \
                        if do_flip else inputs["face"]
            aug_inputs = {"face": aug_face,
                          "dct" : inputs["dct"],
                          "lm"  : inputs["lm"]}

            out = model(aug_inputs, training=False)
            for name in ALL_EVAL_HEADS:
                tta_preds[name].append(out[name].numpy())

        for name in ALL_EVAL_HEADS:
            # 기하 평균 앙상블
            stack = np.stack(tta_preds[name], axis=0)          # (n_tta, B, 1)
            geo   = np.exp(np.mean(np.log(stack + 1e-8), axis=0))
            preds_all[name].append(geo.flatten())

        labels_all.append(labels.numpy().flatten())

    y_true = np.concatenate(labels_all)

    log.info(f"\n{'='*62}")
    log.info(f"  [{tag}]  Ablation Table  (TTA x{n_tta})")
    log.info(f"{'='*62}")
    log.info(f"  {'Head':<12}  {'AUC':>6}  {'F1':>6}  {'ACC':>6}  {'Thr':>5}")
    log.info(f"  {'-'*46}")

    results = {}
    for name in ALL_EVAL_HEADS:
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
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = steps_per_epoch * cfg["warmup_epochs"]

    # STEP 7.5: 가중치 로드 (컴파일 전에 resume 여부 파악)
    saved_ckpts = sorted(glob.glob(os.path.join(cfg["ckpt_dir"], "best_epoch*.index")))
    initial_best_auc = None
    is_resume = bool(saved_ckpts)

    # Resume이면 lr을 낮게 시작, 처음이면 정상 스케줄
    if is_resume:
        resume_lr_init = cfg["lr_init"] * 0.05  # 기존의 5%
        resume_warmup = steps_per_epoch * 2  # 2에폭 재웜업
        schedule = WarmupCosineDecay(
            resume_lr_init, cfg["lr_min"],
            int(resume_warmup),
            int(steps_per_epoch * cfg["epochs"]),
        )
        log.info(f"Resume 모드: lr_init {cfg['lr_init']:.2e} → {resume_lr_init:.2e}")
    else:
        schedule = WarmupCosineDecay(
            cfg["lr_init"], cfg["lr_min"],
            int(warmup_steps), int(total_steps),
        )

    base_optimizer = keras.optimizers.Adam(learning_rate=schedule, clipnorm=1.0)
    optimizer = keras.mixed_precision.LossScaleOptimizer(base_optimizer)
    model.compile(optimizer=optimizer)

    log.info(f"steps/epoch={steps_per_epoch:,}  total={total_steps:,}  warmup={warmup_steps:,}")

    if is_resume:
        latest_ckpt = saved_ckpts[-1].replace(".index", "")
        initial_best_auc = parse_best_auc_from_ckpt(saved_ckpts[-1])
        log.info(f"가중치 로드: {latest_ckpt}  (best_auc={initial_best_auc})")
        model.load_weights(latest_ckpt).expect_partial()
        log.info("✅ 가중치 로드 완료")
    else:
        log.info("새 학습 시작")

    # STEP 8: 학습
    initial_epoch = 0
    if is_resume:
        m = re.search(r"best_epoch(\d+)", saved_ckpts[-1])
        initial_epoch = int(m.group(1)) if m else 0
        log.info(f"Epoch {initial_epoch}부터 재개")

    log.info("=" * 60)
    log.info("STEP 8: 학습 시작")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        initial_epoch=initial_epoch,
        callbacks=build_callbacks(cfg, model, initial_best_auc),  # ← 전달
        verbose=1,
    )
    log.info(f"학습 완료 — {(time.time() - t0) / 3600:.2f}h")

    # STEP 9: 평가
    log.info("=" * 60)
    log.info("STEP 9: 최종 평가")
    int_res = evaluate_model(model, int_test_ds, "Internal (FF++/DFF/HIDF)")

    ext_res = {}  # ← 수정
    for ds_name in cfg["test_datasets"]:
        ds_samples = collect_samples(cfg["processed_root"], [ds_name])
        if not ds_samples:
            log.warning(f"[{ds_name}] 샘플 없음 — 스킵")
            continue
        ds_test = build_tf_dataset(ds_samples, "test", bs, seed)
        ext_res[ds_name] = evaluate_model(model, ds_test, f"External [{ds_name}]")  # ← 수정

    # STEP 10: 저장
    final_path = os.path.join(cfg["ckpt_dir"], f"final_{_ts}")
    model.save_weights(final_path)
    log.info(f"저장 완료: {final_path}")

    return history, int_res, ext_res


if __name__ == "__main__":
    history, int_res, ext_res = main()