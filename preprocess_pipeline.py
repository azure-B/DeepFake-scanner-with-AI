"""
Unified Deepfake Detection Preprocessing Pipeline
지원 데이터셋: DFDC, DeepFakeFace(DFF), FF++, Celeb-DF v2
프레임워크: TensorFlow
얼굴 검출: OpenCV DNN YuNet (MTCNN TFLite hang 완전 대체)
랜드마크: MediaPipe FaceMesh (CPU)

[hang 원인 및 해결]
  - 구 MTCNN(pip install mtcnn)은 TFLite 세션을 내부에서 직접 열기 때문에
    Windows에서 TF GPU 컨텍스트와 충돌 → 프로세스 hang
  - YuNet(OpenCV DNN)은 TF와 완전히 독립적 → 충돌 없음
  - DCT 연산만 TF GPU에서 수행, 검출/랜드마크는 CPU 전용

[YuNet 모델 다운로드]
  아래 URL에서 face_detection_yunet_2023mar.onnx 를 받아
  스크립트와 같은 폴더(또는 CFG["yunet_model"] 경로)에 두세요.
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
"""

# ── 환경변수는 가장 먼저 ──────────────────────────────────────────────────────
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# ── 표준 라이브러리 ─────────────────────────────────────────────────────────
import cv2
import json
import logging
import numpy as np
from pathlib import Path

# ── MediaPipe (CPU 전용, TF 이전에 import해도 무방) ──────────────────────────
import mediapipe as mp

# ── TensorFlow — GPU 설정 완료 후 MTCNN/다른 TFLite 라이브러리 import ────────
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tqdm import tqdm

# ── 로거 설정 ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
CFG = {
    "crop_size"     : 224,
    "frame_interval": 10,        # 영상: N 프레임마다 1장 추출
    "max_frames"    : 30,        # 영상: 클립당 최대 프레임 수
    "lm_fail_limit" : 0.05,      # 랜드마크 실패율 허용 상한 (5 %)
    "output_root"   : "./processed",
    "seed"          : 42,
    # YuNet ONNX 모델 경로 (없으면 Haar Cascade 폴백)
    "yunet_model"   : "./face_detection_yunet_2023mar.onnx",
}

DATASETS = {
    "dff"    : ("./data/dff",     "parse_dff"),
    "ffpp"   : ("./data/ff++",    "parse_ffpp"),
    "celebdf": ("./data/celebdf", "parse_celebdf"),
    "hidf": ("./data/hidf", "parse_hidf"),
    "redface": ("./data/redface", "parse_redface"),
}

# ─────────────────────────────────────────────
# MediaPipe 468점 중 68점 선택 인덱스
# ─────────────────────────────────────────────
MP_68_INDICES = [
    # 얼굴 외곽 17점
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378, 152,
    # 왼쪽 눈썹 5점
    336, 296, 334, 293, 300,
    # 오른쪽 눈썹 5점
    107, 66, 105, 63, 70,
    # 코 9점
    168, 197, 195, 5, 4, 45, 275, 19, 94,
    # 왼쪽 눈 6점
    362, 382, 381, 380, 374, 373,
    # 오른쪽 눈 6점
    33, 160, 158, 133, 153, 144,
    # 입 외곽 12점
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 306,
    # 입 내부 8점
    78, 191, 80, 81, 82, 13, 312, 311,
]  # 총 68점

MP_REGIONS = {
    "jaw"   : [10,338,297,332,284,251,389,356,454,323,361,288,
               397,365,379,378,152,148,176,149,150,136,172,58,
               132,93,234,127,162,21,54,103,67,109],
    "brow_l": [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
    "brow_r": [107, 66, 105, 63, 70, 46, 53, 52, 65, 55],
    "nose"  : [168, 197, 195, 5, 4, 45, 275, 19, 94, 1, 2, 98, 327],
    "eye_l" : [362, 382, 381, 380, 374, 373, 390, 249, 263, 388, 387, 386, 385, 384],
    "eye_r" : [33, 160, 158, 133, 153, 144, 7, 163, 145, 154, 155, 157, 159, 161],
    "mouth" : [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146],
}


# ─────────────────────────────────────────────
# 1. LABEL PARSERS
# ─────────────────────────────────────────────

def parse_dff(root: str):
    samples, root = [], Path(root)
    for p in (root / "real").rglob("*.jpg"):
        samples.append((str(p), 0))
    for fake_dir in ["inpainting", "insight", "text2img"]:
        for p in (root / fake_dir).rglob("*.jpg"):
            samples.append((str(p), 1))
    log.info(f"[DFF] {len(samples)} samples")
    return samples


def parse_ffpp(root: str):
    samples, root = [], Path(root)

    for method in ["youtube","actors"]:
        for p in (root / f"original_sequences/{method}/c40/videos").glob("*.mp4"):
            samples.append((str(p), 0))

    for method in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        for p in (root / f"manipulated_sequences/{method}/c40/videos").glob("*.mp4"):
            samples.append((str(p), 1))
    log.info(f"[FF++] {len(samples)} samples")

    return samples

def parse_redface(root: str):
    """
    RedFace 구조:
      root/
        Original/   → real (label 0)
        EFS/        → Entire Face Synthesis  (label 1)
        FAM/        → Face Attribute Manipulation (label 1)
        FR/         → Face Reenactment (label 1)
        FS/         → Face Swapping (label 1)
    """
    samples, root = [], Path(root)

    # 진짜
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.mp4"]:
        for p in (root / "Original").rglob(ext):
            samples.append((str(p), 0))

    # 가짜 4종
    for fake_dir in ["EFS", "FAM", "FR", "FS"]:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.mp4"]:
            for p in (root / fake_dir).rglob(ext):
                samples.append((str(p), 1))

    log.info(f"[RedFace] {len(samples)} samples  "
             f"(real={sum(1 for _,l in samples if l==0)}, "
             f"fake={sum(1 for _,l in samples if l==1)})")
    return samples

def parse_celebdf(root: str):
    samples, root = [], Path(root)
    for p in (root / "Celeb-real").glob("*.mp4"):
        samples.append((str(p), 0))
    for p in (root / "YouTube-real").glob("*.mp4"):
        samples.append((str(p), 0))
    for p in (root / "Celeb-synthesis").glob("*.mp4"):
        samples.append((str(p), 1))
    log.info(f"[Celeb-DF] {len(samples)} samples")
    return samples

def parse_hidf(root: str):
    """
    HIDF 구조:
      root/
        Real-vid/   *.mp4  → label 0
        Fake-vid/   *.mp4  → label 1
        Real-img/   *.jpg/*.png  → label 0
        Fake-img/   *.jpg/*.png  → label 1
    """
    samples, root = [], Path(root)

    # 영상
    for p in (root / "Real-vid").glob("*.mp4"):
        samples.append((str(p), 0))
    for p in (root / "Fake-vid").glob("*.mp4"):
        samples.append((str(p), 1))

    # 이미지
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for p in (root / "Real-img").glob(ext):
            samples.append((str(p), 0))
        for p in (root / "Fake-img").glob(ext):
            samples.append((str(p), 1))

    log.info(f"[HIDF] {len(samples)} samples  "
             f"(real={sum(1 for _,l in samples if l==0)}, "
             f"fake={sum(1 for _,l in samples if l==1)})")
    return samples




PARSERS = {
    "parse_dff"    : parse_dff,
    "parse_ffpp"   : parse_ffpp,
    "parse_celebdf": parse_celebdf,
    "parse_hidf" : parse_hidf,
    "parse_redFace" : parse_redface
}


# ─────────────────────────────────────────────
# 2. FRAME EXTRACTOR
# ─────────────────────────────────────────────

def extract_frames(video_path: str, interval: int, max_frames: int) -> list:
    """
    이미지 파일이면 그대로 반환, 영상이면 interval마다 프레임 추출.
    반환: BGR ndarray 리스트
    """
    ext = Path(video_path).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = cv2.imread(video_path)
        return [img] if img is not None else []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"영상 열기 실패: {video_path}")
        return []

    frames, idx = [], 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


# ─────────────────────────────────────────────
# 3. FACE DETECTOR — OpenCV YuNet (TF 충돌 없음)
# ─────────────────────────────────────────────

class FaceDetector:
    """
    YuNet (OpenCV DNN, ONNX) 기반 얼굴 검출기.
    TFLite MTCNN 대체 → TF GPU 컨텍스트와 완전히 분리.

    YuNet 모델이 없으면 Haar Cascade로 폴백.
    """

    def __init__(self, crop_size: int, model_path: str = ""):
        self.size       = crop_size
        self.detector   = None
        self.use_yunet  = False
        self._init_detector(model_path)

    # ── 초기화 ───────────────────────────────────────────────────────────────
    def _init_detector(self, model_path: str):
        if model_path and Path(model_path).exists():
            try:
                # OpenCV 4.8+ FaceDetectorYN
                self.detector  = cv2.FaceDetectorYN.create(
                    model_path,
                    "",
                    (self.size, self.size),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=1,
                )
                self.use_yunet = True
                log.info(f"YuNet 초기화 완료: {model_path}")
                return
            except Exception as e:
                log.warning(f"YuNet 초기화 실패 ({e}) → Haar Cascade 폴백")

        # Haar Cascade 폴백
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector  = cv2.CascadeClassifier(cascade_path)
        self.use_yunet = False
        log.info("Haar Cascade 초기화 완료 (폴백)")

    # ── 공개 메서드 ──────────────────────────────────────────────────────────
    def detect(self, bgr_frame: np.ndarray):
        """
        BGR ndarray → 크롭 & 리사이즈된 BGR ndarray (224×224).
        검출 실패 시 None 반환.
        """
        if self.use_yunet:
            return self._detect_yunet(bgr_frame)
        return self._detect_haar(bgr_frame)

    def center_crop_fallback(self, bgr_frame: np.ndarray):
        h, w   = bgr_frame.shape[:2]
        s      = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        return cv2.resize(bgr_frame[y0:y0+s, x0:x0+s], (self.size, self.size))

    # ── 내부 구현 ─────────────────────────────────────────────────────────────
    def _detect_yunet(self, bgr_frame: np.ndarray):
        h, w = bgr_frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr_frame)

        if faces is None or len(faces) == 0:
            return None

        # 신뢰도 최고 얼굴 선택 (첫 번째가 top_k=1 기준 최고)
        x, y, fw, fh = [int(v) for v in faces[0][:4]]
        x, y = max(0, x), max(0, y)
        crop = bgr_frame[y:y+fh, x:x+fw]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (self.size, self.size))

    def _detect_haar(self, bgr_frame: np.ndarray):
        gray  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None

        # 가장 큰 얼굴 선택
        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        x, y = max(0, x), max(0, y)
        crop = bgr_frame[y:y+fh, x:x+fw]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (self.size, self.size))


# ─────────────────────────────────────────────
# 4. LANDMARK EXTRACTOR (MediaPipe FaceMesh — CPU)
# ─────────────────────────────────────────────

class LandmarkExtractor:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self.fail_count = 0
        self.total      = 0

    # ── 공개 메서드 ──────────────────────────────────────────────────────────
    def extract(self, bgr_face: np.ndarray):
        """
        Returns
        -------
        lm      : np.ndarray, shape (68, 2), dtype float32 — 픽셀 좌표
        success : bool
        """
        self.total += 1
        h, w   = bgr_face.shape[:2]
        rgb    = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            self.fail_count += 1
            return self._uniform_fallback(), False

        all_lm = result.multi_face_landmarks[0].landmark
        lm = np.array(
            [[all_lm[i].x * w, all_lm[i].y * h] for i in MP_68_INDICES],
            dtype=np.float32,
        )
        return lm, True

    def region_masks(self, bgr_face: np.ndarray, img_size: int = 224):
        """영역별 binary mask — Region-Aware Patch 분할용"""
        mask   = {k: np.zeros((img_size, img_size), dtype=np.uint8) for k in MP_REGIONS}
        rgb    = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return mask

        all_lm = result.multi_face_landmarks[0].landmark
        for key, indices in MP_REGIONS.items():
            pts  = np.array(
                [[int(all_lm[i].x * img_size), int(all_lm[i].y * img_size)]
                 for i in indices],
                dtype=np.int32,
            )
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask[key], hull, 255)
        return mask

    @property
    def fail_rate(self):
        return self.fail_count / max(self.total, 1)

    def _uniform_fallback(self):
        xs   = np.linspace(20, 204, 9)
        ys   = np.linspace(20, 204, 8)
        grid = np.array([[x, y] for y in ys for x in xs])
        return grid[:68].astype(np.float32)

    def close(self):
        self.face_mesh.close()

# ─────────────────────────────────────────────
# 6. SAVE UTILS
# ─────────────────────────────────────────────

def save_sample(out_dir: Path, stem: str, face: np.ndarray,
                lm: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stem}_face.jpg"), face,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.save(str(out_dir / f"{stem}_lm.npy"),  lm)


# ─────────────────────────────────────────────
# 7. TF DATASET 빌더
# ─────────────────────────────────────────────

def build_tf_dataset(processed_root: str, split: str = "train",
                     batch_size: int = 32, seed: int = 42):
    """
    processed/ 폴더를 읽어 tf.data.Dataset 반환.

    split : 'train' | 'val' | 'test'  (7:1:2 분할)

    반환 배치 구조
    --------------
    {
      "face"    : (B, 224, 224, 3)  float32  — ImageNet 정규화 RGB
      "landmark": (B, 68, 2)        float32  — 0~1 정규화 좌표
    },
    labels : (B,) int32
    """
    root        = Path(processed_root)
    all_samples = []

    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        for label_str in ["0", "1"]:
            label_dir = ds_dir / label_str
            if not label_dir.exists():
                continue
            label = int(label_str)
            for face_path in sorted(label_dir.glob("*_face.jpg")):
                stem     = face_path.stem.replace("_face", "")
                lm_path  = label_dir / f"{stem}_lm.npy"
                if lm_path.exists():
                    all_samples.append((
                        str(face_path), str(lm_path), label
                    ))

    if not all_samples:
        raise RuntimeError(
            f"[tf.data] 전처리된 샘플 없음: {processed_root}\n"
            "run_pipeline()을 먼저 실행하세요."
        )

    # 재현 가능한 셔플 후 7:1:2 분할
    rng = np.random.default_rng(seed)
    rng.shuffle(all_samples)
    n = len(all_samples)
    splits = {
        "train": all_samples[:int(n * 0.7)],
        "val"  : all_samples[int(n * 0.7):int(n * 0.8)],
        "test" : all_samples[int(n * 0.8):],
    }
    chosen = splits[split]
    log.info(f"[tf.data] {split} split — {len(chosen)}개 샘플")

    # ImageNet 정규화 상수
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    # ── .npy 헤더-없이 raw bytes 파싱 ──────────────────────────────────────
    # np.save 헤더: magic(6) + major(1) + minor(1) + header_len(2 or 4) + header
    # 헤더 크기가 가변적이므로 py_function으로 안전하게 로드
    # (헤더 고정 가정 시 깨질 수 있어 py_function 유지)
    def _load_npy(path_tensor):
        return np.load(path_tensor.numpy().decode("utf-8")).astype(np.float32)

    def load_sample(face_p, lm_p, label):
        # ── 얼굴 이미지 ──────────────────────────────────────────────────────
        face = tf.image.decode_jpeg(tf.io.read_file(face_p), channels=3)
        face = (tf.cast(face, tf.float32) / 255.0 - mean) / std   # (224,224,3)
        # ── 랜드마크 ─────────────────────────────────────────────────────────
        lm = tf.py_function(_load_npy, [lm_p], tf.float32)
        lm.set_shape([68, 2])
        lm = lm / 224.0    # 픽셀 → [0, 1] 정규화

        return (
            {"face": face, "landmark": lm},
            tf.cast(label, tf.int32),
        )

    face_paths = [s[0] for s in chosen]
    lm_paths   = [s[1] for s in chosen]
    labels     = [s[2] for s in chosen]

    ds = tf.data.Dataset.from_tensor_slices(
        (face_paths, lm_paths, labels)
    )

    # ── train: 클래스 균형 가중 셔플 ────────────────────────────────────────
    if split == "train":
        n_real = labels.count(0)
        n_fake = labels.count(1)
        w_real = 1.0 / (n_real + 1e-8)
        w_fake = 1.0 / (n_fake + 1e-8)
        weights = [w_real if lb == 0 else w_fake for lb in labels]
        weights_ds = tf.data.Dataset.from_tensor_slices(
            tf.constant(weights, dtype=tf.float32)
        )
        ds = tf.data.Dataset.zip((ds, weights_ds))
        ds = ds.map(
            lambda sample, _w: load_sample(*sample),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.shuffle(buffer_size=2000, seed=seed)
    else:
        ds = ds.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    ds = (
        ds
        .batch(batch_size, drop_remainder=(split == "train"))
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


# ─────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(dataset_keys: list = None):
    """
    전처리 실행 후 processed_root 경로 반환.

    dataset_keys : None이면 DATASETS 전체 실행
    """
    if dataset_keys is None:
        dataset_keys = list(DATASETS.keys())

    detector    = FaceDetector(CFG["crop_size"], CFG["yunet_model"])
    extractor   = LandmarkExtractor()
    out_root    = Path(CFG["output_root"])
    total_saved = 0

    # ✅ 추가 — 이미 처리된 stem 전체를 set으로 미리 수집
    log.info("기처리 샘플 스캔 중...")
    done_stems = set()
    for existing in out_root.rglob("*_lm.npy"):
        # {stem}_lm.npy → stem 추출
        done_stems.add(existing.stem.replace("_lm", ""))
    log.info(f"기처리 샘플: {len(done_stems):,}개")

    try:
        for ds_key in dataset_keys:
            root, parser_name = DATASETS[ds_key]
            if not Path(root).exists():
                log.warning(f"[{ds_key}] 경로 없음: {root} — 스킵")
                continue

            samples = PARSERS[parser_name](root)
            if not samples:
                log.warning(f"[{ds_key}] 파일 없음 — 스킵")
                continue

            for file_path, label in tqdm(samples, desc=f"[{ds_key}]"):

                check_stem = f"{Path(file_path).stem}_f000"

                if check_stem in done_stems:  # ← 이 줄로
                    continue

                frames = extract_frames(
                    file_path,
                    interval=CFG["frame_interval"],
                    max_frames=CFG["max_frames"],
                )
                for fi, frame in enumerate(frames):
                    if frame is None:
                        continue

                    # 얼굴 검출 (실패 시 중앙 크롭)
                    face = detector.detect(frame)
                    if face is None:
                        face = detector.center_crop_fallback(frame)

                    # 랜드마크 추출
                    lm, _ = extractor.extract(face)


                    # 저장
                    stem    = f"{Path(file_path).stem}_f{fi:03d}"
                    out_dir = out_root / ds_key / str(label)
                    save_sample(out_dir, stem, face, lm)
                    total_saved += 1

            # 랜드마크 실패율 경고
            if extractor.fail_rate > CFG["lm_fail_limit"]:
                log.warning(
                    f"[{ds_key}] 랜드마크 실패율 {extractor.fail_rate:.1%} "
                    f"(허용 상한 {CFG['lm_fail_limit']:.0%} 초과)"
                )

    finally:
        extractor.close()

    log.info(f"완료 — 총 {total_saved}개 샘플 저장 → {out_root}")
    return str(out_root)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── GPU 환경 확인 ────────────────────────────────────────────────────────
    _gpus = tf.config.list_physical_devices("GPU")
    if _gpus:
        log.info(f"GPU 감지: {[g.name for g in _gpus]}")
    else:
        log.warning("GPU 없음 — CPU로 실행됩니다 (DCT 속도 저하)")

    # ── 데이터 경로 사전 확인 ────────────────────────────────────────────────
    log.info("── 데이터셋 경로 확인 ──")
    for key, (root, _) in DATASETS.items():
        status = "✓ 존재" if Path(root).exists() else "✗ 없음"
        log.info(f"  [{key:8s}] {root}  {status}")

    # ── YuNet 모델 확인 ──────────────────────────────────────────────────────
    if Path(CFG["yunet_model"]).exists():
        log.info(f"YuNet 모델: {CFG['yunet_model']}  ✓")
    else:
        log.warning(
            f"YuNet 모델 없음: {CFG['yunet_model']}\n"
            "  → Haar Cascade 폴백으로 실행됩니다.\n"
            "  → 다운로드: https://github.com/opencv/opencv_zoo/raw/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        )

    # ── 전처리 실행 ──────────────────────────────────────────────────────────
    processed_root = run_pipeline(dataset_keys=["celebdf","ffpp","hidf","redface","dff"])

    # ── tf.data 검증 ─────────────────────────────────────────────────────────
    if not any(Path(processed_root).rglob("*_face.jpg")):
        log.error("전처리된 파일 없음 — tf.data 테스트 스킵")
    else:
        log.info("── tf.data 검증 시작 ──")
        train_ds = build_tf_dataset(processed_root, split="train", batch_size=8)
        for batch, lbls in train_ds.take(1):
            print(f"  face     : {batch['face'].shape}")      # (8, 224, 224, 3)
            print(f"  landmark : {batch['landmark'].shape}")  # (8, 68, 2)
            print(f"  labels   : {lbls.numpy()}")
        log.info("tf.data 정상 동작 ✓")