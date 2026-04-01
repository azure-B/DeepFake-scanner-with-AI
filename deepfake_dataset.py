"""
Deepfake Detection — TensorFlow Dataset & tf.data Pipeline
preprocess_pipeline.py 실행 후 생성된 processed/ 폴더를 읽어서 사용
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ImageNet 정규화 상수
MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


# ─────────────────────────────────────────────
# 1. 증강 함수 (train 전용)
# ─────────────────────────────────────────────

def augment(face: tf.Tensor) -> tf.Tensor:
    """
    train split에만 적용
    입력/출력: (224, 224, 3) float32 [0, 1]
    """
    face = tf.image.random_flip_left_right(face)
    face = tf.image.random_brightness(face, max_delta=0.2)
    face = tf.image.random_contrast(face, lower=0.8, upper=1.2)
    face = tf.image.random_saturation(face, lower=0.9, upper=1.1)
    # JPEG 압축 시뮬레이션 (deepfake artifact 강화)
    face = tf.cast(face * 255, tf.uint8)
    face = tf.image.encode_jpeg(face, quality=tf.random.uniform([], 70, 100, dtype=tf.int32))
    face = tf.cast(tf.image.decode_jpeg(face, channels=3), tf.float32) / 255.0
    return face


# ─────────────────────────────────────────────
# 2. 샘플 로더
# ─────────────────────────────────────────────

def _load_npy(path: tf.Tensor, shape: list, dtype=np.float32) -> tf.Tensor:
    """tf.py_function으로 .npy 파일 로드"""
    arr = np.load(path.numpy().decode()).astype(dtype)
    return arr


def make_load_fn(split: str):
    """
    split에 따라 증강 여부 다르게 적용하는 로드 함수 반환
    """
    do_aug = (split == "train")

    def load_fn(face_p, lm_p, dct_p, label):
        # ── 얼굴 이미지 ──
        raw  = tf.io.read_file(face_p)
        face = tf.image.decode_jpeg(raw, channels=3)
        face = tf.cast(face, tf.float32) / 255.0          # [0, 1]

        if do_aug:
            face = augment(face)

        face = (face - MEAN) / STD                         # ImageNet 정규화
        face.set_shape([224, 224, 3])

        # ── DCT 맵 ──
        dct_arr = tf.py_function(
            lambda p: _load_npy(p, [224, 224, 3]),
            [dct_p], tf.float32,
        )
        dct_arr.set_shape([224, 224, 3])

        # ── 랜드마크 (0~1 정규화) ──
        lm = tf.py_function(
            lambda p: (_load_npy(p, [68, 2]) / 224.0),
            [lm_p], tf.float32,
        )
        lm.set_shape([68, 2])

        return (
            {"face": face, "dct": dct_arr, "landmark": lm},
            tf.cast(label, tf.int32),
        )

    return load_fn


# ─────────────────────────────────────────────
# 3. 샘플 수집 (processed/ 폴더 스캔)
# ─────────────────────────────────────────────

def collect_samples(processed_root: str, datasets: list = None) -> list:
    """
    Returns: [(face_path, lm_path, dct_path, label), ...]
    """
    root    = Path(processed_root)
    samples = []

    ds_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if datasets:
        ds_dirs = [d for d in ds_dirs if d.name in datasets]

    for ds_dir in ds_dirs:
        for label_str in ["0", "1"]:
            label_dir = ds_dir / label_str
            if not label_dir.exists():
                continue
            label = int(label_str)
            for face_path in sorted(label_dir.glob("*_face.jpg")):
                stem     = face_path.stem.replace("_face", "")
                lm_path  = label_dir / f"{stem}_lm.npy"
                dct_path = label_dir / f"{stem}_dct.npy"
                if lm_path.exists() and dct_path.exists():
                    samples.append((str(face_path), str(lm_path), str(dct_path), label))

    return samples


# ─────────────────────────────────────────────
# 4. tf.data.Dataset 빌더
# ─────────────────────────────────────────────

def build_dataset(
    processed_root: str,
    split: str      = "train",      # 'train' | 'val' | 'test'
    split_ratio: tuple = (0.7, 0.1, 0.2),
    batch_size: int = 32,
    datasets: list  = None,         # None이면 전체, 아니면 ['celebdf','dff',...]
    seed: int       = 42,
) -> tf.data.Dataset:
    """
    반환 배치 구조:
        inputs  = {
            "face"    : (B, 224, 224, 3)  float32  — 정규화된 RGB
            "dct"     : (B, 224, 224, 3)  float32  — DCT 맵
            "landmark": (B, 68, 2)        float32  — 0~1 정규화 좌표
        }
        labels  = (B,) int32

    사용 예:
        ds = build_dataset("./processed", split="train", batch_size=32)
        for inputs, labels in ds:
            face = inputs["face"]       # (32, 224, 224, 3)
            dct  = inputs["dct"]        # (32, 224, 224, 3)
            lm   = inputs["landmark"]   # (32, 68, 2)
    """
    all_samples = collect_samples(processed_root, datasets)

    # 재현 가능한 셔플 후 분할
    rng = np.random.default_rng(seed)
    idx = np.arange(len(all_samples))
    rng.shuffle(idx)
    all_samples = [all_samples[i] for i in idx]

    n      = len(all_samples)
    tr_end = int(n * split_ratio[0])
    va_end = tr_end + int(n * split_ratio[1])

    chosen = {
        "train": all_samples[:tr_end],
        "val"  : all_samples[tr_end:va_end],
        "test" : all_samples[va_end:],
    }[split]

    # 클래스 분포 로그
    n_real = sum(1 for *_, l in chosen if l == 0)
    n_fake = sum(1 for *_, l in chosen if l == 1)
    log.info(f"[{split}] total={len(chosen)}  real={n_real}  fake={n_fake}")

    face_paths = [s[0] for s in chosen]
    lm_paths   = [s[1] for s in chosen]
    dct_paths  = [s[2] for s in chosen]
    labels     = [s[3] for s in chosen]

    ds = tf.data.Dataset.from_tensor_slices(
        (face_paths, lm_paths, dct_paths, labels)
    )

    load_fn = make_load_fn(split)
    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # train: 균형 샘플링 + 셔플
    if split == "train":
        n_real = max(n_real, 1)
        n_fake = max(n_fake, 1)
        w      = [1.0 / n_real if l == 0 else 1.0 / n_fake for l in labels]
        w_ds   = tf.data.Dataset.from_tensor_slices(w)

        # (inputs, label, weight) 형태로 zip 후 weighted 셔플
        ds = tf.data.Dataset.zip((ds, w_ds))
        ds = ds.map(lambda xy, w: (*xy, w))        # (inputs, label, weight)
        ds = ds.shuffle(buffer_size=2000, seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda inputs, label, w: (inputs, label))  # weight 제거

    ds = (
        ds
        .batch(batch_size, drop_remainder=(split == "train"))
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


# ─────────────────────────────────────────────
# 5. 전체 split 한 번에 반환
# ─────────────────────────────────────────────

def build_all_datasets(
    processed_root: str,
    batch_size: int = 32,
    datasets: list  = None,
    seed: int       = 42,
) -> dict:
    """
    train / val / test Dataset 딕셔너리로 반환

    사용 예:
        dsets = build_all_datasets("./processed", batch_size=32)
        for inputs, labels in dsets["train"]:
            ...
    """
    return {
        split: build_dataset(
            processed_root=processed_root,
            split=split,
            batch_size=batch_size,
            datasets=datasets,
            seed=seed,
        )
        for split in ["train", "val", "test"]
    }


# ─────────────────────────────────────────────
# 6. 동작 확인
# ─────────────────────────────────────────────

if __name__ == "__main__":
    dsets = build_all_datasets("./processed", batch_size=8)

    print("\n=== 배치 shape 확인 ===")
    for split, ds in dsets.items():
        for inputs, labels in ds.take(1):
            print(f"[{split}]")
            print(f"  face     : {inputs['face'].shape}")      # (8, 224, 224, 3)
            print(f"  dct      : {inputs['dct'].shape}")       # (8, 224, 224, 3)
            print(f"  landmark : {inputs['landmark'].shape}")  # (8, 68, 2)
            print(f"  labels   : {labels.numpy()}")
    print("\n✅ tf.data.Dataset 정상 동작")