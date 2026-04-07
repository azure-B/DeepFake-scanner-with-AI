import numpy as np
from pathlib import Path

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


def check_external_label_distribution(processed_root, test_datasets):
    """
    External 데이터셋 레이블 분포 확인.
    AUC < 0.5이면 레이블이 뒤집혔거나 예측 방향이 반전된 것.
    """
    root = Path(processed_root)
    for ds in test_datasets:
        ds_dir = root / ds
        if not ds_dir.exists():
            continue
        n0 = len(list((ds_dir / "0").glob("*_face.jpg"))) if (ds_dir/"0").exists() else 0
        n1 = len(list((ds_dir / "1").glob("*_face.jpg"))) if (ds_dir/"1").exists() else 0
        total = n0 + n1
        print(f"[{ds}] label=0: {n0:,} ({100*n0/total:.1f}%)  "
              f"label=1: {n1:,} ({100*n1/total:.1f}%)  total={total:,}")
        # 레이블 규칙: 0=real, 1=fake 인지 확인
        # CelebDF 원본은 real/fake 폴더가 다를 수 있음

# 실행
check_external_label_distribution(CFG["processed_root"], CFG["test_datasets"])

# 추가: 모델 예측값 분포도 확인
def check_prediction_distribution(model, ext_test_ds, n_batches=5):
    preds_real, preds_fake = [], []
    for i, (x, y) in enumerate(ext_test_ds):
        if i >= n_batches: break
        out = model(x, training=False)
        p = out["all"].numpy().flatten()
        y_np = y.numpy().flatten()
        preds_real.extend(p[y_np == 0])
        preds_fake.extend(p[y_np == 1])
    print(f"real 샘플 예측값 평균: {np.mean(preds_real):.3f}  "
          f"(높으면 모델이 real을 fake로 분류 중)")
    print(f"fake 샘플 예측값 평균: {np.mean(preds_fake):.3f}  "
          f"(낮으면 모델이 fake를 real로 분류 중)")