# DeepLearning_train - Huấn luyện model MRI gối

Thư mục này chứa pipeline huấn luyện model deep learning để phân loại MRI gối theo 3 mặt cắt: `axial`, `coronal`, `sagittal`.

Các task đang hỗ trợ:

- `abnormal`: phát hiện bất thường tổng quát.
- `acl`: phát hiện tổn thương dây chằng chéo trước.
- `meniscus`: phát hiện tổn thương sụn chêm.

## Cấu trúc thư mục

```text
DeepLearning_train/
├── config.py                  # Cấu hình train mặc định
├── train.py                   # Script train chính
├── requirements.txt           # Dependency cài đặt
├── dataset/
│   └── dataset.py             # Dataset loader và DataLoader
├── models/
│   ├── EfficientNetB0.py      # Model EfficientNet-B0 cho 3 mặt cắt
│   └── Densenet121.py         # Model DenseNet121 cho 3 mặt cắt
├── preprocessing/             # Augmentation, resize, sampling, normalization
├── utils/                     # Hàm hỗ trợ train/evaluate
├── tools/                     # Công cụ phụ trợ
├── weights/                   # Checkpoint .pth sau khi train
└── evaluation/                # CSV metrics, ROC, confusion matrix, curves
```

## Cài đặt

```powershell
pip install -r requirements.txt
```

## Cấu trúc dữ liệu

Dữ liệu mặc định được đọc từ:

```text
data/
├── train/
│   ├── axial/
│   ├── coronal/
│   └── sagittal/
├── valid/
│   ├── axial/
│   ├── coronal/
│   └── sagittal/
└── test/
    ├── axial/
    ├── coronal/
    └── sagittal/
```

Mỗi file MRI là `.npy`, ví dụ:

```text
data/train/axial/0001.npy
data/train/coronal/0001.npy
data/train/sagittal/0001.npy
```

Label được đọc từ thư mục `labels/`:

```text
labels/
├── train-abnormal.csv
├── valid-abnormal.csv
├── test-abnormal.csv
├── train-acl.csv
├── valid-acl.csv
├── test-acl.csv
├── train-meniscus.csv
├── valid-meniscus.csv
└── test-meniscus.csv
```

Định dạng CSV:

```csv
id,label
```

Code hiện đọc CSV bằng `header=None`, nên file label thực tế không nên có dòng header.

## Cấu hình chính

Cấu hình nằm trong `config.py`. Các tham số thường chỉnh:

```python
'lr': 2e-5
'batch_size': 4
'image_size': 224
'target_slices': 24
'weight_decay': 1e-4
'patience': 5
'use_gradient_accumulation': 1
'gradient_accumulation_steps': 8
```

Effective batch size = `batch_size * gradient_accumulation_steps`.

## Train model

Chạy tất cả task bằng EfficientNet-B0:

```powershell
python train.py --model efficientnetb0 --tasks abnormal,acl,meniscus --data-root data --labels-root labels
```

Chạy từng task:

```powershell
python train.py --model efficientnetb0 --tasks abnormal --data-root data --labels-root labels
python train.py --model efficientnetb0 --tasks acl --data-root data --labels-root labels
python train.py --model efficientnetb0 --tasks meniscus --data-root data --labels-root labels
```

Warm-start ACL/Meniscus từ checkpoint Abnormal:

```powershell
python train.py --model efficientnetb0 --tasks acl,meniscus --abnormal-pth weights/abnormal/efficientnetb0_best_model.pth
```

## Output sau khi train

Checkpoint được lưu vào:

```text
weights/<task>/
├── efficientnetb0_best_model.pth
└── efficientnetb0_last_checkpoint.pth
```

Kết quả đánh giá được lưu vào:

```text
evaluation/efficientnetb0_<task>/
├── efficientnetb0_<task>_metrics.csv
├── efficientnetb0_<task>_test_metrics.csv
├── efficientnetb0_<task>_curves.png
├── efficientnetb0_<task>_roc.png
├── efficientnetb0_<task>_confusion.png
├── efficientnetb0_<task>_test_roc.png
└── efficientnetb0_<task>_test_confusion.png
```

## Pipeline hiện tại

- `BCEWithLogitsLoss`
- `Adam`
- `ReduceLROnPlateau` theo `val_auc`
- Early stopping theo `val_auc`
- Mixed precision khi có CUDA
- Gradient accumulation
- Lưu best checkpoint theo validation AUC
- Tìm best threshold theo F1 trên validation set
- Đánh giá test set bằng threshold tốt nhất từ validation

Kết quả model chỉ phục vụ nghiên cứu/hỗ trợ kỹ thuật, không thay thế chẩn đoán y khoa.
