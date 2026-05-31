# DeepLearning_train

Repo huấn luyện mô hình deep learning để phân loại MRI khớp gối theo 3 mặt cắt:

- `axial`
- `coronal`
- `sagittal`

Mỗi task là một bài toán phân loại nhị phân:

- `abnormal`: phát hiện bất thường tổng quát.
- `acl`: phát hiện tổn thương dây chằng chéo trước.
- `meniscus`: phát hiện tổn thương sụn chêm.

Pipeline hiện tại đọc volume MRI dạng `.npy`, chuẩn hóa số slice/kích thước ảnh, đưa 3 mặt cắt vào model PyTorch, train bằng `BCEWithLogitsLoss`, lưu checkpoint và sinh metric/plot đánh giá.

## Cấu trúc thư mục

```text
DeepLearning_train/
|-- api (api để web gọi đến)
|-- application (thực nghiệm chuẩn đoán)
|-- config.py
|-- train.py
|-- requirements.txt
|-- README.md
|-- dataset/
|   |-- __init__.py
|   `-- dataset.py
|-- models/
|   |-- __init__.py
|   |-- Densenet121.py
|   |-- EfficientNetB0.py
|   `-- EfficientNetB0_ViT.py
|-- preprocessing/
|   `-- slice_sampling.py
|-- utils/
|   |-- __init__.py
|   `-- utils.py
|-- weights/
`-- evaluation/
```

## Chức năng từng file

### File chính

| File | Chức năng |
|---|---|
| `train.py` | File train chính. Nhận tham số CLI, load data, khởi tạo model, train/validation/test, tính metric, tìm best threshold theo F1, lưu checkpoint và vẽ biểu đồ. |
| `config.py` | Chứa cấu hình mặc định: epoch, learning rate, batch size, image size, target slices, patience, gradient accumulation. |
| `requirements.txt` | Danh sách thư viện cần cài: PyTorch, torchvision, numpy, pandas, scikit-learn, matplotlib, tensorboard, tqdm. |
| `README.md` | Tài liệu mô tả repo, dữ liệu, cách train và output. |

### Dataset và tiền xử lý

| File | Chức năng |
|---|---|
| `dataset/dataset.py` | Định nghĩa `MRData` và `load_data`. Đọc label CSV, nạp 3 file `.npy` theo 3 mặt cắt, resize/crop về `image_size`, chuẩn hóa ảnh, tạo tensor 3 kênh, tính `pos_weight` cho loss và tạo DataLoader train/valid/test. |
| `dataset/__init__.py` | Export `MRData` và `load_data` để import ngắn gọn bằng `from dataset import load_data`. |
| `preprocessing/slice_sampling.py` | Hàm `uniform_slice_sampling` nội suy/chọn đều volume MRI về số slice cố định `target_slices`. |

### Model

| File | Chức năng |
|---|---|
| `models/Densenet121.py` | Model DenseNet121 riêng cho 3 mặt cắt. Mỗi mặt cắt có backbone DenseNet121, gộp đặc trưng theo slice bằng max pooling, nối 3 vector đặc trưng và đưa qua `Linear` ra 1 logit. |
| `models/EfficientNetB0.py` | Model EfficientNet-B0 riêng cho 3 mặt cắt. Cách xử lý tương tự DenseNet121 nhưng backbone là EfficientNet-B0, đặc trưng cuối có kích thước 1280. |
| `models/EfficientNetB0_ViT.py` | Model EfficientNet-B0 kết hợp Transformer nhẹ. EfficientNet trích đặc trưng từng slice, `SliceViT` gộp đặc trưng theo chiều slice bằng class token/positional embedding, sau đó nối 3 mặt cắt để phân loại. |
| `models/__init__.py` | Export các class model: `Densenet121`, `EfficientNetB0`, `EfficientNetB0_ViT`. |

### Utils

| File | Chức năng |
|---|---|
| `utils/utils.py` | Hàm train/evaluate phiên bản cũ và hàm `_get_lr` đang được `train.py` dùng để lấy learning rate hiện tại. |
| `utils/__init__.py` | Export các helper trong `utils.py`. |

## Model hỗ trợ

Tham số `--model` trong `train.py` hỗ trợ:

```text
densenet121
efficientnetb0
efficientnetb0_vit
```

Model khuyên dùng hiện tại:

```text
efficientnetb0_vit
```

## Cấu trúc dữ liệu

`--data-root` phải trỏ tới thư mục có 3 split `train`, `valid`, `test`. Mỗi split có 3 mặt cắt:

```text
data/
|-- train/
|   |-- axial/
|   |-- coronal/
|   `-- sagittal/
|-- valid/
|   |-- axial/
|   |-- coronal/
|   `-- sagittal/
`-- test/
    |-- axial/
    |-- coronal/
    `-- sagittal/
```

Mỗi mẫu cần có đủ 3 file `.npy` cùng id:

```text
data/train/axial/0001.npy
data/train/coronal/0001.npy
data/train/sagittal/0001.npy
```

`--labels-root` phải trỏ tới thư mục label:

```text
labels/
|-- train-abnormal.csv
|-- valid-abnormal.csv
|-- test-abnormal.csv
|-- train-acl.csv
|-- valid-acl.csv
|-- test-acl.csv
|-- train-meniscus.csv
|-- valid-meniscus.csv
`-- test-meniscus.csv
```

CSV không có header vì code đọc với `header=None`. Mỗi dòng gồm `id,label`:

```csv
1,0
2,1
```

Code sẽ tự chuyển id về dạng 4 chữ số, ví dụ `1` thành `0001`.

## Cài đặt local

```powershell
cd F:\NgDuyLinh\Do_an\DeepLearning_FN\DeepLearning_train
pip install -r requirements.txt
```

## Train trên Kaggle

Trong Kaggle Notebook, bật GPU trước khi train: `Settings` -> `Accelerator` -> chọn `GPU`.

Nếu repo nằm trong `/kaggle/working/DeepLearning_train`, chạy:

```python
%cd /kaggle/working/DeepLearning_train
!pip install -q -r requirements.txt
```

Khai báo đường dẫn dataset. Sửa `DATASET_DIR` theo đúng tên dataset trên Kaggle của bạn:

```python
DATASET_DIR = "/kaggle/input/nhom5-deeplearning-dataset"
DATA_ROOT = f"{DATASET_DIR}/data"
LABELS_ROOT = f"{DATASET_DIR}/labels"
```

Train 1 task:

```python
!python train.py \
  --model efficientnetb0_vit \
  --tasks abnormal \
  --data-root "$DATA_ROOT" \
  --labels-root "$LABELS_ROOT" \
  --batch-size 8 \
  --target-slices 24 \
  --grad-accum-steps 4
```

Train cả 3 task:

```python
!python train.py \
  --model efficientnetb0_vit \
  --tasks abnormal,acl,meniscus \
  --data-root "$DATA_ROOT" \
  --labels-root "$LABELS_ROOT" \
  --batch-size 8 \
  --target-slices 24 \
  --grad-accum-steps 4
```

Nếu bị thiếu GPU memory, giảm `--batch-size` hoặc `--target-slices`:

```python
!python train.py \
  --model efficientnetb0_vit \
  --tasks acl \
  --data-root "$DATA_ROOT" \
  --labels-root "$LABELS_ROOT" \
  --batch-size 4 \
  --target-slices 16 \
  --grad-accum-steps 4
```

Output sẽ được lưu theo thư mục đang dùng để chạy lệnh. Nếu `%cd /kaggle/working/DeepLearning_train` thì checkpoint nằm tại:

```text
/kaggle/working/DeepLearning_train/weights/<task>/<model_name>_best_model.pth
/kaggle/working/DeepLearning_train/weights/<task>/<model_name>_last_checkpoint.pth
```

## Train local

```powershell
cd F:\NgDuyLinh\Do_an\DeepLearning_FN\DeepLearning_train

python train.py `
  --model efficientnetb0_vit `
  --tasks abnormal `
  --data-root data `
  --labels-root labels `
  --batch-size 8 `
  --target-slices 24 `
  --grad-accum-steps 4
```

## Tham số CLI của train.py

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--model` | `efficientnetb0` | Chọn model: `densenet121`, `efficientnetb0`, `efficientnetb0_vit`. |
| `--tasks` | `abnormal,acl,meniscus` | Danh sách task cần train, ngăn cách bằng dấu phẩy. |
| `--data-root` | `data` | Thư mục chứa `train/valid/test` và 3 mặt cắt. |
| `--labels-root` | `labels` | Thư mục chứa các file CSV label. |
| `--batch-size` | theo `config.py` | Micro-batch size đưa vào GPU. |
| `--target-slices` | theo `config.py` | Số slice mỗi volume sau tiền xử lý. |
| `--grad-accum-steps` | theo `config.py` | Số bước tích lũy gradient. Effective batch size = `batch_size * grad_accum_steps`. |

## Cấu hình mặc định

Một số giá trị quan trọng trong `config.py`:

```python
config = {
    "max_epoch": 50,
    "lr": 2e-5,
    "batch_size": 8,
    "weight_decay": 1e-4,
    "patience": 5,
    "image_size": 224,
    "target_slices": 32,
    "num_workers": 2,
    "use_gradient_accumulation": 1,
    "gradient_accumulation_steps": 4,
}
```

## Output sinh ra

Checkpoint:

```text
weights/<task>/<model_name>_best_model.pth
weights/<task>/<model_name>_last_checkpoint.pth
```

Ví dụ:

```text
weights/abnormal/efficientnetb0_vit_best_model.pth
weights/acl/efficientnetb0_vit_best_model.pth
weights/meniscus/efficientnetb0_vit_best_model.pth
```

Metric và biểu đồ:

```text
evaluation/<model_name>_<task>/
|-- <model_name>_<task>_metrics.csv
|-- <model_name>_<task>_test_metrics.csv
|-- <model_name>_<task>_curves.png
|-- <model_name>_<task>_roc.png
|-- <model_name>_<task>_confusion.png
|-- <model_name>_<task>_test_roc.png
`-- <model_name>_<task>_test_confusion.png
```

Trong quá trình train, code cũng ghi TensorBoard log thông qua `SummaryWriter`.

## Tóm tắt pipeline train

1. Đọc label CSV và 3 volume `.npy` từ `axial`, `coronal`, `sagittal`.
2. Đưa mỗi volume về `target_slices` bằng `uniform_slice_sampling`.
3. Center crop/resize về `image_size x image_size`.
4. Chuẩn hóa intensity và tạo tensor 3 kênh giả lập RGB.
5. Đưa 3 mặt cắt vào model.
6. Train với `BCEWithLogitsLoss(pos_weight=...)` và optimizer `Adam`.
7. Dùng mixed precision khi có CUDA.
8. Dùng gradient accumulation để tăng effective batch size.
9. Scheduler `ReduceLROnPlateau` theo `val_auc`.
10. Early stopping khi `val_auc` không cải thiện sau `patience` epoch.
11. Lưu best checkpoint theo validation AUC và last checkpoint để resume.
12. Tìm threshold tốt nhất theo F1 trên validation set.
13. Đánh giá test set bằng threshold tốt nhất từ validation set.
14. Lưu CSV metric, ROC curve, confusion matrix và training curves.

## Lưu ý

- `train.py` tự resume nếu tồn tại `weights/<task>/<model_name>_last_checkpoint.pth`.
- Nếu muốn train lại từ đầu, đổi tên/xóa checkpoint cũ trong `weights/<task>/`.
- Test split là tùy chọn. Nếu không có `test-<task>.csv` hoặc `data/test`, code sẽ bỏ qua phần đánh giá test.
- Đây là code nghiên cứu/kỹ thuật, không dùng thay thế chẩn đoán y khoa của bác sĩ.
