# DeepLearning_train

Pipeline huan luyen mo hinh deep learning de phan loai MRI goi theo 3 mat cat:

- `axial`
- `coronal`
- `sagittal`

Repo ho tro 3 task:

- `abnormal`: phat hien bat thuong tong quat.
- `acl`: phat hien ton thuong day chang cheo truoc.
- `meniscus`: phat hien ton thuong sun chem.

## Cau Truc

```text
DeepLearning_train/
+-- config.py
+-- train.py
+-- requirements.txt
+-- dataset/
|   +-- dataset.py
+-- models/
|   +-- Densenet121.py
|   +-- EfficientNetB0.py
|   +-- EfficientNetB0_ViT.py
+-- preprocessing/
+-- tools/
+-- utils/
+-- weights/
+-- evaluation/
```

## Model

`train.py` ho tro cac gia tri `--model`:

- `densenet121`
- `efficientnetb0`
- `efficientnetb0_vit`

Model chinh nen dung hien tai:

```text
efficientnetb0_vit
```

Huong train hien tai la train doc lap tung task hoac train nhieu task trong cung mot lenh.

## Cai Dat

```powershell
cd F:\NgDuyLinh\Do_an\DeepLearning_FN\DeepLearning_train
pip install -r requirements.txt
```

Tren Kaggle/Colab, cai dependency theo notebook neu moi truong da co san PyTorch thi khong can cai lai toan bo.

## Cau Truc Du Lieu

Thu muc anh:

```text
data/
+-- train/
|   +-- axial/
|   +-- coronal/
|   +-- sagittal/
+-- valid/
|   +-- axial/
|   +-- coronal/
|   +-- sagittal/
+-- test/
    +-- axial/
    +-- coronal/
    +-- sagittal/
```

Moi mau can co du 3 file `.npy` cung id:

```text
data/train/axial/0001.npy
data/train/coronal/0001.npy
data/train/sagittal/0001.npy
```

Thu muc label:

```text
labels/
+-- train-abnormal.csv
+-- valid-abnormal.csv
+-- test-abnormal.csv
+-- train-acl.csv
+-- valid-acl.csv
+-- test-acl.csv
+-- train-meniscus.csv
+-- valid-meniscus.csv
+-- test-meniscus.csv
```

CSV khong nen co header vi code doc voi `header=None`. Dinh dang moi dong:

```csv
0001,1
0002,0
```

## Cau Hinh

Tham so mac dinh nam trong `config.py`.

Mot so tham so quan trong:

```python
'lr': 2e-5
'batch_size': 8
'image_size': 224
'target_slices': 32
'weight_decay': 1e-4
'patience': 5
'gradient_accumulation_steps': 4
```

Effective batch size:

```text
batch_size * gradient_accumulation_steps
```

## Train Tren Kaggle

Train mot task:

```python
!python train.py \
  --model efficientnetb0_vit \
  --tasks abnormal \
  --data-root /kaggle/input/datasets/zuylyn/nhom5-deeplearning-dataset/data \
  --labels-root /kaggle/input/datasets/zuylyn/nhom5-deeplearning-dataset/labels \
  --batch-size 8 \
  --target-slices 24 \
  --grad-accum-steps 4
```

Sau khi train xong, checkpoint se nam o:

```text
/kaggle/working/weights/<task>/efficientnetb0_vit_best_model.pth
```

Train nhieu task:

```python
!python train.py \
  --model efficientnetb0_vit \
  --tasks acl,meniscus \
  --data-root /kaggle/input/datasets/zuylyn/nhom5-deeplearning-dataset/data \
  --labels-root /kaggle/input/datasets/zuylyn/nhom5-deeplearning-dataset/labels \
  --batch-size 8 \
  --target-slices 24 \
  --grad-accum-steps 4
```

## Train Local

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

## Output

Checkpoint:

```text
weights/<task>/<model_name>_best_model.pth
weights/<task>/<model_name>_last_checkpoint.pth
```

Vi du:

```text
weights/abnormal/efficientnetb0_vit_best_model.pth
weights/acl/efficientnetb0_vit_best_model.pth
weights/meniscus/efficientnetb0_vit_best_model.pth
```

Ket qua danh gia:

```text
evaluation/<model_name>_<task>/
+-- <model_name>_<task>_metrics.csv
+-- <model_name>_<task>_test_metrics.csv
+-- <model_name>_<task>_curves.png
+-- <model_name>_<task>_roc.png
+-- <model_name>_<task>_confusion.png
+-- <model_name>_<task>_test_roc.png
+-- <model_name>_<task>_test_confusion.png
```

## Pipeline

- Resize/crop ve `image_size`.
- Lay mau deu ve `target_slices`.
- Chuan hoa cuong do anh.
- Tao input 3 kenh cho backbone torchvision.
- Dung `BCEWithLogitsLoss`.
- Dung `Adam`.
- Dung `ReduceLROnPlateau` theo `val_auc`.
- Early stopping theo `val_auc`.
- Mixed precision khi co CUDA.
- Gradient accumulation.
- Luu best checkpoint theo validation AUC.
- Tim best threshold theo F1 tren validation set.
- Danh gia test set bang best threshold tu validation.
- Loc/sanitize `NaN/Inf` de tranh `Val Loss = NaN`.

## Luu Y

Ket qua chi phuc vu muc dich nghien cuu va ho tro ky thuat. Khong dung thay the chan doan y khoa cua bac si.
