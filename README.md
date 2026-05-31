# DeepLearning_train

Repo huan luyen mo hinh deep learning phan loai MRI khop goi theo 3 mat cat:

- `axial`
- `coronal`
- `sagittal`

Moi task la mot bai toan binary classification:

- `abnormal`: phat hien bat thuong tong quat.
- `acl`: phat hien ton thuong day chang cheo truoc.
- `meniscus`: phat hien ton thuong sun chem.

Pipeline hien tai doc volume MRI dang `.npy`, chuan hoa so slice/kich thuoc anh, dua 3 mat cat vao model PyTorch, train bang `BCEWithLogitsLoss`, luu checkpoint va sinh metric/plot danh gia.

## Cau truc thu muc

```text
DeepLearning_train/
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
|-- tools/
|   |-- split_dataset.py
|   |-- lr_finder.py
|   `-- convert_n5_efficientnet_pretrained.py
|-- utils/
|   |-- __init__.py
|   `-- utils.py
|-- weights/
`-- evaluation/
```

## Chuc nang tung file

### File chinh

| File | Chuc nang |
|---|---|
| `train.py` | File train chinh. Nhan tham so CLI, load data, khoi tao model, train/validation/test, tinh metric, tim best threshold theo F1, luu checkpoint va ve bieu do. |
| `config.py` | Chua cau hinh mac dinh: epoch, learning rate, batch size, image size, target slices, patience, gradient accumulation. |
| `requirements.txt` | Danh sach thu vien can cai: PyTorch, torchvision, numpy, pandas, scikit-learn, matplotlib, tensorboard, tqdm. |
| `README.md` | Tai lieu mo ta repo, du lieu, cach train va output. |

### Dataset va tien xu ly

| File | Chuc nang |
|---|---|
| `dataset/dataset.py` | Dinh nghia `MRData` va `load_data`. Doc label CSV, nap 3 file `.npy` theo 3 mat cat, resize/crop ve `image_size`, chuan hoa anh, tao tensor 3 kenh, tinh `pos_weight` cho loss va tao DataLoader train/valid/test. |
| `dataset/__init__.py` | Export `MRData` va `load_data` de import ngan gon bang `from dataset import load_data`. |
| `preprocessing/slice_sampling.py` | Ham `uniform_slice_sampling` noi suy/chon deu volume MRI ve so slice co dinh `target_slices`. |

### Model

| File | Chuc nang |
|---|---|
| `models/Densenet121.py` | Model DenseNet121 rieng cho 3 mat cat. Moi mat cat co backbone DenseNet121, gop dac trung theo slice bang max pooling, noi 3 vector dac trung va dua qua `Linear` ra 1 logit. |
| `models/EfficientNetB0.py` | Model EfficientNet-B0 rieng cho 3 mat cat. Cach xu ly tuong tu DenseNet121 nhung backbone la EfficientNet-B0, dac trung cuoi co kich thuoc 1280. |
| `models/EfficientNetB0_ViT.py` | Model EfficientNet-B0 ket hop Transformer nhe. EfficientNet trich dac trung tung slice, `SliceViT` gop dac trung theo chieu slice bang class token/positional embedding, sau do noi 3 mat cat de phan loai. |
| `models/__init__.py` | Export cac class model: `Densenet121`, `EfficientNetB0`, `EfficientNetB0_ViT`. |

### Tools va utils

| File | Chuc nang |
|---|---|
| `tools/split_dataset.py` | Tach du lieu tu train thanh train/valid/test theo ti le, di chuyen file `.npy` va ghi lai CSV label cho tung split. Co `--dry-run` de xem thu truoc khi move file. |
| `tools/lr_finder.py` | Chay LR finder cho `densenet121` hoac `efficientnetb0`, quet learning rate tu `--lr-start` den `--lr-end`, luu plot loss theo LR vao `evaluation/`. |
| `tools/convert_n5_efficientnet_pretrained.py` | Convert checkpoint EfficientNet-B0 tu format co `backbone.*` sang format 3 backbone `axial/coronal/sagittal` cua repo. |
| `utils/utils.py` | Ham train/evaluate phien ban cu va ham `_get_lr` dang duoc `train.py` dung de lay learning rate hien tai. |
| `utils/__init__.py` | Export cac helper trong `utils.py`. |

## Model ho tro

Tham so `--model` trong `train.py` ho tro:

```text
densenet121
efficientnetb0
efficientnetb0_vit
```

Model khuyen dung hien tai:

```text
efficientnetb0_vit
```

## Cau truc du lieu

`--data-root` phai tro toi thu muc co 3 split `train`, `valid`, `test`. Moi split co 3 mat cat:

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

Moi mau can co du 3 file `.npy` cung id:

```text
data/train/axial/0001.npy
data/train/coronal/0001.npy
data/train/sagittal/0001.npy
```

`--labels-root` phai tro toi thu muc label:

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

CSV khong co header vi code doc voi `header=None`. Moi dong gom `id,label`:

```csv
1,0
2,1
```

Code se tu chuyen id ve dang 4 chu so, vi du `1` thanh `0001`.

## Cai dat local

```powershell
cd F:\NgDuyLinh\Do_an\DeepLearning_FN\DeepLearning_train
pip install -r requirements.txt
```

## Train tren Kaggle

Trong Kaggle Notebook, bat GPU truoc khi train: `Settings` -> `Accelerator` -> chon `GPU`.

Neu repo nam trong `/kaggle/working/DeepLearning_train`, chay:

```python
%cd /kaggle/working/DeepLearning_train
!pip install -q -r requirements.txt
```

Khai bao duong dan dataset. Sua `DATASET_DIR` theo dung ten dataset tren Kaggle cua ban:

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

Train ca 3 task:

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

Neu bi thieu GPU memory, giam `--batch-size` hoac `--target-slices`:

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

Output se duoc luu theo thu muc dang dung de chay lenh. Neu `%cd /kaggle/working/DeepLearning_train` thi checkpoint nam tai:

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

## Tham so CLI cua train.py

| Tham so | Mac dinh | Y nghia |
|---|---:|---|
| `--model` | `efficientnetb0` | Chon model: `densenet121`, `efficientnetb0`, `efficientnetb0_vit`. |
| `--tasks` | `abnormal,acl,meniscus` | Danh sach task can train, ngan cach bang dau phay. |
| `--data-root` | `data` | Thu muc chua `train/valid/test` va 3 mat cat. |
| `--labels-root` | `labels` | Thu muc chua cac file CSV label. |
| `--batch-size` | theo `config.py` | Micro-batch size dua vao GPU. |
| `--target-slices` | theo `config.py` | So slice moi volume sau tien xu ly. |
| `--grad-accum-steps` | theo `config.py` | So buoc tich luy gradient. Effective batch size = `batch_size * grad_accum_steps`. |

## Cau hinh mac dinh

Mot so gia tri quan trong trong `config.py`:

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

Vi du:

```text
weights/abnormal/efficientnetb0_vit_best_model.pth
weights/acl/efficientnetb0_vit_best_model.pth
weights/meniscus/efficientnetb0_vit_best_model.pth
```

Metric va bieu do:

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

Trong qua trinh train, code cung ghi TensorBoard log thong qua `SummaryWriter`.

## Tom tat pipeline train

1. Doc label CSV va 3 volume `.npy` tu `axial`, `coronal`, `sagittal`.
2. Dua moi volume ve `target_slices` bang `uniform_slice_sampling`.
3. Center crop/resize ve `image_size x image_size`.
4. Chuan hoa intensity va tao tensor 3 kenh gia lap RGB.
5. Dua 3 mat cat vao model.
6. Train voi `BCEWithLogitsLoss(pos_weight=...)` va optimizer `Adam`.
7. Dung mixed precision khi co CUDA.
8. Dung gradient accumulation de tang effective batch size.
9. Scheduler `ReduceLROnPlateau` theo `val_auc`.
10. Early stopping khi `val_auc` khong cai thien sau `patience` epoch.
11. Luu best checkpoint theo validation AUC va last checkpoint de resume.
12. Tim threshold tot nhat theo F1 tren validation set.
13. Danh gia test set bang threshold tot nhat tu validation set.
14. Luu CSV metric, ROC curve, confusion matrix va training curves.

## Lenh phu tro

Tach train thanh train/valid/test:

```powershell
python tools/split_dataset.py --data-root data --label-root labels --task abnormal --ratios 0.7 0.15 0.15
```

Chay thu truoc khi move file:

```powershell
python tools/split_dataset.py --data-root data --label-root labels --task abnormal --dry-run
```

Tim learning rate phu hop:

```powershell
python tools/lr_finder.py --model efficientnetb0 --task acl --lr-start 1e-6 --lr-end 1e-2 --iters 100
```

Convert checkpoint EfficientNet-B0:

```powershell
python tools/convert_n5_efficientnet_pretrained.py --input path/to/input.pth --output weights/converted_efficientnetb0.pth
```

## Luu y

- `train.py` tu resume neu ton tai `weights/<task>/<model_name>_last_checkpoint.pth`.
- Neu muon train lai tu dau, doi ten/xoa checkpoint cu trong `weights/<task>/`.
- Test split la tuy chon. Neu khong co `test-<task>.csv` hoac `data/test`, code se bo qua phan danh gia test.
- Day la code nghien cuu/ky thuat, khong dung thay the chan doan y khoa cua bac si.
