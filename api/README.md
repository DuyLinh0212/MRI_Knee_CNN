# EfficientNetB0_Api

FastAPI service de du doan chan thuong goi tu anh MRI theo 3 mat cat:

- `axial`
- `coronal`
- `sagittal`

API co the load cac checkpoint cho 3 task:

- `abnormal`
- `acl`
- `meniscus`

API ho tro 2 kieu model:

- `efficientnetb0`
- `efficientnetb0_vit`

## Cau Truc

```text
EfficientNetB0_Api/
+-- main.py
+-- models.py
+-- requirements.txt
+-- README.md
+-- weights/
    +-- abnormal/
    +-- acl/
    +-- meniscus/
```

## Vi Tri Weight

API tu tim checkpoint trong cac thu muc sau:

1. `EfficientNetB0_Api/weights`
2. `DeepLearning_train/weights`
3. `DeepLearning_v/weights`

Moi task co thu muc rieng:

```text
weights/abnormal/
weights/acl/
weights/meniscus/
```

Vi du voi model `efficientnetb0_vit`:

```text
EfficientNetB0_Api/weights/abnormal/efficientnetb0_vit_best_model.pth
EfficientNetB0_Api/weights/acl/efficientnetb0_vit_best_model.pth
EfficientNetB0_Api/weights/meniscus/efficientnetb0_vit_best_model.pth
```

Neu co nhieu file `.pth`, API uu tien file co chu `best`, sau do den `last`.

## Cai Dat

```powershell
cd F:\NgDuyLinh\Do_an\DeepLearning_FN\EfficientNetB0_Api
py -3.13 -m pip install -r requirements.txt
```

Neu may khong co Python 3.13, co the dung Python khac da cai du PyTorch/FastAPI:

```powershell
python -m pip install -r requirements.txt
```

## Chay API

Chay truc tiep:

```powershell
py -3.13 main.py
```

Hoac chay bang Uvicorn:

```powershell
py -3.13 -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Kiem tra:

```text
http://127.0.0.1:8000/health
```

## Bien Moi Truong

```powershell
$env:TARGET_SLICES="24"
$env:MAX_FILES_PER_PLANE="30"
```

Mac dinh:

- `TARGET_SLICES=24`
- `MAX_FILES_PER_PLANE=30`

## Endpoint

### GET `/health`

Tra ve:

- Trang thai API.
- Device dang dung.
- So slice muc tieu.
- So file toi da moi mat cat.
- Danh sach model/task da load duoc weight.

### POST `/predict`

Nhan `multipart/form-data`.

Field bat buoc:

- `axial`
- `coronal`
- `sagittal`

Field tuy chon:

- `threshold`: nguong du doan, mac dinh `0.5`.
- `modelName`: ten model can dung, vi du `efficientnetb0_vit`.

Dinh dang file ho tro:

- `.npy`: slice 2D hoac volume 3D.
- `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`.
- `.dcm`, `.dicom` neu da cai `pydicom`.

Moi mat cat upload toi da `MAX_FILES_PER_PLANE` file. Neu upload 1 file `.npy` volume 3D thi tinh la 1 file.

## Vi Du curl

```powershell
curl -X POST http://127.0.0.1:8000/predict `
  -F "modelName=efficientnetb0_vit" `
  -F "threshold=0.5" `
  -F "axial=@data/axial.npy" `
  -F "coronal=@data/coronal.npy" `
  -F "sagittal=@data/sagittal.npy"
```

## Hau Xu Ly

Neu `acl` hoac `meniscus` duong tinh thi `abnormal` cung duoc danh dau duong tinh theo suy luan. Xac suat goc cua model abnormal van duoc giu trong response.

## Luu Y

API chi phuc vu muc dich nghien cuu va ho tro ky thuat. Khong dung thay the chan doan y khoa.
