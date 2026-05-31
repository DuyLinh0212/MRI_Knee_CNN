# EfficientNetB0_App

Ung dung Angular de upload MRI goi theo 3 mat cat va goi API FastAPI de du doan:

- Bat thuong tong quat (`abnormal`)
- Ton thuong ACL (`acl`)
- Ton thuong sun chem (`meniscus`)

Ung dung doc trang thai model tu API `/health` va chi hien task nao co weight hop le.

## Yeu Cau

- Node.js phu hop voi Angular CLI 20.
- API dang chay tai `http://127.0.0.1:8000`.

## Cai Dat

```powershell
cd F:\NgDuyLinh\Do_an\DeepLearning_FN\EfficientNetB0_App
npm install
```

## Chay Ung Dung

```powershell
npm run start
```

Mo trinh duyet:

```text
http://127.0.0.1:4200
```

## Chay API Di Kem

Mo terminal khac:

```powershell
cd F:\NgDuyLinh\Do_an\DeepLearning_FN\EfficientNetB0_Api
py -3.13 main.py
```

Kiem tra API:

```text
http://127.0.0.1:8000/health
```

## Cach Su Dung

1. Chon model trong danh sach model API tra ve.
2. Upload file cho mat cat `Axial`.
3. Upload file cho mat cat `Coronal`.
4. Upload file cho mat cat `Sagittal`.
5. Bam chan doan.

Moi mat cat cho phep chon nhieu file, toi da theo `maxFilesPerPlane` API tra ve. Mac dinh API la `30`.

File ho tro phu thuoc API:

- `.npy`
- anh pho bien nhu `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`
- DICOM neu API co `pydicom`

Anh pho bien co preview tren trinh duyet. `.npy` va DICOM thuong chi hien ten file.

## API URL

URL API hien dang khai bao trong:

```text
src/app/app.ts
```

Gia tri hien tai:

```ts
http://127.0.0.1:8000
```

## Build

```powershell
npm run build
```

Output nam trong:

```text
dist/
```

## Luu Y

Ket qua chi phuc vu muc dich nghien cuu va ho tro ky thuat. Khong dung thay the chan doan y khoa.
