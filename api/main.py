import io
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from models import Densenet121, EfficientNetB0, EfficientNetB0ViT

try:
    import pydicom
except ImportError:
    pydicom = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TASKS = ("abnormal", "acl", "meniscus")
PLANES = ("axial", "coronal", "sagittal")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "efficientnetb0").lower()
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
TARGET_SLICES = int(os.getenv("TARGET_SLICES", "24"))
MAX_FILES_PER_PLANE = int(os.getenv("MAX_FILES_PER_PLANE", "30"))
MEAN = 58.09
STDDEV = 49.73
MAX_PIXEL_VAL = 255.0

WEIGHT_ROOTS = [
    BASE_DIR / "weights",
    PROJECT_ROOT / "DeepLearning_train" / "weights",
    PROJECT_ROOT / "DeepLearning_v" / "weights",
]

TASK_LABELS = {
    "abnormal": "Bất thường",
    "acl": "Tổn thương ACL",
    "meniscus": "Tổn thương sụn chêm",
}

MODEL_REGISTRY = {
    "efficientnetb0": {
        "label": "EfficientNet-B0",
        "factory": EfficientNetB0,
        "aliases": ("efficientnetb0", "efficientnet_b0"),
    },
    "densenet121": {
        "label": "DenseNet121",
        "factory": Densenet121,
        "aliases": ("densenet121", "dense_net121"),
    },
    "efficientnetb0_vit": {
        "label": "EfficientNet-B0 + ViT",
        "factory": EfficientNetB0ViT,
        "aliases": ("efficientnetb0_vit", "efficientnet_b0_vit", "efficientnetb0vit"),
    },
}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary.")

    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


def _weight_sort_key(path: Path):
    name = path.name.lower()
    priority = 0 if "best" in name else 1 if "last" in name else 2
    return priority, name


def _normalize_model_name(model_name: str) -> str:
    normalized = (model_name or DEFAULT_MODEL).strip().lower()
    if normalized not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unsupported modelName: {model_name}")
    return normalized


def _matches_model(path: Path, model_name: str) -> bool:
    filename = path.name.lower()
    aliases = MODEL_REGISTRY[model_name]["aliases"]
    return any(alias in filename for alias in aliases)


def find_weight_path(model_name: str, task: str) -> Optional[Path]:
    env_path = os.getenv(f"{model_name.upper()}_{task.upper()}_PTH")
    if not env_path and model_name == DEFAULT_MODEL:
        env_path = os.getenv(f"{task.upper()}_PTH")
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return candidate

    candidates: List[Path] = []
    for root in WEIGHT_ROOTS:
        candidates.extend(path for path in (root / task).glob("*.pth") if _matches_model(path, model_name))
        candidates.extend(path for path in root.glob(f"*{task}*.pth") if _matches_model(path, model_name))

    if not candidates:
        return None

    return sorted(candidates, key=_weight_sort_key)[0]


class DiagnosisService:
    def __init__(self):
        self.device = _device()
        self.models: Dict[tuple[str, str], torch.nn.Module] = {}
        self.weight_paths: Dict[tuple[str, str], Optional[Path]] = {}
        self.weight_mtimes: Dict[tuple[str, str], Optional[float]] = {}
        self.errors: Dict[tuple[str, str], Optional[str]] = {}

    def refresh_models(self, model_name: Optional[str] = None):
        model_items = (
            [(model_name, MODEL_REGISTRY[model_name])]
            if model_name is not None
            else MODEL_REGISTRY.items()
        )
        for model_name, model_config in model_items:
            for task in TASKS:
                key = (model_name, task)
                weight_path = find_weight_path(model_name, task)
                mtime = weight_path.stat().st_mtime if weight_path else None

                if weight_path is None:
                    self.models.pop(key, None)
                    self.weight_paths[key] = None
                    self.weight_mtimes[key] = None
                    self.errors[key] = None
                    continue

                if (
                    key in self.models
                    and self.weight_paths.get(key) == weight_path
                    and self.weight_mtimes.get(key) == mtime
                ):
                    continue

                try:
                    checkpoint = torch.load(weight_path, map_location=self.device)
                    model = model_config["factory"]().to(self.device)
                    model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
                    model.eval()
                    self.models[key] = model
                    self.weight_paths[key] = weight_path
                    self.weight_mtimes[key] = mtime
                    self.errors[key] = None
                except Exception as exc:
                    self.models.pop(key, None)
                    self.weight_paths[key] = weight_path
                    self.weight_mtimes[key] = mtime
                    self.errors[key] = str(exc)

    def _task_statuses(self, model_name: str):
        statuses = []
        for task in TASKS:
            key = (model_name, task)
            path = self.weight_paths.get(key) or find_weight_path(model_name, task)
            error = self.errors.get(key)
            statuses.append(
                {
                    "task": task,
                    "label": TASK_LABELS[task],
                    "available": path is not None and error is None,
                    "path": str(path) if path else None,
                    "error": error,
                }
            )
        return statuses

    def status(self):
        selected_model = DEFAULT_MODEL if DEFAULT_MODEL in MODEL_REGISTRY else "efficientnetb0"
        return {
            "status": "ok",
            "device": str(self.device),
            "defaultModel": selected_model,
            "imageSize": IMAGE_SIZE,
            "targetSlices": TARGET_SLICES,
            "maxFilesPerPlane": MAX_FILES_PER_PLANE,
            "modelOptions": [
                {
                    "name": model_name,
                    "label": config["label"],
                    "availableCount": sum(
                        1
                        for status in self._task_statuses(model_name)
                        if status["available"]
                    ),
                    "tasks": self._task_statuses(model_name),
                }
                for model_name, config in MODEL_REGISTRY.items()
            ],
            "models": self._task_statuses(selected_model),
        }

    @torch.inference_mode()
    def predict(self, volumes: Dict[str, torch.Tensor], threshold: float, model_name: str):
        model_name = _normalize_model_name(model_name)
        self.refresh_models(model_name)
        if not any(model == model_name for model, _ in self.models):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No .pth model was found for {MODEL_REGISTRY[model_name]['label']}. "
                    "Add weights to EfficientNetB0_Api/weights/<task>."
                ),
            )

        inputs = [volumes[plane].to(self.device) for plane in PLANES]
        tasks = {}

        for task in TASKS:
            key = (model_name, task)
            if key not in self.models:
                tasks[task] = {
                    "label": TASK_LABELS[task],
                    "available": False,
                    "prediction": None,
                    "probability": None,
                    "rawProbability": None,
                    "inferred": False,
                    "inferredFrom": [],
                    "weightPath": str(self.weight_paths.get(key)) if self.weight_paths.get(key) else None,
                    "message": "Chưa có file .pth cho tác vụ này.",
                }
                continue

            logits = self.models[key](inputs)
            probability = torch.sigmoid(logits).item()
            prediction = probability >= threshold
            tasks[task] = {
                "label": TASK_LABELS[task],
                "available": True,
                "prediction": prediction,
                "probability": probability,
                "rawProbability": probability,
                "inferred": False,
                "inferredFrom": [],
                "weightPath": str(self.weight_paths[key]),
                "message": "Dương tính" if prediction else "Âm tính",
            }

        specific_positive_tasks = [
            task
            for task in ("acl", "meniscus")
            if tasks.get(task, {}).get("prediction") is True
        ]
        if specific_positive_tasks and tasks.get("abnormal", {}).get("available"):
            abnormal = tasks["abnormal"]
            if abnormal.get("prediction") is not True:
                abnormal["prediction"] = True
                abnormal["inferred"] = True
                abnormal["inferredFrom"] = specific_positive_tasks
                inferred_labels = ", ".join(TASK_LABELS[task] for task in specific_positive_tasks)
                abnormal["message"] = f"Dương tính (suy luận từ {inferred_labels})"

        positive_labels = [
            TASK_LABELS[task]
            for task in TASKS
            if tasks.get(task, {}).get("prediction") is True
        ]

        if positive_labels:
            diagnosis = "Phát hiện dấu hiệu: " + ", ".join(positive_labels) + "."
        else:
            diagnosis = "Chưa phát hiện dấu hiệu bất thường theo các model hiện có."

        return {
            "diagnosis": diagnosis,
            "modelName": model_name,
            "modelLabel": MODEL_REGISTRY[model_name]["label"],
            "threshold": threshold,
            "device": str(self.device),
            "tasks": tasks,
            "availableModels": [task for task in TASKS if (model_name, task) in self.models],
        }


service = DiagnosisService()
app = FastAPI(title="Knee Injury Diagnosis API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|192\.168\.\d+\.\d+)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_image(bytes_data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(bytes_data)).convert("L")
    return np.asarray(image, dtype=np.float32)


def _read_dicom(bytes_data: bytes) -> np.ndarray:
    if pydicom is None:
        raise HTTPException(
            status_code=400,
            detail="DICOM upload requires pydicom. Install it with: pip install pydicom",
        )
    dataset = pydicom.dcmread(io.BytesIO(bytes_data))
    return dataset.pixel_array.astype(np.float32)


async def _files_to_volume(files: List[UploadFile], plane: str) -> np.ndarray:
    if not files:
        raise HTTPException(status_code=400, detail=f"Missing {plane} files.")
    if len(files) > MAX_FILES_PER_PLANE:
        raise HTTPException(
            status_code=400,
            detail=f"{plane} chỉ được upload tối đa {MAX_FILES_PER_PLANE} file.",
        )

    slices = []
    for upload in sorted(files, key=lambda item: item.filename or ""):
        suffix = Path(upload.filename or "").suffix.lower()
        content = await upload.read()
        if not content:
            continue

        try:
            if suffix == ".npy":
                array = np.load(io.BytesIO(content)).astype(np.float32)
                if array.ndim == 2:
                    slices.append(array)
                elif array.ndim == 3:
                    slices.extend(array)
                else:
                    raise ValueError(".npy must be a 2D slice or 3D volume.")
            elif suffix in {".dcm", ".dicom"}:
                slices.append(_read_dicom(content))
            else:
                slices.append(_read_image(content))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot read {plane} file '{upload.filename}': {exc}",
            ) from exc

    if not slices:
        raise HTTPException(status_code=400, detail=f"No readable {plane} slices were uploaded.")

    return np.stack(slices).astype(np.float32)


def _sample_slices(volume: np.ndarray, target_slices: int) -> np.ndarray:
    if volume.shape[0] == target_slices:
        return volume

    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
    tensor = F.interpolate(
        tensor,
        size=(target_slices, volume.shape[1], volume.shape[2]),
        mode="trilinear",
        align_corners=False,
    )
    return tensor.squeeze(0).squeeze(0).cpu().numpy()


def _resize_volume(volume: np.ndarray, image_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(volume).unsqueeze(1).float()
    tensor = F.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return tensor.squeeze(1)


def _preprocess(volume: np.ndarray) -> torch.Tensor:
    if volume.ndim != 3:
        raise HTTPException(status_code=400, detail=f"Expected volume shape [slices,height,width], got {volume.shape}.")

    volume = _sample_slices(volume, TARGET_SLICES)
    volume_min = float(np.min(volume))
    volume_max = float(np.max(volume))
    if volume_max > volume_min:
        volume = (volume - volume_min) / (volume_max - volume_min) * MAX_PIXEL_VAL
    else:
        volume = np.zeros_like(volume, dtype=np.float32)

    volume = (volume - MEAN) / STDDEV
    tensor = _resize_volume(volume.astype(np.float32), IMAGE_SIZE)
    tensor = torch.stack((tensor, tensor, tensor), dim=1)
    return tensor.unsqueeze(0)


@app.get("/health")
def health():
    return service.status()


@app.post("/predict")
async def predict(
    axial: List[UploadFile] = File(...),
    coronal: List[UploadFile] = File(...),
    sagittal: List[UploadFile] = File(...),
    threshold: float = Form(0.5),
    modelName: str = Form(DEFAULT_MODEL),
):
    if threshold <= 0 or threshold >= 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1.")

    uploaded = {
        "axial": await _files_to_volume(axial, "axial"),
        "coronal": await _files_to_volume(coronal, "coronal"),
        "sagittal": await _files_to_volume(sagittal, "sagittal"),
    }
    volumes = {plane: _preprocess(volume) for plane, volume in uploaded.items()}
    return service.predict(volumes, threshold, modelName)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
