import base64
import io
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover - requirements install this in deployment
    hf_hub_download = None


CLASS_LABELS = [
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "AMD",
    "Hypertensive Retinopathy",
]
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
MODEL_PATH = os.getenv("MODEL_PATH")
MODEL_URL = os.getenv("MODEL_URL")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "best_efficientnet_b0.pth")
MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def build_model() -> torch.nn.Module:
    model = EfficientNet.from_name("efficientnet-b0")
    model._fc = torch.nn.Sequential(
        torch.nn.Linear(model._fc.in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, len(CLASS_LABELS)),
    )
    return model


def resolve_model_path() -> Path:
    if MODEL_PATH:
        path = Path(MODEL_PATH)
        if not path.is_file():
            raise RuntimeError(f"MODEL_PATH does not point to a file: {path}")
        return path

    if HF_MODEL_REPO:
        if hf_hub_download is None:
            raise RuntimeError("huggingface-hub is required for HF_MODEL_REPO")
        return Path(
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=HF_MODEL_FILENAME,
                token=os.getenv("HF_TOKEN"),
            )
        )

    if MODEL_URL:
        import urllib.request

        cache_dir = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/retinal-model"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / HF_MODEL_FILENAME
        if not target.exists():
            urllib.request.urlretrieve(MODEL_URL, target)
        return target

    local_path = Path(__file__).resolve().parent / "models" / HF_MODEL_FILENAME
    if local_path.is_file():
        return local_path
    raise RuntimeError(
        "No model configured. Set HF_MODEL_REPO, MODEL_URL, or MODEL_PATH."
    )


def load_model() -> torch.nn.Module:
    model_path = resolve_model_path()
    model = build_model()
    state = torch.load(model_path, map_location=MODEL_DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(MODEL_DEVICE)
    model.eval()
    return model


try:
    model = load_model()
    MODEL_LOAD_ERROR: Optional[str] = None
except Exception as error:
    model = None
    MODEL_LOAD_ERROR = str(error)

app = FastAPI(title="Retinal Disease Screening API", version="1.0.0")
allowed_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def make_gradcam(image_tensor: torch.Tensor, class_index: int) -> Optional[str]:
    if model is None or not hasattr(model, "_blocks"):
        return None

    activation = None
    gradient = None
    target_layer = model._blocks[-1]

    def save_activation(_, __, output):
        nonlocal activation
        activation = output

    def save_gradient(_, __, grad_output):
        nonlocal gradient
        gradient = grad_output[0]

    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_full_backward_hook(save_gradient)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        logits[0, class_index].backward()
        if activation is None or gradient is None:
            return None
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activation).sum(dim=1)).squeeze(0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap = transforms.Resize((224, 224))(heatmap.unsqueeze(0)).squeeze(0)
        source = image_tensor.detach().cpu().squeeze(0)
        source = source * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        source = source + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        source = source.clamp(0, 1).permute(1, 2, 0).numpy()
        heat = heatmap.detach().cpu().numpy()
        overlay = source * 0.55
        overlay[:, :, 0] += heat * 0.45
        overlay[:, :, 1] += heat * 0.08
        overlay = (overlay.clip(0, 1) * 255).astype(np.uint8)
        output = io.BytesIO()
        Image.fromarray(overlay).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")
    finally:
        forward_handle.remove()
        backward_handle.remove()


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_error": MODEL_LOAD_ERROR,
        "classes": CLASS_LABELS,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, object]:
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {MODEL_LOAD_ERROR}")
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Only JPG, JPEG, and PNG images are supported.")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than the 12 MB upload limit.")
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    if image.width < 224 or image.height < 224:
        raise HTTPException(status_code=400, detail="Image must be at least 224 x 224 pixels.")

    image_tensor = preprocess(image).unsqueeze(0).to(MODEL_DEVICE)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(image_tensor))[0].cpu().tolist()
    probability_map = {
        label: round(float(probability), 6)
        for label, probability in zip(CLASS_LABELS, probabilities)
    }
    top_index = int(np.argmax(probabilities))
    gradcam = make_gradcam(image_tensor, top_index)
    return {
        "predicted_class": CLASS_LABELS[top_index],
        "confidence": round(float(probabilities[top_index]), 6),
        "probabilities": probability_map,
        "positive_classes": [
            label for label, probability in probability_map.items() if probability >= 0.5
        ],
        "gradcam_overlay": gradcam,
    }

