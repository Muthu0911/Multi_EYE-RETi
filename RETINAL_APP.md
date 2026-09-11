# RetinaLens retinal screening prototype

## What was reusable from the upstream project

The supplied `dbdemet/Multi-Ocular-Disease-Detection` repository includes a
real PyTorch checkpoint at `outputs/best_efficientnet_b0.pth` (19 MB in the
GitHub repository), not a 2 GB artifact. It also provides:

- EfficientNet-B0 with a `512 -> 5` classifier head and sigmoid multilabel inference.
- Class order: Diabetic Retinopathy, Glaucoma, Cataract, AMD, Hypertensive Retinopathy.
- 224 x 224 resize and ImageNet normalization
  (`mean=[.485,.456,.406]`, `std=[.229,.224,.225]`).
- ODIR-5K as the stated dataset and the original training code.
- A Gradio demo, but no FastAPI backend, React frontend, upload validation, or
  Grad-CAM implementation.

The existing farm application in this workspace is left intact. The new
application lives in `retinal_backend/` and `retinal_frontend/`.

## Local run

1. Put the upstream checkpoint at
   `retinal_backend/models/best_efficientnet_b0.pth`, or configure `MODEL_PATH`.
2. Install Python packages from `retinal_backend/requirements.txt`.
3. Start the API from `retinal_backend/`:
   `uvicorn main:app --reload --port 8000`
4. Start the UI from `retinal_frontend/`:
   `npm install && npm run dev`

The API loads the checkpoint once at process startup. It never writes uploaded
images to disk. `/predict` returns the primary class, all five probabilities,
positive classes at the 0.5 threshold, and a real Grad-CAM overlay when the
EfficientNet hooks are available.

## Render deployment

`render.yaml` defines one Python web service and one static site. Upload the
checkpoint to a Hugging Face model repository and set these Render variables:

The backend includes `retinal_backend/runtime.txt` to pin Render to Python
3.12.8. This is required because the upstream checkpoint integration uses
PyTorch 2.5.1, whose published wheels do not support Python 3.14.

- API: `HF_MODEL_REPO`, optional `HF_MODEL_FILENAME`, `FRONTEND_ORIGINS`
- Static site: `VITE_API_URL` set to the API service URL

Use `HF_TOKEN` only when the Hugging Face repository is private. `MODEL_URL`
and `MODEL_PATH` are supported for non-Hugging-Face hosting and local runs.
No model weights are committed to this workspace.
