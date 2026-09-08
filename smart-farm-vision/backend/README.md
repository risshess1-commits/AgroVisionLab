# Smart Farm Vision API

Backend for the smart urban farm computer-vision platform.

## Run

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Models

Place YOLO weights in `backend/models`.

- `customized_teplitza_cucumber.pt` - cucumber fruits and flowers.
- `universal_cucumber.pt` - comparative cucumber model.
- `strawberry_best2.pt` - strawberry berries.
- `lettuce_health.pt`, `cucumber_leaf.pt` are optional production weights. Until they are added, the API uses deterministic demo CV heuristics so the platform remains demonstrable. `tomato_best.pt` is connected in the current build.
