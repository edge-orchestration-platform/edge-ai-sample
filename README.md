# edge-onnx-sample

## Run locally

1. Create a venv and install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r app/requirements.txt
````

2. Run the app:

```bash
python -m app.main
```

3. Test the API (example):

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"a": [1,2,3], "b": [4,5,6]}' \
  http://localhost:8080/infer
```

## Run tests

```bash
pip install pytest
pytest -q
```

## CI / Building images

Push to `main` (or run workflow manually) and the GitHub Actions workflow will build images for linux/amd64 and linux/arm64 and push to GHCR with tags `latest` and commit SHA.

```

---

## How to use this

1. Copy the files into your repository with the same structure.
2. Commit and push to GitHub.
3. On push to `main`, the Actions workflow will build and push the image to GHCR.
