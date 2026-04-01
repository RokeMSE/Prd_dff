# Stain Detective — Setup Guide
A desktop app for tracing manufacturing defects back to their origin step using Vision Language Models (VLM).

---

## Requirements
- Python 3.10+
- Windows (PySide6 desktop UI)
- Access to a supported VLM API (Azure OpenAI)

---

## Installation
```bash
# From the Product Diff/ directory
python -m venv venv
venv\Scripts\activate
pip install -r src/requirements.txt

## Environment Setup
Create a `.env` file at `src/.env` (one already exists — update it with your credentials).

```env example
VLM_PROVIDER=azure
OPENAI_API_KEY=<your_azure_api_key>
OPENAI_ENDPOINT=<your_azure_endpoint>
OPENAI_MODEL=gpt-5.4
OPENAI_API_VERSION=2025-01-01-preview
```

### Proxy
```env
PROXY_HTTP=http://proxy-chain.intel.com:911
PROXY_HTTPS=http://proxy-chain.intel.com:912
```

---
## Run the App

```bash
python src/Frontend/UI/main.py
```

---
## App Usage
1. **Search** — Enter a Visual ID to load historical process images.
2. **Select mode**:
   - **Auto** — reads defect bounding boxes from `DVI_box_data.csv`
   - **Manual** — draw defect boxes on the OG image
3. **Run traceback** — the app aligns process images and queries the VLM at each step.
4. **View results** — traceback panels and report are shown; results can be saved as a ZIP.
