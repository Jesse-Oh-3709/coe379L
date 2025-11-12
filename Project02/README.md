# Project 02 – Inference Server

### Endpoints
- **GET /summary** → model metadata (JSON)
- **POST /inference** → binary image → `{ "prediction": "damage" | "no_damage" }`

### Run locally
```bash
pip install -r requirements.txt
python server.py
