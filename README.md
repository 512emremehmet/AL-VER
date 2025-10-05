# 🎪 Will It Rain On My Parade? — NASA Climate ML Dashboard

**Will It Rain On My Parade?** is a NASA-backed machine learning dashboard that forecasts precipitation and temperature extremes, with interactive maps, confidence-scored predictions, and historical trend analysis.

---

## Features

* Multi-model ensemble forecasts for temperature and precipitation
* Confidence scoring for predictions
* Interactive map visualization
* Historical extreme weather analysis
* API endpoints for integration

---

## Technologies

* **Backend:** Python, FastAPI, asyncio, aiohttp
* **ML:** Scikit-learn, ensemble models
* **Frontend:** Leaflet.js, HTML/CSS/JS
* **Data:** JSON, Streaming APIs

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/512emremehmet/AL-VER.git
cd AL-VER
```

2. Create a virtual environment and activate it:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Backend (FastAPI)

Start the backend server:

```bash
uvicorn main:app --reload --port 8000
```

* Open your browser at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to see interactive API documentation.
* Available endpoints:

  * `/forecast` → Get weather forecast with confidence scores
  * `/historical` → Retrieve historical climate data
  * `/health` → Check server status

### 2. Frontend (Leaflet.js Interactive Dashboard)

1. Open the `index.html` file in your browser.
2. Use the map interface to select locations and view predicted precipitation, temperature, and confidence metrics.

**Optional:** Run frontend via local server to avoid CORS issues:

```bash
python -m http.server 8080
```

Open [http://127.0.0.1:8080/index.html](http://127.0.0.1:8080/index.html) in your browser.

### 3. Backend & Frontend on Same Port

To serve frontend files directly from FastAPI:

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()
```

Place `index.html` and static assets in a folder named `static`.

---

### 4. Example API Call (Python)

```python
import requests

url = "http://127.0.0.1:8000/forecast"
params = {"lat": 40.7128, "lon": -74.0060}  # New York City example

response = requests.get(url, params=params)
data = response.json()
print(data)
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## License

MIT License
