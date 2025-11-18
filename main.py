import os
import io
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from database import db

app = FastAPI(title="Data Intelligence Platform", version="0.1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- Schemas --------------------
class IngestResponse(BaseModel):
    dataset_id: str
    rows: int
    columns: List[str]
    created_at: datetime

class DatasetMeta(BaseModel):
    name: str
    source: str
    columns: List[str]
    row_count: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

class StatSummary(BaseModel):
    column: str
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]

class PredictiveFeatureConfig(BaseModel):
    target: str
    features: Optional[List[str]] = None


# -------------------- Utilities --------------------

def df_from_file(upload: UploadFile) -> pd.DataFrame:
    try:
        content = upload.file.read()
        if upload.filename.endswith(".csv"):
            return pd.read_csv(io.BytesIO(content))
        if upload.filename.endswith(".xlsx") or upload.filename.endswith(".xls"):
            return pd.read_excel(io.BytesIO(content))
        raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")


# -------------------- Routes: Health --------------------
@app.get("/")
def root():
    return {"service": "backend", "status": "ok"}

@app.get("/test")
def test_database():
    resp = {
        "backend": "running",
        "db": "connected" if db else "not-configured",
        "collections": []
    }
    try:
        if db:
            resp["collections"] = db.list_collection_names()
    except Exception as e:
        resp["db"] = f"error: {e}"
    return resp


# -------------------- Routes: Ingestion --------------------
@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    df = df_from_file(file)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how="all", inplace=True)

    dataset_id = str(uuid.uuid4())
    doc = {
        "_id": dataset_id,
        "name": name or file.filename,
        "columns": df.columns.tolist(),
        "row_count": int(len(df)),
        "created_at": datetime.utcnow(),
    }
    if db:
        db["dataset"].insert_one(doc)
        # Store raw data in a separate collection, chunked if large
        records = df.to_dict(orient="records")
        if records:
            db["dataset_row"].insert_many([{**r, "dataset_id": dataset_id} for r in records])

    return IngestResponse(
        dataset_id=dataset_id,
        rows=len(df),
        columns=df.columns.tolist(),
        created_at=doc["created_at"],
    )


@app.post("/ingest/api")
async def ingest_api(url: str = Form(...), name: Optional[str] = Form(None)):
    import requests
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.json_normalize(data)
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch: {e}")

    dataset_id = str(uuid.uuid4())
    doc = {
        "_id": dataset_id,
        "name": name or url,
        "columns": df.columns.tolist(),
        "row_count": int(len(df)),
        "created_at": datetime.utcnow(),
    }
    if db:
        db["dataset"].insert_one(doc)
        records = df.to_dict(orient="records")
        if records:
            db["dataset_row"].insert_many([{**r, "dataset_id": dataset_id} for r in records])

    return {"dataset_id": dataset_id, "rows": len(df), "columns": df.columns.tolist()}


# -------------------- Routes: Analysis --------------------
@app.get("/analysis/summary/{dataset_id}")
async def summary(dataset_id: str):
    if not db:
        raise HTTPException(500, "Database not configured")
    rows = list(db["dataset_row"].find({"dataset_id": dataset_id}))
    if not rows:
        raise HTTPException(404, "Dataset not found")
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["_id", "dataset_id"]} for r in rows])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summaries: List[Dict[str, Any]] = []
    for col in numeric_cols:
        s = df[col].dropna()
        summaries.append({
            "column": col,
            "mean": float(s.mean()) if not s.empty else None,
            "median": float(s.median()) if not s.empty else None,
            "std": float(s.std()) if not s.empty else None,
            "min": float(s.min()) if not s.empty else None,
            "max": float(s.max()) if not s.empty else None,
        })

    # Correlation
    corr = None
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True).fillna(0).to_dict()

    # Anomaly detection using Isolation Forest for numeric features
    anomalies = {}
    try:
        if len(numeric_cols) > 0 and len(df) > 10:
            iso = IsolationForest(contamination=0.05, random_state=42)
            scores = iso.fit_predict(df[numeric_cols].fillna(0))
            anomalies = {
                "count": int((scores == -1).sum()),
                "indices": [int(i) for i, v in enumerate(scores) if v == -1]
            }
    except Exception as e:
        anomalies = {"error": str(e)}

    return {"summaries": summaries, "correlation": corr, "anomalies": anomalies}


# -------------------- SSE (basic) --------------------
@app.get("/events")
async def events():
    async def event_stream():
        yield "data: {\"type\": \"heartbeat\", \"time\": \"%s\"}\n\n" % datetime.utcnow().isoformat()
    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
