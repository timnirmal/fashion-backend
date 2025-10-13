import os
import json
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .service import RecommenderService


class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    method: Optional[str] = Field(default="blend_bpr_tfidf", description="Recommender method")
    top_k: int = Field(default=20, ge=1, le=200, description="Number of items to return")


class RecommendResponse(BaseModel):
    user_id: str
    method: str
    items: List[str]


def create_app() -> FastAPI:
    app = FastAPI(title="Fashion Recsys API", version="1.0.0")

    # CORS for local dev and typical ports; adjust via env ALLOW_ORIGINS
    allow_origins = os.getenv("ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allow_origins if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    svc = RecommenderService()

    @app.get("/health")
    def health():
        return {"status": "ok", "ready": svc.is_ready()}

    @app.post("/train")
    def train():
        try:
            svc.load()
            return {"status": "ok", "users": len(svc.user_ids), "items": len(svc.item_ids)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/methods")
    def methods():
        if not svc.is_ready():
            # lazy load to be helpful
            try:
                svc.load()
            except Exception:
                pass
        return {"methods": svc.methods()}

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(body: RecommendRequest):
        if not svc.is_ready():
            try:
                svc.load()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Service not ready: {e}")

        items = svc.recommend(body.user_id, method=body.method, top_k=body.top_k)
        return RecommendResponse(user_id=body.user_id, method=(body.method or "blend_bpr_tfidf"), items=items)

    @app.get("/items/{item_id}")
    def get_item(item_id: str):
        if not svc.is_ready():
            try:
                svc.load()
            except Exception:
                pass
        data = svc.get_item(item_id)
        if not data:
            raise HTTPException(status_code=404, detail="Item not found")
        return data

    @app.get("/metrics")
    def metrics():
        # Try experiments/latest/metrics.json then fallback to experiments/metrics_latest.json
        latest_dir = os.path.join(svc.project_root, "experiments", "latest")
        candidates = [
            os.path.join(latest_dir, "metrics.json"),
            os.path.join(svc.project_root, "experiments", "metrics_latest.json"),
        ]
        for path in candidates:
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        return {"metrics": json.load(f)}
            except Exception:
                continue
        return {"metrics": {}}

    return app


app = create_app()


