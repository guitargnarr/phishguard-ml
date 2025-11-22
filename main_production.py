#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhishGuard ML - Production FastAPI Server with Dual-Mode Support
-------------------------------------------------------------------
Supports both simple (fast) and ensemble (accurate) prediction modes

Run:  ./run_production.sh
API:  http://localhost:8000
Docs: http://localhost:8000/docs
"""
import os
import csv
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import unified detector
from phishing_detector import PhishingDetector

# ------------------------------------------------------------------ #
# 1. Load Both Models at Startup
# ------------------------------------------------------------------ #
print("🚀 Loading PhishGuard ML models...")
print("-" * 50)

# Load simple model (fast, default)
simple_detector = PhishingDetector(mode='simple')
print(f"✅ Simple model loaded: {simple_detector.model_info}")

# Load ensemble model (accurate, optional)
try:
    ensemble_detector = PhishingDetector(mode='ensemble')
    print(f"✅ Ensemble model loaded: {ensemble_detector.model_info['models']}")
    ensemble_available = True
except Exception as e:
    print(f"⚠️  Ensemble model not available: {e}")
    ensemble_available = False

print("-" * 50)

# ------------------------------------------------------------------ #
# 2. FastAPI App
# ------------------------------------------------------------------ #
app = FastAPI(
    title="PhishGuard ML - Production API",
    version="2.0",
    description="Dual-mode phishing detection: Fast (simple) or Accurate (ensemble)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
# 3. Request / Response Models
# ------------------------------------------------------------------ #


class ClassificationRequest(BaseModel):
    email_text: str


class ClassificationResponse(BaseModel):
    classification: str
    confidence: float
    is_phishing: bool
    model_mode: str


class DetailedClassificationResponse(BaseModel):
    classification: str
    confidence: float
    is_phishing: bool
    model_mode: str
    probabilities: dict
    model_info: dict
    individual_votes: Optional[dict] = None
    agreement_rate: Optional[float] = None


# ------------------------------------------------------------------ #
# 4. Logging System
# ------------------------------------------------------------------ #
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SECURITY_LOG = LOG_DIR / "security_log.csv"


def init_security_log():
    """Initialize the security log with headers if it doesn't exist."""
    if not SECURITY_LOG.exists():
        with SECURITY_LOG.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "mode", "classification",
                "confidence", "text_preview"
            ])


def log_classification(mode: str, classification: str, confidence: float, text: str):
    """Log classification event."""
    with SECURITY_LOG.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            mode,
            classification,
            f"{confidence:.3f}",
            text[:100]  # Privacy: only first 100 chars
        ])


# Initialize log
init_security_log()

# ------------------------------------------------------------------ #
# 5. Classification Endpoints
# ------------------------------------------------------------------ #


@app.post("/classify", response_model=ClassificationResponse)
async def classify_email(
    req: ClassificationRequest,
    mode: str = Query('simple', pattern="^(simple|ensemble)$")
):
    """
    Classify an email as phishing or legitimate.

    Args:
        req: Email text to classify
        mode: 'simple' (fast, default) or 'ensemble' (accurate)

    Returns:
        Classification with confidence score
    """
    if not req.email_text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")

    # Select detector based on mode
    if mode == 'ensemble':
        if not ensemble_available:
            raise HTTPException(
                status_code=503,
                detail="Ensemble mode not available. Ensemble models not loaded."
            )
        detector = ensemble_detector
    else:
        detector = simple_detector

    # Classify
    try:
        classification, confidence, is_phishing = detector.predict(req.email_text)

        # Log
        log_classification(mode, classification, confidence, req.email_text)

        return ClassificationResponse(
            classification=classification,
            confidence=confidence,
            is_phishing=is_phishing,
            model_mode=mode
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/classify_detailed", response_model=DetailedClassificationResponse)
async def classify_detailed(
    req: ClassificationRequest,
    mode: str = Query('ensemble', pattern="^(simple|ensemble)$")
):
    """
    Get detailed classification with model votes and confidence breakdown.

    Best used with ensemble mode for individual model insights.

    Args:
        req: Email text to classify
        mode: 'simple' or 'ensemble' (default: ensemble)

    Returns:
        Detailed classification with probabilities and model votes
    """
    if not req.email_text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")

    # Select detector
    if mode == 'ensemble':
        if not ensemble_available:
            raise HTTPException(
                status_code=503,
                detail="Ensemble mode not available."
            )
        detector = ensemble_detector
    else:
        detector = simple_detector

    # Get detailed prediction
    try:
        result = detector.predict_with_details(req.email_text)

        # Log
        log_classification(mode, result['classification'], result['confidence'], req.email_text)

        return DetailedClassificationResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


# ------------------------------------------------------------------ #
# 6. Health & Stats Endpoints
# ------------------------------------------------------------------ #
@app.get("/health")
async def health_check():
    """Check if the API is healthy and which models are loaded."""
    return {
        "status": "healthy",
        "simple_model_loaded": simple_detector.loaded,
        "ensemble_model_loaded": ensemble_available,
        "models_available": {
            "simple": simple_detector.model_info,
            "ensemble": ensemble_detector.model_info if ensemble_available else None
        },
        "endpoints": [
            "/classify (mode=simple|ensemble)",
            "/classify_detailed (mode=simple|ensemble)",
            "/stats",
            "/models"
        ]
    }


@app.get("/stats")
async def get_stats():
    """Get statistics about logged classifications."""
    if not SECURITY_LOG.exists():
        return {"total_events": 0}

    with SECURITY_LOG.open("r") as f:
        reader = csv.DictReader(f)
        events = list(reader)

    # Count by mode
    simple_count = sum(1 for e in events if e.get("mode") == "simple")
    ensemble_count = sum(1 for e in events if e.get("mode") == "ensemble")

    # Count by classification
    phishing_count = sum(1 for e in events if e.get("classification") == "phishing")
    legitimate_count = sum(1 for e in events if e.get("classification") == "legitimate")

    return {
        "total_events": len(events),
        "by_mode": {
            "simple": simple_count,
            "ensemble": ensemble_count
        },
        "by_classification": {
            "phishing": phishing_count,
            "legitimate": legitimate_count
        },
        "log_file": str(SECURITY_LOG)
    }


@app.get("/models")
async def get_models_info():
    """Get information about available models."""
    return {
        "simple": simple_detector.get_model_info(),
        "ensemble": ensemble_detector.get_model_info() if ensemble_available else {"loaded": False}
    }


# ------------------------------------------------------------------ #
# 7. CLI Entrypoint
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("🛡️  PhishGuard ML - Production Server v2.0")
    print("=" * 70)
    print("\n📦 Available Models:")
    print(f"  ⚡ Simple:   {simple_detector.model_info['type']} ({simple_detector.model_info['features']} features)")
    if ensemble_available:
        print(f"  🎯 Ensemble: {len(ensemble_detector.model_info['models'])} models ({ensemble_detector.model_info['features']} features)")
    else:
        print("  ⚠️  Ensemble: Not available")

    print("\n🌐 Endpoints:")
    print("  POST /classify?mode=simple|ensemble")
    print("  POST /classify_detailed?mode=simple|ensemble")
    print("  GET  /health")
    print("  GET  /stats")
    print("  GET  /models")
    print("  GET  /docs (Swagger UI)")

    print("\n📡 Starting server on http://localhost:8000")
    print("   Press Ctrl+C to stop")
    print("=" * 70 + "\n")

    uvicorn.run("main_production:app", host="0.0.0.0", port=8000, reload=True)
