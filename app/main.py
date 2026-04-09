"""
FastAPI server for VinFast Warranty AI Agent.

Serves the frontend, provides REST API for vehicle data,
chat endpoint for AI agent, and booking confirmation.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

from app import agent, data


# ─── Lifespan: init data & start TTL worker ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    data.init_data()
    data.start_ttl_worker()
    print("[OK] Data loaded. TTL worker started.")
    yield


app = FastAPI(
    title="VinFast Warranty AI Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────
class ChatRequest(BaseModel):
    messages: list[dict]
    selected_vehicle_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    tool_calls_log: list = []
    booking: dict | None = None


class ConfirmRequest(BaseModel):
    booking_id: str


# ─── API Endpoints ───────────────────────────────────────────────

@app.get("/api/vehicles")
def get_vehicles():
    """Get all vehicles belonging to the demo user."""
    vehicles = data.get_all_vehicles()
    # Return simplified list for the selector
    return [
        {
            "id": v["id"],
            "model": v["model"],
            "vin": v["vin"],
            "color": v["color"],
            "purchase_date": v["purchase_date"],
            "odo_km": v["telemetry"]["odo_km"],
            "battery_soh_percent": v["telemetry"]["battery_soh_percent"],
            "error_count": len(v["telemetry"].get("last_error_codes", [])),
        }
        for v in vehicles
    ]


@app.get("/api/vehicles/{vehicle_id}")
def get_vehicle(vehicle_id: str):
    """Get detailed vehicle info."""
    v = data.get_vehicle(vehicle_id)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return v


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """Send a message to the AI agent and get a response."""
    try:
        result = agent.chat(
            messages=req.messages,
            selected_vehicle_id=req.selected_vehicle_id,
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/booking/confirm")
def confirm_booking(req: ConfirmRequest):
    """Confirm a PENDING booking (PENDING -> CONFIRMED)."""
    booking = data.confirm_booking(req.booking_id)
    if not booking:
        raise HTTPException(
            status_code=400,
            detail="Không thể xác nhận lịch hẹn. Lịch hẹn đã hết hạn hoặc không tồn tại."
        )
    return booking


@app.get("/api/bookings")
def get_bookings(vehicle_id: str | None = None):
    """Get bookings of the demo user, optionally filtered by vehicle_id."""
    return data.get_user_bookings(user_id="U_VIN_001", vehicle_id=vehicle_id)


@app.get("/api/booking/{booking_id}")
def get_booking(booking_id: str):
    """Get booking details including TTL remaining."""
    booking = data.get_booking(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    ttl = data.get_booking_ttl_remaining(booking_id)
    return {**booking, "ttl_remaining_seconds": ttl}


# ─── Serve static frontend ───────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# Mount only if directory exists (avoid error on import)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Frontend not found. Place index.html in /static/"}
