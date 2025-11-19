from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field, HttpUrl


class Lead(BaseModel):
    id: Optional[str] = Field(default=None, description="MongoDB ObjectId as string")
    niche: str
    area: str
    name: str
    address: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    email: Optional[str] = None
    source: str = "openstreetmap"
    osm_id: Optional[str] = None
    created_at: Optional[str] = None


class ServedHash(BaseModel):
    id: Optional[str] = Field(default=None)
    niche: str
    area: str
    hash: str
    created_at: Optional[str] = None
