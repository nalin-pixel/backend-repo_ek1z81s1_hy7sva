"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List

# Lead schema for persisted leads
class Lead(BaseModel):
    niche: str = Field(..., description="Niche or industry, e.g., 'plumbers'")
    area: str = Field(..., description="Area or city searched, e.g., 'Austin, TX'")
    name: str = Field(..., description="Business or contact name")
    email: Optional[str] = Field(None, description="Primary contact email")
    phone: Optional[str] = Field(None, description="Phone number")
    website: Optional[str] = Field(None, description="Website URL")
    address: Optional[str] = Field(None, description="Formatted address if available")
    source: Optional[str] = Field(None, description="Where the lead came from (osm/website)")
    osm_id: Optional[str] = Field(None, description="OpenStreetMap element id if applicable")

class ServedHash(BaseModel):
    niche: str
    area: str
    unique_key: str = Field(..., description="Deterministic hash based on name/website/osm_id to avoid duplicates")

# Example schemas (kept for reference; not used by the app)
class User(BaseModel):
    name: str
    email: str
    address: str
    age: Optional[int] = None
    is_active: bool = True

class Product(BaseModel):
    title: str
    description: Optional[str] = None
    price: float
    category: str
    in_stock: bool = True
