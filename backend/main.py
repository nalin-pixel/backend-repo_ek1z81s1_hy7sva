from __future__ import annotations

import csv
import hashlib
import io
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from database import db, create_document, get_documents
from schemas import Lead, ServedHash

app = FastAPI(title="Lead Generator API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    niche: str
    area: str
    limit: int = 20


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


async def nominatim_geocode(area: str) -> Optional[Dict[str, Any]]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": area, "format": "json", "limit": 1, "addressdetails": 0}
    headers = {"User-Agent": "LeadGenerator/1.0 (contact@example.com)"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params, headers=headers)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        item = data[0]
        # bbox: [south, north, west, east]
        bbox = item.get("boundingbox")
        if not bbox:
            return None
        south, north, west, east = map(float, bbox)
        return {"south": south, "north": north, "west": west, "east": east}


OSM_TAGS = [
    "name",
    "addr:full",
    "addr:street",
    "addr:housenumber",
    "addr:postcode",
    "addr:city",
    "contact:phone",
    "phone",
    "contact:website",
    "website",
    "contact:email",
    "email",
]


def build_overpass_query(niche: str, bbox: Dict[str, float], limit: int) -> str:
    # search in nodes/ways/relations with common tags matching niche as substring
    # For performance, use case-insensitive regex matching in name and amenity/craft/shop tags
    south, north, west, east = bbox["south"], bbox["north"], bbox["west"], bbox["east"]
    niche_re = niche
    query = f"""
    [out:json][timeout:25];
    (
      node["name"~"{niche_re}",i]({south},{west},{north},{east});
      way["name"~"{niche_re}",i]({south},{west},{north},{east});
      relation["name"~"{niche_re}",i]({south},{west},{north},{east});
      node["amenity"~"{niche_re}",i]({south},{west},{north},{east});
      way["amenity"~"{niche_re}",i]({south},{west},{north},{east});
      relation["amenity"~"{niche_re}",i]({south},{west},{north},{east});
      node["shop"~"{niche_re}",i]({south},{west},{north},{east});
      way["shop"~"{niche_re}",i]({south},{west},{north},{east});
      relation["shop"~"{niche_re}",i]({south},{west},{north},{east});
      node["craft"~"{niche_re}",i]({south},{west},{north},{east});
      way["craft"~"{niche_re}",i]({south},{west},{north},{east});
      relation["craft"~"{niche_re}",i]({south},{west},{north},{east});
    );
    out body {limit};
    >;
    out skel qt;
    """
    return query


async def overpass_search(niche: str, bbox: Dict[str, float], limit: int) -> List[Dict[str, Any]]:
    query = build_overpass_query(niche, bbox, limit)
    headers = {"User-Agent": "LeadGenerator/1.0 (contact@example.com)", "Content-Type": "text/plain"}
    async with httpx.AsyncClient(timeout=40) as client:
        r = await client.post("https://overpass-api.de/api/interpreter", content=query, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data.get("elements", [])


EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)


async def fetch_email_from_site(url: str) -> Optional[str]:
    try:
        headers = {"User-Agent": "LeadGenerator/1.0 (contact@example.com)"}
        async with httpx.AsyncClient(timeout=10, follow_redirects=True, headers=headers) as client:
            # try homepage first
            resp = await client.get(url)
            if resp.status_code == 200:
                m = EMAIL_REGEX.search(resp.text)
                if m:
                    return m.group(0)
            # try common contact pages
            for path in ["/contact", "/contact-us", "/about", "/impressum"]:
                try:
                    resp = await client.get(url.rstrip("/") + path)
                    if resp.status_code == 200:
                        m = EMAIL_REGEX.search(resp.text)
                        if m:
                            return m.group(0)
                except Exception:
                    continue
    except Exception:
        return None
    return None


def normalize_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    u = url.strip()
    if not u:
        return None
    if not u.startswith("http://") and not u.startswith("https://"):
        u = "https://" + u
    return u


async def build_leads(niche: str, area: str, elements: List[Dict[str, Any]], limit: int) -> List[Lead]:
    leads: List[Lead] = []
    count = 0
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        address_parts = [
            tags.get("addr:full"),
            " ".join(filter(None, [tags.get("addr:housenumber"), tags.get("addr:street")])),
            " ".join(filter(None, [tags.get("addr:postcode"), tags.get("addr:city")])),
        ]
        address = ", ".join(filter(None, [p.strip() for p in address_parts if p])) or None
        phone = tags.get("contact:phone") or tags.get("phone")
        website = normalize_url(tags.get("contact:website") or tags.get("website"))
        email = tags.get("contact:email") or tags.get("email")

        if not email and website:
            email = await fetch_email_from_site(website)

        osm_id = str(el.get("id"))
        lead = Lead(
            niche=niche,
            area=area,
            name=name,
            address=address,
            phone=phone,
            website=website,
            email=email,
            osm_id=osm_id,
            source="openstreetmap",
            created_at=datetime.utcnow().isoformat(),
        )
        leads.append(lead)
        count += 1
        if count >= limit:
            break
    return leads


async def is_served(niche: str, area: str, h: str) -> bool:
    coll = db["servedhash"]
    doc = coll.find_one({"niche": niche, "area": area, "hash": h})
    return doc is not None


async def mark_served(niche: str, area: str, h: str) -> None:
    coll = db["servedhash"]
    coll.insert_one({"niche": niche, "area": area, "hash": h, "created_at": datetime.utcnow().isoformat()})


@app.post("/api/leads/search")
async def search_leads(req: SearchRequest):
    bbox = await nominatim_geocode(req.area)
    if not bbox:
        raise HTTPException(status_code=404, detail="Area not found")

    elements = await overpass_search(req.niche, bbox, limit=max(req.limit * 3, 50))

    # Deduplicate by hash and avoid previously served
    fresh_elements: List[Dict[str, Any]] = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        website = normalize_url(tags.get("contact:website") or tags.get("website")) or ""
        osm_id = str(el.get("id"))
        h = sha256("|".join([name or "", website, osm_id]))
        if await is_served(req.niche, req.area, h):
            continue
        fresh_elements.append(el)

    leads = await build_leads(req.niche, req.area, fresh_elements, req.limit)

    # Persist leads and mark served hashes
    out: List[Dict[str, Any]] = []
    for lead in leads:
        h = sha256("|".join([lead.name or "", lead.website or "", lead.osm_id or ""]))
        await mark_served(req.niche, req.area, h)
        create_document("lead", lead.model_dump())
        out.append(lead.model_dump())

    return JSONResponse(content=out)


@app.get("/api/leads/export")
async def export_leads(niche: str = Query(...), area: str = Query(...)):
    # Export the most recent leads for this niche+area
    coll = db["lead"]
    cursor = coll.find({"niche": niche, "area": area}).sort("_id", -1).limit(1000)
    rows = list(cursor)

    def generate():
        output = io.StringIO()
        writer = csv.writer(output)
        header = ["niche", "area", "name", "email", "phone", "website", "address", "source"]
        writer.writerow(header)
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)
        for r in rows:
            writer.writerow([
                r.get("niche", ""),
                r.get("area", ""),
                r.get("name", ""),
                r.get("email", ""),
                r.get("phone", ""),
                r.get("website", ""),
                r.get("address", ""),
                r.get("source", "openstreetmap"),
            ])
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    filename = f"leads_{niche}_{area}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    headers = {
        "Content-Disposition": f"attachment; filename=\"{filename}\"",
        "Content-Type": "text/csv",
    }
    return StreamingResponse(generate(), headers=headers, media_type="text/csv")


@app.get("/api/leads/history")
async def leads_history(niche: Optional[str] = None, area: Optional[str] = None, limit: int = 100):
    filter_dict: Dict[str, Any] = {}
    if niche:
        filter_dict["niche"] = niche
    if area:
        filter_dict["area"] = area
    docs = get_documents("lead", filter_dict=filter_dict, limit=limit)
    return JSONResponse(content=docs)


@app.get("/test")
async def test():
    # quick DB test
    try:
        db.list_collection_names()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
