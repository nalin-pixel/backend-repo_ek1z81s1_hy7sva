import os
import re
import time
import hashlib
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from database import db

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OSM_NOMINATIM = "https://nominatim.openstreetmap.org/search"
OSM_OVERPASS = "https://overpass-api.de/api/interpreter"
UA = {
    "User-Agent": "Flames-LeadGen/1.1 (+https://flames.blue; contact support@flames.blue)",
}


class LeadRequest(BaseModel):
    niche: str = Field(..., description="Niche or industry, e.g., plumbers")
    area: str = Field(..., description="City/region/country to search, e.g., Austin, TX")
    limit: int = Field(10, ge=1, le=100, description="How many leads to collect")
    require_email: bool = Field(False, description="Only include leads with an email")
    require_website: bool = Field(False, description="Only include leads with a website")
    require_phone: bool = Field(False, description="Only include leads with a phone number")
    global_dedupe: bool = Field(False, description="Avoid repeats across all searches (not just niche+area)")


class Lead(BaseModel):
    niche: str
    area: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    source: Optional[str] = None
    osm_id: Optional[str] = None


@app.get("/")
def read_root():
    return {"message": "Lead Generator Backend Running"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Connected & Working"
            response["database_url"] = "✅ Set"
            response["database_name"] = db.name
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()[:10]
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


# Utility functions

def _hash_key(name: str, website: Optional[str], osm_id: Optional[str]) -> str:
    base = (name or "").strip().lower() + "|" + (website or "").strip().lower() + "|" + (osm_id or "").strip().lower()
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _normalize_url(url: str) -> Optional[str]:
    if not url:
        return None
    u = url.strip()
    if u.startswith("mailto:"):
        return None
    if u.startswith("//"):
        u = "https:" + u
    if not u.startswith("http"):
        u = "https://" + u
    return u


EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)


def _extract_emails_from_html(html: str) -> List[str]:
    if not html:
        return []
    found = set(re.findall(EMAIL_REGEX, html))
    return list(found)


def _try_fetch(url: str, timeout: int = 8) -> Optional[str]:
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code == 200 and r.text:
            return r.text
    except Exception:
        return None
    return None


CONTACT_PATHS = [
    "/contact",
    "/contact/",
    "/contact-us",
    "/contact-us/",
    "/about",
    "/about/",
    "/about-us",
    "/about-us/",
    "/impressum",
    "/impressum/",
    "/kontakt",
    "/kontakt/",
    "/contacto",
    "/contacto/",
    "/contacts",
    "/contacts/",
    "/empresa",
    "/empresa/",
]


def discover_email(website: Optional[str]) -> Optional[str]:
    site = _normalize_url(website) if website else None
    candidates = []
    if site:
        candidates.append(site)
        for path in CONTACT_PATHS:
            candidates.append(site.rstrip("/") + path)

    for u in candidates:
        html = _try_fetch(u)
        if not html:
            continue
        emails = _extract_emails_from_html(html)
        if emails:
            return emails[0]
        # be polite to servers
        time.sleep(0.3)
    return None


def _request_with_retries(method: str, url: str, *, params=None, data=None, timeout=15, retries=2, backoff=0.8):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            if method == "GET":
                r = requests.get(url, params=params, headers=UA, timeout=timeout)
            else:
                r = requests.post(url, data=data, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception as e:
            last_exc = e
        time.sleep(backoff * (attempt + 1))
    if last_exc:
        raise last_exc
    raise HTTPException(status_code=502, detail="External service error")


def geocode_area(area: str):
    params = {"q": area, "format": "json", "limit": 1}
    r = _request_with_retries("GET", OSM_NOMINATIM, params=params, timeout=12)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Geocoding service error")
    data = r.json()
    if not data:
        raise HTTPException(status_code=404, detail="Area not found")
    bbox = data[0]["boundingbox"]  # [south, north, west, east]
    south, north, west, east = bbox
    return float(south), float(north), float(west), float(east)


def overpass_search(niche: str, bbox, max_results: int) -> List[dict]:
    south, north, west, east = bbox
    niche_regex = niche.replace("\"", " ")
    query = f"""
    [out:json][timeout:25];
    (
      node["name"~"{niche_regex}",i]({south},{west},{north},{east});
      node["shop"~"{niche_regex}",i]({south},{west},{north},{east});
      node["amenity"~"{niche_regex}",i]({south},{west},{north},{east});
      node["craft"~"{niche_regex}",i]({south},{west},{north},{east});
      way["name"~"{niche_regex}",i]({south},{west},{north},{east});
      relation["name"~"{niche_regex}",i]({south},{west},{north},{east});
    );
    out tags {max_results};
    """
    r = _request_with_retries("POST", OSM_OVERPASS, data=query.encode("utf-8"), timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Overpass service error")
    data = r.json()
    return data.get("elements", [])[: max_results * 3]


def build_leads(
    niche: str,
    area: str,
    limit: int,
    *,
    require_email: bool = False,
    require_website: bool = False,
    require_phone: bool = False,
    global_dedupe: bool = False,
    mark_served: bool = True,
) -> List[Lead]:
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    bbox = geocode_area(area)
    elements = overpass_search(niche, bbox, limit)

    served_coll = db["servedhash"]
    leads_coll = db["lead"]

    results: List[Lead] = []

    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        website = tags.get("website") or tags.get("contact:website")
        email = tags.get("email") or tags.get("contact:email")
        phone = tags.get("phone") or tags.get("contact:phone")

        address_parts = [
            tags.get("addr:housenumber"),
            tags.get("addr:street"),
            tags.get("addr:city"),
            tags.get("addr:state"),
            tags.get("addr:postcode"),
            tags.get("addr:country"),
        ]
        address = ", ".join([p for p in address_parts if p]) or None

        osm_id = str(el.get("id"))
        unique_key = _hash_key(name, website, osm_id)

        # Check served: scoped or global
        if global_dedupe:
            served_query = {"unique_key": unique_key, "global": True}
        else:
            served_query = {"niche": niche.lower(), "area": area.lower(), "unique_key": unique_key, "global": {"$ne": True}}
        if served_coll.find_one(served_query):
            continue

        if not email:
            email = discover_email(website)

        # Apply filters
        if require_email and not email:
            continue
        if require_website and not website:
            continue
        if require_phone and not phone:
            continue

        lead_doc = {
            "niche": niche,
            "area": area,
            "name": name,
            "email": email,
            "phone": phone,
            "website": _normalize_url(website) if website else None,
            "address": address,
            "source": "osm",
            "osm_id": osm_id,
        }

        # Persist lead (upsert on unique combination)
        try:
            leads_coll.update_one(
                {"osm_id": osm_id, "niche": niche, "area": area},
                {"$set": lead_doc, "$setOnInsert": {"created_at": requests.utils.datetime.datetime.utcnow()}},
                upsert=True,
            )
        except Exception:
            # Non-fatal: continue even if persistence fails
            pass

        if mark_served:
            served_doc = {"unique_key": unique_key}
            if global_dedupe:
                served_doc.update({"global": True})
                served_coll.update_one({"unique_key": unique_key, "global": True}, {"$set": served_doc}, upsert=True)
            else:
                served_doc.update({"niche": niche.lower(), "area": area.lower(), "global": False})
                served_coll.update_one(
                    {"niche": niche.lower(), "area": area.lower(), "unique_key": unique_key, "global": False},
                    {"$set": served_doc},
                    upsert=True,
                )

        results.append(Lead(**lead_doc))
        if len(results) >= limit:
            break

    return results


@app.post("/api/leads/search", response_model=List[Lead])
def search_leads(req: LeadRequest):
    try:
        return build_leads(
            req.niche.strip(),
            req.area.strip(),
            req.limit,
            require_email=req.require_email,
            require_website=req.require_website,
            require_phone=req.require_phone,
            global_dedupe=req.global_dedupe,
            mark_served=True,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


@app.get("/api/leads/export", response_class=PlainTextResponse)
def export_leads(
    niche: str = Query(...),
    area: str = Query(...),
    limit: int = Query(10, ge=1, le=100),
    require_email: bool = Query(False),
    require_website: bool = Query(False),
    require_phone: bool = Query(False),
    global_dedupe: bool = Query(False),
):
    leads = build_leads(
        niche.strip(),
        area.strip(),
        limit,
        require_email=require_email,
        require_website=require_website,
        require_phone=require_phone,
        global_dedupe=global_dedupe,
        mark_served=True,
    )
    # Build CSV
    headers = ["niche", "area", "name", "email", "phone", "website", "address", "source", "osm_id"]
    lines = [",".join(headers)]

    def esc(val: Optional[str]) -> str:
        if val is None:
            return ""
        s = str(val).replace('"', '""')
        if "," in s or "\n" in s:
            return f'"{s}"'
        return s

    for lead in leads:
        row = [esc(getattr(lead, h)) for h in headers]
        lines.append(",".join(row))

    csv_text = "\n".join(lines)
    safe_niche = re.sub(r"[^A-Za-z0-9_-]+", "_", niche.strip())
    safe_area = re.sub(r"[^A-Za-z0-9_-]+", "_", area.strip())
    return PlainTextResponse(
        content=csv_text,
        headers={
            "Content-Disposition": f"attachment; filename=leads_{safe_niche}_{safe_area}.csv",
            "Content-Type": "text/csv; charset=utf-8",
        },
    )


@app.get("/api/leads/history", response_model=List[Lead])
def history(niche: Optional[str] = None, area: Optional[str] = None, limit: int = 50):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    query = {}
    if niche:
        query["niche"] = niche
    if area:
        query["area"] = area
    docs = list(db["lead"].find(query).sort("_id", -1).limit(limit))
    results: List[Lead] = []
    for d in docs:
        results.append(
            Lead(
                niche=d.get("niche"),
                area=d.get("area"),
                name=d.get("name"),
                email=d.get("email"),
                phone=d.get("phone"),
                website=d.get("website"),
                address=d.get("address"),
                source=d.get("source"),
                osm_id=d.get("osm_id"),
            )
        )
    return results


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
