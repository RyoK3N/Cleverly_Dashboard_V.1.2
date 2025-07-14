#!/usr/bin/env python3
"""
fetch_calendly_data.py
─────────────────────────────
Fetch scheduled events (active & canceled) for target event types,
then enrich each event with the first invitee’s name & email,
using retry‐backoff and concurrent fetching for speed.

Usage
=====
export CALENDLY_API_KEY="apt_…"
python fetch_calendly_data.py
"""

import os
import sys
import time
import datetime as dt
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE = "https://api.calendly.com/"
TARGET_EVENT_TYPES = [
    "https://api.calendly.com/event_types/3f3b8e40-246e-4723-8690-d0de0419e231",
    "https://api.calendly.com/event_types/6b4aa5e3-b4a2-4ef2-b1b2-1405b02e9806"
]
MAX_WORKERS = 5  # concurrency level for invitee fetching
THROTTLE_DELAY = 0.2  # seconds between retries on 429

# ─── LOAD API KEY ──────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("CALENDLY_API_KEY")
if not API_KEY:
    sys.exit("❌  CALENDLY_API_KEY not set in environment")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# ─── SESSION WITH RETRIES ──────────────────────────────────────────────────────
session = requests.Session()
retry_strategy = Retry(total=5,
                       backoff_factor=1,
                       status_forcelist=[429, 500, 502, 503, 504],
                       allowed_methods=["GET"],
                       raise_on_status=False)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def _get(url: str, params: dict | None = None, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            resp = session.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 429:
                retry_after = int(
                    resp.headers.get("Retry-After", THROTTLE_DELAY))
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                print(f"[Error] {_short(url)}: {e}")
                return {"collection": []}
            time.sleep(2**attempt)
    return {"collection": []}


def _paginate(url: str, params: dict) -> list[dict]:
    coll: list[dict] = []
    while url:
        page = _get(url, params)
        items = page.get("collection", [])
        coll.extend(items)
        url = page.get("pagination", {}).get("next_page")
        params = None
    return coll


def _short(url: str, length: int = 40) -> str:
    return url if len(url) <= length else url[:length - 3] + "..."


def current_org_uri() -> str:
    me = _get(urljoin(BASE, "users/me"))
    resource = me.get("resource", {})
    org = resource.get("current_organization")
    if not org:
        sys.exit("❌ Failed to retrieve current organization URI")
    return org


def list_scheduled_events(org_uri: str,
                          status: str,
                          min_start_time: str,
                          sort: str = "start_time:desc",
                          count: int = 100) -> list[dict]:
    url = urljoin(BASE, "scheduled_events")
    params = {
        "organization": org_uri,
        "status": status,
        "min_start_time": min_start_time,
        "sort": sort,
        "count": count
    }
    return _paginate(url, params)


def get_invitees(event_uri: str, status: str = "active") -> dict:
    url = f"{event_uri}/invitees"
    params = {"status": status, "count": 100}
    return _get(url, params)


def fetch_invitee(
        uri: str,
        statuses: list[str] | None = None) -> tuple[str | None, str | None]:
    if statuses is None:
        statuses = ["active", "canceled"]
    for s in statuses:
        data = get_invitees(uri, status=s) or {}
        coll = data.get("collection") or []
        if coll:
            first = coll[0]
            return first.get("name"), first.get("email")
    return None, None


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────
def main():
    # 1️⃣  Fetch / filter scheduled events
    org_uri = current_org_uri()
    cutoff = (dt.datetime.now(dt.timezone.utc) -
              dt.timedelta(days=120)).isoformat()

    active = list_scheduled_events(org_uri,
                                   status="active",
                                   min_start_time=cutoff)
    canceled = list_scheduled_events(org_uri,
                                     status="canceled",
                                     min_start_time=cutoff)
    events = active + canceled

    filtered = [e for e in events if e.get("event_type") in TARGET_EVENT_TYPES]
    print(f"🔍  Found {len(filtered)} matching events")

    if not filtered:
        sys.exit("❌  No events matched your filters")

    # 2️⃣  Build DataFrame
    df = pd.DataFrame(filtered)
    df = df.sort_values("start_time", ascending=False).reset_index(drop=True)

    # 3️⃣  Enrich with invitee_name & invitee_email in parallel
    df["invitee_name"] = None
    df["invitee_email"] = None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exec:
        futures = {
            exec.submit(fetch_invitee, uri): idx
            for idx, uri in df["uri"].items()
        }
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Enriching invitees"):
            idx = futures[fut]
            try:
                name, email = fut.result()
            except Exception:
                name, email = None, None
            df.at[idx, "invitee_name"] = name
            df.at[idx, "invitee_email"] = email

    # 4️⃣  Output summary & save
    print("\n🎯  Sample enriched events:")
    display_cols = [
        "name", "start_time", "status", "invitee_name", "invitee_email"
    ]
    print(df[display_cols].head().to_string(index=False))

    out_file = "calendly_enriched_events.csv"
    df.to_csv(out_file, index=False)
    print(f"\n💾  Saved consolidated data to {out_file}")


if __name__ == "__main__":
    main()
