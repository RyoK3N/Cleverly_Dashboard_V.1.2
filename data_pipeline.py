#!/usr/bin/env python3
"""
data_pipeline.py
─────────────────────
Fetch Monday.com board data and Calendly scheduled events concurrently,
then annotate Monday data with origin based on Calendly invitee emails.
"""
import os
import sys
import time

datetime_import = "unused"
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
import dotenv
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

# Import Monday helper functions
from monday_extract_groups import fetch_items_recursive, fetch_groups

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# Monday.com settings
dotenv.load_dotenv()


class MondayConfig:
    BOARD_ID: str = os.getenv("MONDAY_BOARD_ID", "6942829967")
    ITEMS_LIMIT: int = int(os.getenv("MONDAY_ITEMS_LIMIT", "500"))
    GROUP_MAPPING = {
        "topics": "scheduled",
        "new_group34578__1": "unqualified",
        "new_group27351__1": "won",
        "new_group54376__1": "cancelled",
        "new_group64021__1": "noshow",
        "new_group65903__1": "proposal",
        "new_group62617__1": "lost"
    }
    COLUMN_MAPPING = {
        'name': 'Name',
        'auto_number__1': 'Auto number',
        'person': 'Owner',
        'last_updated__1': 'Last updated',
        'link__1': 'Linkedin',
        'phone__1': 'Phone',
        'email__1': 'Email',
        'text7__1': 'Company',
        'date4': 'Sales Call Date',
        'status9__1': 'Follow Up Tracker',
        'notes__1': 'Notes',
        'interested_in__1': 'Interested In',
        'status4__1': 'Plan Type',
        'numbers__1': 'Deal Value',
        'status6__1': 'Email Template #1',
        'dup__of_email_template__1': 'Email Template #2',
        'status__1': 'Deal Status',
        'status2__1': 'Send Panda Doc?',
        'utm_source__1': 'UTM Source',
        'date__1': 'Deal Status Date',
        'utm_campaign__1': 'UTM Campaign',
        'utm_medium__1': 'UTM Medium',
        'utm_content__1': 'UTM Content',
        'link3__1': 'UTM LINK',
        'lead_source8__1': 'Lead Source',
        'color__1': 'Channel FOR FUNNEL METRICS',
        'subitems__1': 'Subitems',
        'date5__1': 'Date Created'
    }


# Calendly settings
BASE = "https://api.calendly.com/"
TARGET_EVENT_TYPES = [
    "https://api.calendly.com/event_types/3f3b8e40-246e-4723-8690-d0de0419e231",
    "https://api.calendly.com/event_types/6b4aa5e3-b4a2-4ef2-b1b2-1405b02e9806"
]
CAL_MAX_WORKERS = int(os.getenv("CAL_MAX_WORKERS", "5"))
CAL_THROTTLE = float(os.getenv("CAL_THROTTLE", "0.2"))

# ─── SHARED SESSION WITH RETRIES ───────────────────────────────────────────────
session = requests.Session()
retry_strategy = Retry(total=5,
                       backoff_factor=1,
                       status_forcelist=[429, 500, 502, 503, 504],
                       allowed_methods=["GET"],
                       raise_on_status=False)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)


# ─── MONDAY PIPELINE ──────────────────────────────────────────────────────────
class MondayDataProcessor:

    def __init__(self, config: MondayConfig):
        self.config = config
        key = os.getenv("MONDAY_API_KEY")
        if not key:
            raise ValueError("MONDAY_API_KEY not set in environment")
        self.api_key = key

    def _items_to_df(self, items: List[dict]) -> pd.DataFrame:
        if not items or not items[0].get('column_values'):
            return pd.DataFrame()
        cols = [c['id'] for c in items[0]['column_values']]
        rows = []
        for it in items:
            row = {'Item ID': it['id'], 'Item Name': it['name']}
            for col in it['column_values']:
                row[col['id']] = col.get('text', '')
            rows.append(row)
        df = pd.DataFrame(rows, columns=['Item ID', 'Item Name'] + cols)
        return df.rename(columns=self.config.COLUMN_MAPPING)

    def fetch(self) -> pd.DataFrame:
        groups = fetch_groups(self.config.BOARD_ID, self.api_key)
        parts = []
        for gid, gname in self.config.GROUP_MAPPING.items():
            grp = next((g for g in groups if g['id'] == gid), None)
            if not grp:
                continue
            items = fetch_items_recursive(self.config.BOARD_ID, gid,
                                          self.api_key,
                                          self.config.ITEMS_LIMIT)
            df = self._items_to_df(items)
            if not df.empty:
                df['Group'] = gname
                parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ─── CALENDLY PIPELINE ─────────────────────────────────────────────────────────
# Load Calendly key
dotenv.load_dotenv()
CAL_KEY = os.getenv("CALENDLY_API_KEY")
if not CAL_KEY:
    sys.exit("CALENDLY_API_KEY not set")
HEADERS_CAL = {
    "Authorization": f"Bearer {CAL_KEY}",
    "Content-Type": "application/json"
}


#{helper functions}
def _get_cal(url: str, params: dict = None, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            r = session.get(url,
                            headers=HEADERS_CAL,
                            params=params,
                            timeout=30)
            if r.status_code == 429:
                t = int(r.headers.get('Retry-After', CAL_THROTTLE))
                time.sleep(t)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == retries - 1:
                return {'collection': []}
            time.sleep(2**i)
    return {'collection': []}


def _paginate_cal(url: str, params: dict) -> List[dict]:
    allc = []
    while url:
        page = _get_cal(url, params)
        allc.extend(page.get('collection', []))
        url = page.get('pagination', {}).get('next_page')
        params = None
    return allc


def _org_uri():
    js = _get_cal(urljoin(BASE, 'users/me'))
    return js.get('resource', {}).get('current_organization', '')


def list_events(stat: str, cutoff: str) -> List[dict]:
    url = urljoin(BASE, 'scheduled_events')
    return _paginate_cal(
        url, {
            'organization': _org_uri(),
            'status': stat,
            'min_start_time': cutoff,
            'count': 100
        })


def fetch_calendly() -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    evts = list_events('active', cutoff) + list_events('canceled', cutoff)
    filt = [e for e in evts if e.get('event_type') in TARGET_EVENT_TYPES]
    df = pd.DataFrame(filt)
    if df.empty:
        return df
    df.sort_values('start_time', inplace=True, ascending=False)
    # enrich invitee
    df['invitee_name'] = None
    df['invitee_email'] = None

    def _fetch(uri):
        for s in ['active', 'canceled']:
            j = _get_cal(f"{uri}/invitees", {'status': s, 'count': 100})
            c = j.get('collection') or []
            if c:
                return c[0].get('name'), c[0].get('email')
        return None, None

    with ThreadPoolExecutor(max_workers=CAL_MAX_WORKERS) as ex:
        futs = {ex.submit(_fetch, uri): i for i, uri in df['uri'].items()}
        for f in tqdm(as_completed(futs), total=len(futs), desc='Invites'):
            i = futs[f]
            try:
                n, e = f.result()
            except:
                n, e = None, None
            df.at[i, 'invitee_name'] = n
            df.at[i, 'invitee_email'] = e
    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    with ThreadPoolExecutor(max_workers=2) as ex:
        mon_f = ex.submit(MondayDataProcessor(MondayConfig()).fetch)
        cal_f = ex.submit(fetch_calendly)
        monday_df = mon_f.result()
        calendly_df = cal_f.result()
    if monday_df.empty:
        print('No Monday data')
        sys.exit(1)
    if calendly_df.empty:
        print('No Calendly data')
        sys.exit(1)
    # annotate origin
    monday_df['origin'] = 'LinkedIn'
    emails_set = set(calendly_df['invitee_email'].dropna())
    monday_df.loc[monday_df['Email'].isin(emails_set), 'origin'] = 'cold-email'
    # output
    out = monday_df
    print(out.head().to_string(index=False))
    out.to_csv('combined_output.csv', index=False)


if __name__ == '__main__':
    main()
