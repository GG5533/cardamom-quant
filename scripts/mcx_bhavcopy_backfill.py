"""MCX cardamom Bhavcopy backfill via the real Bhav Copy UI (Commodity Wise tab).

    .venv/bin/python scripts/mcx_bhavcopy_backfill.py

MCX's bhavcopy page sits behind Akamai bot management (raw HTTP requests get
a 403 straight from the edge, even with a browser User-Agent). This drives
an actual headed Chromium instance through the real "Commodity Wise" search
— select Instrument/Commodity/Expiry, set a date range, click Show — the
same steps a person does by hand. It just automates the clicking instead of
spoofing the request.

One query per contract expiry since the 2025-07-29 relaunch (not one per
day): the Commodity Wise search takes a from/to date range for a single
expiry and returns every trading day for that contract in one response, so
~17 expiries cover the whole backfill instead of ~380 daily page loads.

Output columns are named to match what src/data/mcx_bhavcopy.py already
recognises (see _CANON in that file) — this is a plain drop-in file, no
parser changes needed.

Writes: data/raw/mcx/CommodityWise_CARDAMOM_backfill.csv
"""
from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "mcx" / "CommodityWise_CARDAMOM_backfill.csv"
RELAUNCH = datetime(2025, 7, 29)
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
FIELDS = ["Date", "Symbol", "Expiry Date", "Open", "High", "Low", "Close",
          "Previous Close", "Volume", "Value", "Open Interest"]


def set_date(page, field_id: str, dt: datetime) -> None:
    page.click(f"#{field_id}")
    page.wait_for_timeout(250)
    page.select_option("#ui-datepicker-div select.ui-datepicker-year", str(dt.year))
    page.wait_for_timeout(150)
    page.select_option("#ui-datepicker-div select.ui-datepicker-month", str(dt.month - 1))
    page.wait_for_timeout(150)
    page.click(f'#ui-datepicker-div a.ui-state-default:text-is("{dt.day}")')
    page.wait_for_timeout(150)


def relevant_expiries(page) -> list[str]:
    """Post-relaunch expiries, taken as the contiguous prefix of the
    dropdown before it drops back to the pre-2021 regime."""
    opts = [o.inner_text() for o in
            page.query_selector("#Expiry-CommodiyWise").query_selector_all("option")]
    out = []
    for o in opts[1:]:  # skip "Select Expiry Date"
        m = re.match(r"(\d{2})([A-Z]{3})(\d{4})", o)
        if not m:
            continue
        exp_dt = datetime.strptime(o, "%d%b%Y")
        if exp_dt < RELAUNCH:
            break  # hit the pre-2021 regime, stop
        out.append(o)
    return out


def fetch_expiry(page, expiry: str, to_date: datetime) -> list[dict]:
    page.select_option("#Instrument-CommodiyWise", label="FUTCOM")
    page.wait_for_timeout(400)
    page.select_option("#commodiy-CommodiyWise", label="CARDAMOM")
    page.wait_for_timeout(400)
    page.select_option("#Expiry-CommodiyWise", label=expiry)
    page.wait_for_timeout(300)
    set_date(page, "CommodiyWisefromDate", RELAUNCH)
    set_date(page, "CommodiyWiseToDate", to_date)

    holder: dict = {}

    def on_resp(r):
        if "GetCommoditywiseBhavCopy" in r.url:
            try:
                holder["body"] = r.json()
            except Exception as e:
                holder["err"] = str(e)

    page.on("response", on_resp)
    page.click("#btnCommodiyWise")
    page.wait_for_timeout(4000)
    page.remove_listener("response", on_resp)

    body = holder.get("body")
    if not body or not body.get("IsSuccess"):
        return []
    return body.get("Data") or []


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(user_agent=UA)
        page.goto("https://www.mcxindia.com/market-data/bhavcopy", timeout=30000)
        page.wait_for_timeout(1500)
        page.click("text=Commodiy Wise")
        page.wait_for_timeout(500)
        page.select_option("#Instrument-CommodiyWise", label="FUTCOM")
        page.wait_for_timeout(400)
        page.select_option("#commodiy-CommodiyWise", label="CARDAMOM")
        page.wait_for_timeout(400)
        expiries = relevant_expiries(page)
        print(f"expiries to fetch: {expiries}")

        rows: list[dict] = []
        for i, expiry in enumerate(expiries, 1):
            data = fetch_expiry(page, expiry, today)
            print(f"  [{i}/{len(expiries)}] {expiry}: {len(data)} rows")
            for d in data:
                rows.append({
                    "Date": d.get("DateDisplay"),
                    "Symbol": (d.get("Symbol") or "").strip(),
                    "Expiry Date": d.get("ExpiryDate"),
                    "Open": d.get("Open"),
                    "High": d.get("High"),
                    "Low": d.get("Low"),
                    "Close": d.get("Close"),
                    "Previous Close": d.get("PreviousClose"),
                    "Volume": d.get("Volume"),
                    "Value": d.get("Value"),
                    "Open Interest": d.get("OpenInterest"),
                })
        browser.close()

    if not rows:
        print("no rows fetched — nothing written")
        return

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
