"""Re-download + re-extract the Banguat annual volume series.

    python scripts/refresh_banguat.py   # needs requests + openpyxl

Writes the pristine xlsx to data/raw/banguat/ and regenerates the
committed CSV extraction src/data/banguat.py parses. Verify the tail
against banguat.gob.gt before committing a refresh.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

URL = ("https://www.banguat.gob.gt/sites/default/files/banguat/estaeco/"
       "comercio/sercom/11_VOLUMEN%201994-2022/volumen_1994_2025.xlsx")
RAW = ROOT / "data" / "raw" / "banguat"


def main() -> None:
    import requests

    xlsx = RAW / "volumen_1994_2025.xlsx"
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    xlsx.write_bytes(r.content)

    df = pd.read_excel(xlsx, header=None)  # openpyxl
    hdr = df[df.iloc[:, 0].astype(str).str.strip().eq("AÑO")].index[0]
    cols = [str(c).strip() for c in df.iloc[hdr]]
    out = []
    for _, row in df.iloc[hdr + 1:].iterrows():
        m = re.match(r"^(\d{4})", str(row.iloc[0]).strip())
        if not m:
            continue
        rec = {"year": int(m.group(1))}
        for i, c in enumerate(cols[1:], start=1):
            rec[c.lower().replace("é", "e").replace("ú", "u")] = float(row.iloc[i])
        out.append(rec)
    t = pd.DataFrame(out).set_index("year")
    t.columns = [f"{c}_mm_qq" for c in t.columns]
    t.to_csv(RAW / "volumen_agricolas_annual.csv")
    print(t.tail(3).round(4).to_string())
    print(f"wrote {len(t)} years -> {RAW / 'volumen_agricolas_annual.csv'}")


if __name__ == "__main__":
    main()
