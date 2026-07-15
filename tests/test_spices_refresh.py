"""Tests for archive-page parsing across markup eras + refresh key matching
(offline)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.refresh_spot import canon_auctioneer  # noqa: E402
from src.data.spices_board import SpicesBoardLoader  # noqa: E402

_ROW = ("<tr><td>01-Jul-2026</td><td>Spice More Trading Company, Kumily</td>"
        "<td>248</td><td>82,601.6</td><td>81,267.5</td><td>3,414.00</td>"
        "<td>2,843.13</td></tr>")
_HDR_CELLS = ("Date of Auction,Auctioneer,No.of Lots,Total Qty Arrived (Kgs),"
              "Qty Sold (Kgs),MaxPrice (Rs./Kg),Avg.Price (Rs./Kg)").split(",")


def _page(th_header: bool) -> str:
    tag = "th" if th_header else "td"
    hdr = "".join(f"<{tag}>{c}</{tag}>" for c in _HDR_CELLS)
    return f"<html><body><table><tr>{hdr}</tr>{_ROW}</table></body></html>"


def test_parse_th_header_markup():
    rows = SpicesBoardLoader._parse_html(_page(th_header=True))
    assert len(rows) == 1
    assert rows.iloc[0]["spot_avg"] == 2843.13
    assert rows.iloc[0]["qty_arrived"] == 82601.6  # thousands commas handled


def test_parse_plain_row_header_markup():
    """The Jul-2026 site era: header cells are <td>, read_html sees ints."""
    rows = SpicesBoardLoader._parse_html(_page(th_header=False))
    assert len(rows) == 1
    assert rows.iloc[0]["spot_avg"] == 2843.13
    assert rows.iloc[0]["date"] == pd.Timestamp("2026-07-01")


def test_canon_auctioneer_bridges_comma_semicolon():
    site = pd.Series(["Spice More Trading Company, Kumily",
                      "Mas  Enterprises, Vandanmettu"])
    dump = pd.Series(["Spice More Trading Company; Kumily",
                      "Mas Enterprises; Vandanmettu"])
    assert (canon_auctioneer(site) == canon_auctioneer(dump)).all()
