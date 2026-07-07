"""Repair thousands-separator commas in the browser-crawled sessions CSV.

Each data line is: date,auctioneer,<5 numeric fields>, where numeric fields
may contain standard 3-digit grouping commas ("11,409.7"). We enumerate all
ways to regroup the numeric fragments into exactly 5 numbers under grouping
rules, and require a unique sanity-valid partition. Anything ambiguous or
invalid is quarantined, never guessed.
"""
from __future__ import annotations
import re, sys

FRAG_FIRST = re.compile(r"^\d{1,3}$")
FRAG_CONT = re.compile(r"^\d{3}$")
FRAG_LAST = re.compile(r"^\d{3}\.\d+$")
LONE = re.compile(r"^\d+(\.\d+)?$")

def partitions(frags, k):
    if k == 1:
        n = join_ok(frags)
        return [[n]] if n is not None else []
    out = []
    for i in range(1, len(frags) - k + 2):
        head = join_ok(frags[:i])
        if head is None: continue
        for rest in partitions(frags[i:], k - 1):
            out.append([head] + rest)
    return out

def join_ok(frags):
    if len(frags) == 1:
        return float(frags[0]) if LONE.match(frags[0]) else None
    if not FRAG_FIRST.match(frags[0]): return None
    for f in frags[1:-1]:
        if not FRAG_CONT.match(f): return None
    last = frags[-1]
    if not (FRAG_CONT.match(last) or FRAG_LAST.match(last)): return None
    return float("".join(frags[:-1]) + last.replace(",", "")) if True else None

def sane(nums):
    lots, arr, sold, mx, avg = nums
    return (lots == int(lots) and 0 < lots < 5000 and 0 < arr < 2e6 and 0 <= sold < 2e6
            and sold <= arr * 1.10 and 50 <= mx <= 25000 and 50 <= avg <= 25000 and avg <= mx * 1.001)

def main(src, dst):
    good, bad = [], []
    with open(src) as f:
        header = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line or line.startswith("Date of Auction"): continue
            parts = line.split(",")
            date, auct = parts[0], parts[1]
            frags = parts[2:]
            if len(frags) == 5 and all(LONE.match(x) for x in frags):
                nums = [float(x) for x in frags]
                (good if sane(nums) else bad).append((line, nums if sane(nums) else None) if False else line) \
                    if False else (good.append((date, auct, nums)) if sane(nums) else bad.append(line))
                continue
            cands = [p for p in partitions(frags, 5) if sane(p)]
            if len(cands) == 1:
                good.append((date, auct, cands[0]))
            else:
                bad.append(f"{line}  [{len(cands)} candidates]")
    with open(dst, "w") as f:
        f.write("date,auctioneer,n_lots,qty_arrived,qty_sold,spot_max,spot_avg\n")
        for date, auct, n in good:
            f.write(f"{date},{auct},{int(n[0])},{n[1]},{n[2]},{n[3]},{n[4]}\n")
    with open(dst + ".rejected", "w") as f:
        f.write("\n".join(bad))
    print(f"repaired: {len(good)} rows, rejected: {len(bad)}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
