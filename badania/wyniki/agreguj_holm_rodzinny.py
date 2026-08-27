# -*- coding: utf-8 -*-
"""Stosuje korekcję Holma nad całą rodziną porównań konfirmacyjnych, w dwóch zasięgach rodziny.

WEJŚCIE:      wyniki_badan/benchmark_results_v2/*/konfirmacja_mcnemar.csv
WYJŚCIE:      zestawienie na konsoli
URUCHOMIENIE: python3 badania/wyniki/agreguj_holm_rodzinny.py
"""

import csv
import glob
import os
from collections import defaultdict

_BADANIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WYNIKI = os.environ.get("WYNIKI_BADAN_DIR",
                        os.path.join(_BADANIA, "wyniki_badan"))

RESULTS_DIR = os.environ.get("WYNIKI_DIR",
                             os.path.join(WYNIKI, "benchmark_results_v2"))

ALGO_ORDER = ["maximum likelihood", "random forest", "multi-layer perceptron",
              "support vector machine", "spectral angle mapping",
              "minimum distance"]
ALGO_SHORT = {"maximum likelihood": "MLC", "random forest": "RF",
              "multi-layer perceptron": "MLP", "support vector machine": "SVM",
              "spectral angle mapping": "SAM", "minimum distance": "min. odl."}
ALPHA = 0.05

def holm_adjusted(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj

def load_rows(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*",
                                              "konfirmacja_mcnemar.csv"))):
        with open(path, encoding="utf-8-sig") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        raise SystemExit("Brak plików konfirmacja_mcnemar.csv w %s" % results_dir)
    return rows

def _count(rows, adj):
    total = sum(1 for a in adj if a <= ALPHA)
    by_algo = defaultdict(lambda: [0, 0])
    for r, a in zip(rows, adj):
        by_algo[r["Algorytm"]][1] += 1
        if a <= ALPHA:
            by_algo[r["Algorytm"]][0] += 1
    return total, by_algo

def family_192(rows):
    best = {}
    for r in rows:
        key = (r["Algorytm"], r["Wariant"], r["Stan"], r["Raster"])
        da = float(r["DA_najlepsza_%"])
        if key not in best or da > float(best[key]["DA_najlepsza_%"]):
            best[key] = r
    return list(best.values())

def main():
    rows = load_rows(RESULTS_DIR)
    print("Wczytano %d porównań konfirmacyjnych (64 przebiegi x 6 algorytmów).\n"
          % len(rows))

    p_all = [float(r["p"]) for r in rows]
    raw_sig = sum(1 for x in p_all if x <= ALPHA)
    local_sig = sum(1 for r in rows
                    if r["Istotne_005_lokalnie"].strip().upper() == "TAK")

    fam384_total, fam384_alg = _count(rows, holm_adjusted(p_all))

    rows192 = family_192(rows)
    p192 = [float(r["p"]) for r in rows192]
    fam192_total, fam192_alg = _count(rows192, holm_adjusted(p192))

    print("ISTOTNYCH przy alpha = 0,05:")
    print("  bez korekcji (surowe p):              %3d / %d" % (raw_sig, len(rows)))
    print("  korekcja LOKALNA (per przebieg):      %3d / %d" % (local_sig, len(rows)))
    print("  rodzina 384 (zakres jako wymiar):     %3d / %d" % (fam384_total, len(rows)))
    print("  rodzina 192 (zakres w konfiguracji):  %3d / %d" % (fam192_total, len(rows192)))

    print("\nPer algorytm (istotnych):")
    print("  %-12s %-10s %-10s %-10s" % ("algorytm", "lokalny", "rodz.384", "rodz.192"))
    loc_alg = defaultdict(lambda: [0, 0])
    for r in rows:
        loc_alg[r["Algorytm"]][1] += 1
        if r["Istotne_005_lokalnie"].strip().upper() == "TAK":
            loc_alg[r["Algorytm"]][0] += 1
    for a in ALGO_ORDER:
        print("  %-12s %-10s %-10s %-10s" % (
            ALGO_SHORT[a],
            "%d/%d" % tuple(loc_alg[a]),
            "%d/%d" % tuple(fam384_alg[a]),
            "%d/%d" % tuple(fam192_alg[a])))

if __name__ == "__main__":
    main()
