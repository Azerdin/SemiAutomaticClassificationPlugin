# -*- coding: utf-8 -*-
"""Zlicza, jak często poszczególne metody detekcji dawały najlepszy wynik.

WEJŚCIE:      wyniki_badan/benchmark_results_v2/*/konfirmacja_mcnemar.csv
WYJŚCIE:      wyniki_badan/agregacja_v2/T8_*.csv oraz T9_*.csv
URUCHOMIENIE: python3 badania/wyniki/analiza_najlepszych_metod.py   (wymaga numpy)
"""

import collections
import csv
import os
import statistics as st

import agreguj_wyniki as A

def _delta(r):
    return A._f(r["Delta_DA_pp"])

def per_classifier(rows):
    by_alg = collections.defaultdict(list)
    for r in rows:
        by_alg[r["Algorytm"]].append(r)
    out = []
    for alg in A.ALGO_ORDER:
        g = by_alg.get(alg, [])
        if not g:
            continue
        wins = collections.Counter(r["Najlepsza_konfiguracja"] for r in g)
        method, w = wins.most_common(1)[0]
        deltas = [_delta(r) for r in g if r["Najlepsza_konfiguracja"] == method]
        pos = sum(1 for r in g if _delta(r) > 0)
        out.append((alg, method, w, len(g), round(st.mean(deltas), 2), pos))
    return out

def overall(rows):
    by_m = collections.defaultdict(list)
    for r in rows:
        by_m[r["Najlepsza_konfiguracja"]].append(r)
    total = len(rows)
    out = []
    for m, g in by_m.items():
        d = [_delta(r) for r in g]
        out.append((m, len(g), round(100.0 * len(g) / total, 1),
                    round(st.mean(d), 2), round(st.median(d), 2),
                    sum(1 for r in g if A._sig(r))))
    out.sort(key=lambda x: (-x[1], -x[3]))
    return out

def _write_csv(fname, header, rows):
    os.makedirs(A.OUT_DIR, exist_ok=True)
    with open(os.path.join(A.OUT_DIR, fname), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  zapisano %s (%d wierszy)" % (fname, len(rows)))

def main():
    rows = A.load_konfirmacje(A.RESULTS_DIR)
    print("Wczytano %d przebiegów.\n" % len(rows))

    t1 = per_classifier(rows)
    print("=== T8 — najczęściej najlepsza metoda per algorytm ===")
    print("%-24s %-24s %8s %10s %8s" %
          ("Algorytm", "Metoda", "zwyc./n", "sr.ΔOA", "ΔOA>0"))
    for alg, m, w, n, d, pos in t1:
        print("%-24s %-24s %5d/%-3d %+9.2f %6d/%d" % (alg, m, w, n, d, pos, n))
    _write_csv("T8_najlepsza_metoda_per_algorytm.csv",
               ["Algorytm", "Metoda", "Zwyciestwa", "n", "sr_delta_pp",
                "wygrane_dodatnie"],
               [[alg, m, w, n, d, pos] for (alg, m, w, n, d, pos) in t1])

    t2 = overall(rows)
    print("\n=== T9 — najlepsza metoda globalnie (top 12) ===")
    print("%-24s %8s %8s %8s %8s %8s" %
          ("Metoda", "zwyc.", "udz.%", "sr.ΔOA", "med.ΔOA", "istotn."))
    for m, w, share, d, med, sig in t2[:12]:
        print("%-24s %8d %7.1f %+8.2f %+8.2f %8d" % (m, w, share, d, med, sig))
    _write_csv("T9_najlepsza_metoda_globalnie.csv",
               ["Metoda", "Zwyciestwa", "Udzial_%", "sr_delta_pp",
                "mediana_delta_pp", "istotnych"],
               [[m, w, share, d, med, sig] for (m, w, share, d, med, sig) in t2])

    print("\nCSV w %s." % A.OUT_DIR)

if __name__ == "__main__":
    main()
