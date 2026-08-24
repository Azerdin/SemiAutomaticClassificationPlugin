# -*- coding: utf-8 -*-
"""Wyznacza narzut czasowy mechanizmu, liczony różnicowo względem poziomu bazowego.

WEJŚCIE:      wyniki_badan/benchmark_results_v2
WYJŚCIE:      wyniki_badan/agregacja_v2/czasy_*.csv oraz podsumowanie_czasy.txt
URUCHOMIENIE: python3 badania/wyniki/analiza_czasy.py   (wymaga numpy)
"""

import argparse
import collections
import csv
import glob
import os
import statistics as st

_BADANIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANE = os.environ.get("DANE_DIR", os.path.join(_BADANIA, "dane_testowe"))
WYNIKI = os.environ.get("WYNIKI_BADAN_DIR",
                        os.path.join(_BADANIA, "wyniki_badan"))
RESULTS_DIR = os.environ.get("WYNIKI_DIR", WYNIKI + "/benchmark_results_v2")
OUT_DIR = os.environ.get("AGREGACJA_DIR", WYNIKI + "/agregacja_v2")

BASELINE_LABELS = ("Poziom bazowy", "Baseline")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wyniki", default=RESULTS_DIR)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    times = {}
    pixels = {}
    for path in sorted(glob.glob(os.path.join(args.wyniki, "*",
                                                 "wyniki_zbiorcze.csv"))):
        run_id = os.path.basename(os.path.dirname(path))
        with open(path, encoding="utf-8-sig") as fh:
            for w in csv.DictReader(fh):
                t = (w.get("Czas_s") or "").strip()
                if not t:
                    continue
                key = (run_id, w["Metoda"], w["Algorytm"])
                times[key] = float(t)
                removed = (w.get("Usuniete_px") or "").strip()
                if removed:
                    pixels[key] = int(float(removed))
    if not times:
        raise SystemExit("Brak danych o czasie w %s" % args.wyniki)

    overhead = collections.defaultdict(list)
    share = collections.defaultdict(list)
    for (run_id, method, algorithm), t in times.items():
        if method in BASELINE_LABELS:
            continue
        baseline_time = None
        for name in BASELINE_LABELS:
            baseline_time = times.get((run_id, name, algorithm))
            if baseline_time is not None:
                break
        if baseline_time is None:
            continue
        overhead[method].append(t - baseline_time)
        if baseline_time > 0:
            share[method].append(100.0 * (t - baseline_time) / baseline_time)

    per_alg = collections.defaultdict(list)
    for (_, method, algorithm), t in times.items():
        if method in BASELINE_LABELS:
            per_alg[algorithm].append(t)

    _save_methods(overhead, share, pixels, times)
    _save_algorithms(per_alg)
    _summarise(times, overhead, per_alg, args.wyniki)

def _save_methods(overhead, share, pixels, times):
    rows = []
    for method in sorted(overhead, key=lambda m: st.median(overhead[m])):
        d = overhead[method]
        u = share.get(method, [0.0])
        removed = [v for (_, m, _), v in pixels.items() if m == method]
        rows.append({
            "Metoda": method,
            "Narzut_mediana_s": "%.2f" % st.median(d),
            "Narzut_srednia_s": "%.2f" % st.mean(d),
            "Narzut_mediana_%": "%.1f" % st.median(u),
            "Usuniete_px_mediana": ("%d" % st.median(removed)) if removed else "",
            "n": len(d),
        })
    _csv("czasy_metody.csv", rows)

def _save_algorithms(per_alg):
    rows = []
    for algorithm in sorted(per_alg, key=lambda a: -st.median(per_alg[a])):
        v = per_alg[algorithm]
        rows.append({
            "Algorytm": algorithm,
            "Czas_mediana_s": "%.1f" % st.median(v),
            "Czas_srednia_s": "%.1f" % st.mean(v),
            "Czas_min_s": "%.1f" % min(v),
            "Czas_max_s": "%.1f" % max(v),
            "n": len(v),
        })
    _csv("czasy_algorytmy.csv", rows)

def _csv(name, rows):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("  zapisano %s (%d wierszy)" % (name, len(rows)))

def _relative_path(p):
    w = os.path.relpath(os.path.abspath(p), _BADANIA)
    if w.startswith(os.pardir):
        return os.path.basename(os.path.abspath(p))
    return w

def _summarise(times, overhead, per_alg, source):
    all_overheads = [x for v in overhead.values() for x in v]
    lines = [
        "CZAS PRZETWARZANIA",
        "=" * 70,
        "źródło danych: %s" % _relative_path(source),
        "pomiarów: %d, łączny czas obliczeń: %.1f h"
        % (len(times), sum(times.values()) / 3600.0),
        "",
        "NARZUT MECHANIZMU (różnicowo względem poziomu bazowego)",
        "  mediana narzutu:   %+.2f s" % st.median(all_overheads),
        "  średnia narzutu:   %+.2f s" % st.mean(all_overheads),
        "  zakres median metod: od %+.2f do %+.2f s"
        % (min(st.median(v) for v in overhead.values()),
           max(st.median(v) for v in overhead.values())),
        "",
        "CZAS KLASYFIKACJI (poziom bazowy, bez usuwania)",
    ]
    for algorithm in sorted(per_alg, key=lambda a: -st.median(per_alg[a])):
        lines.append("  %-24s mediana %6.1f s"
                     % (algorithm, st.median(per_alg[algorithm])))
    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "podsumowanie_czasy.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text)
    print("\nWyniki w: %s" % OUT_DIR)

if __name__ == "__main__":
    main()
