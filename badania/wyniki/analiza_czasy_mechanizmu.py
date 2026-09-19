# -*- coding: utf-8 -*-
"""Mierzy czas pełnej operacji usuwania wartości odstających, z podziałem na fazy.

WEJŚCIE:      obszary treningowe wariantu oraz raster
WYJŚCIE:      wyniki_badan/czasy_mechanizmu/czasy_mechanizmu.csv
URUCHOMIENIE: python3 badania/wyniki/analiza_czasy_mechanizmu.py   (Python QGIS-LTR)
"""

import argparse
import csv
import os
import statistics as st
import sys
import time

import remotior_sensus

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_ROOT, os.path.join(_ROOT, "badania", "wyniki")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import benchmark_outlier_removal as B
from core import outlier_catalog as catalog_ops
from core import outlier_filters as filters

OUT_DIR = os.environ.get(
    "CZASY_MECHANIZMU_DIR", B.RESULTS_ROOT + "/czasy_mechanizmu"
)


def _resolve_method(name):
    if name in B._DEFAULT_PARAMS:
        return name
    for key in B._DEFAULT_PARAMS:
        if B._rename_pca(key) == name:
            return key
    return None

PHASE_WARP = "wycinanie"
PHASE_DETECTION = "detekcja"
PHASE_SIGNATURES = "sygnatury"
PHASES = (PHASE_WARP, PHASE_DETECTION, PHASE_SIGNATURES)

_totals = {}
_counts = {}
_originals = {}


def _reset_totals():
    for phase in PHASES:
        _totals[phase] = 0.0
        _counts[phase] = 0


def _make_probe(func, phase):
    def probe(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            _totals[phase] += time.perf_counter() - start
            _counts[phase] += 1

    return probe


def _install_probes():
    targets = [
        (catalog_ops, "warp_roi_stack", PHASE_WARP),
        (catalog_ops, "recalculate_modified_signatures", PHASE_SIGNATURES),
        (filters, "run_pipeline", PHASE_DETECTION),
        (filters, "run_pipeline_group", PHASE_DETECTION),
    ]
    for module, name, phase in targets:
        original = getattr(module, name)
        _originals[(module, name)] = original
        setattr(module, name, _make_probe(original, phase))


def _restore_probes():
    for (module, name), original in _originals.items():
        setattr(module, name, original)
    _originals.clear()


def _measure(rs, scpx_path, bandset, steps, scope, repeats):
    runs = []
    for _ in range(repeats):
        catalog = B._load_fresh_catalog(rs, scpx_path, bandset)
        _reset_totals()
        start = time.perf_counter()
        try:
            modified, deleted, _report = catalog_ops.apply_removal_to_catalog(
                catalog,
                lambda *_: True,
                steps,
                False,
                2,
                scope=scope,
                temp_path_fn=lambda: B._tmp(".gpkg"),
                warn_missing=False,
            )
        except Exception as err:
            print("    BŁĄD: %s" % err)
            return None
        total = time.perf_counter() - start
        runs.append(
            {
                "total": total,
                "phases": dict(_totals),
                "counts": dict(_counts),
                "modified": len(modified),
                "deleted": len(deleted),
            }
        )
    return runs


def _summarise(runs):
    best = min(runs, key=lambda r: r["total"])
    other = sum(best["phases"][p] for p in PHASES)
    return {
        "total": best["total"],
        "warp": best["phases"][PHASE_WARP],
        "detection": best["phases"][PHASE_DETECTION],
        "signatures": best["phases"][PHASE_SIGNATURES],
        "rest": max(best["total"] - other, 0.0),
        "counts": best["counts"],
        "modified": best["modified"],
        "deleted": best["deleted"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wariant", default="1234567")
    parser.add_argument("--raster", default="L1C")
    parser.add_argument("--stan", default="zbledami",
                        choices=("poprawny", "zbledami"))
    parser.add_argument("--zakresy", default="roi,class")
    parser.add_argument("--metody", default="",
                        help="lista metod rozdzielona przecinkami; "
                             "pominięcie oznacza wszystkie metody")
    parser.add_argument("--powtorzenia", type=int, default=1)
    parser.add_argument("--nadpisz", action="store_true",
                        help="pozwala nadpisać istniejący plik wynikowy")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "czasy_mechanizmu.csv")
    if os.path.exists(out_path) and not args.nadpisz:
        raise SystemExit(
            "Plik %s już istnieje. Użyj --nadpisz albo wskaż inny katalog "
            "zmienną CZASY_MECHANIZMU_DIR." % out_path
        )

    requested = ([m.strip() for m in args.metody.split(",") if m.strip()]
                 or list(B.ALL_METHODS))
    methods = []
    unknown = []
    for name in requested:
        key = _resolve_method(name)
        (methods if key else unknown).append(key or name)
    if unknown:
        raise SystemExit("Nieznane metody: %s" % ", ".join(unknown))
    scopes = [s.strip() for s in args.zakresy.split(",") if s.strip()]

    vcfg = B.VARIANTS[args.wariant]
    roi_path = vcfg["roi"][args.stan]
    image_path = B.RASTERS[args.raster]

    print("Wariant %s, raster %s, stan %s" % (args.wariant, args.raster, args.stan))
    print("Metod: %d, zakresów: %d, powtórzeń: %d\n"
          % (len(methods), len(scopes), args.powtorzenia))

    session = remotior_sensus.Session(
        n_processes=B.N_PROCESSES, available_ram=B.RAM_MB
    )
    band_catalog = session.bandset_catalog()
    band_catalog.create_bandset(paths=B.RASTERS[args.raster], bandset_number=1)
    B._set_wavelengths(band_catalog)
    bandset = band_catalog.get(1)

    print("Przygotowanie katalogu sygnatur (poza pomiarem) ...")
    scpx_path = B._prepare_signatures_scpx(
        session, bandset, roi_path, image_path, OUT_DIR
    )

    _install_probes()
    rows = []
    try:
        for scope in scopes:
            print("\nZakres %s" % scope)
            for method in methods:
                steps = [(method, B._DEFAULT_PARAMS[method])]
                runs = _measure(
                    session, scpx_path, bandset, steps, scope, args.powtorzenia
                )
                if runs is None:
                    continue
                s = _summarise(runs)
                rows.append({
                    "Wariant": args.wariant,
                    "Stan": args.stan,
                    "Raster": args.raster,
                    "Zakres": scope,
                    "Metoda": B._rename_pca(method),
                    "Calosc_s": "%.4f" % s["total"],
                    "Wycinanie_s": "%.4f" % s["warp"],
                    "Detekcja_s": "%.4f" % s["detection"],
                    "Sygnatury_s": "%.4f" % s["signatures"],
                    "Pozostale_s": "%.4f" % s["rest"],
                    "Udzial_sygnatur_%": "%.1f" % (
                        100.0 * s["signatures"] / s["total"] if s["total"] else 0.0
                    ),
                    "Udzial_detekcji_%": "%.1f" % (
                        100.0 * s["detection"] / s["total"] if s["total"] else 0.0
                    ),
                    "Wywolan_wycinania": s["counts"][PHASE_WARP],
                    "Wywolan_detekcji": s["counts"][PHASE_DETECTION],
                    "Obszarow_zmienionych": s["modified"],
                    "Obszarow_usunietych": s["deleted"],
                })
                print("  %-20s całość %6.2f s | wycinanie %5.2f | detekcja %5.2f "
                      "| sygnatury %5.2f | reszta %5.2f"
                      % (B._rename_pca(method), s["total"], s["warp"], s["detection"],
                         s["signatures"], s["rest"]))
    finally:
        _restore_probes()

    if not rows:
        raise SystemExit("Brak wyników pomiaru.")

    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\n  zapisano %s (%d wierszy)" % (out_path, len(rows)))

    for scope in scopes:
        sub = [r for r in rows if r["Zakres"] == scope]
        if not sub:
            continue
        print("\n  Zakres %s, mediany:" % scope)
        print("    całość:      %6.2f s" % st.median(float(r["Calosc_s"]) for r in sub))
        print("    wycinanie:   %6.2f s" % st.median(float(r["Wycinanie_s"]) for r in sub))
        print("    detekcja:    %6.2f s" % st.median(float(r["Detekcja_s"]) for r in sub))
        print("    sygnatury:   %6.2f s" % st.median(float(r["Sygnatury_s"]) for r in sub))
        print("    pozostałe:   %6.2f s" % st.median(float(r["Pozostale_s"]) for r in sub))


if __name__ == "__main__":
    main()
