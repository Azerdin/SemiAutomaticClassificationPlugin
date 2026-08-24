# -*- coding: utf-8 -*-
"""Mierzy czas samej detekcji wartości odstających, osobno dla obu zakresów odniesienia.

WEJŚCIE:      obszary treningowe wariantu oraz raster
WYJŚCIE:      wyniki_badan/agregacja_v2/czasy_detekcji.csv oraz tabela pracy
URUCHOMIENIE: python3 badania/wyniki/analiza_czasy_detekcji.py   (Python QGIS-LTR)
"""

import argparse
import collections
import csv
import os
import statistics as st
import sys
import time

import numpy as np
import remotior_sensus
from osgeo import ogr

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_ROOT, os.path.join(_ROOT, "badania", "wyniki")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import benchmark_outlier_removal as B
from core import outlier_filters as filters
from core.outlier_catalog import warp_roi_stack

OUT_DIR = os.environ.get(
    "AGREGACJA_DIR", B.DATA_DIR + "/agregacja_v2"
)

def _geometries(path):
    ds = ogr.Open(path)
    if ds is None:
        raise SystemExit("Nie można otworzyć %s" % path)
    layer = ds.GetLayer()
    srs = layer.GetSpatialRef()
    out = {}
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        out[feat.GetField("roi_id")] = (
            geom.Clone(), feat.GetField(B.ROI_CLASS_FIELD)
        )
    return out, srs

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wariant", default="1234567")
    parser.add_argument("--raster", default="L1C")
    parser.add_argument("--stan", default="zbledami",
                        choices=("poprawny", "zbledami"))
    parser.add_argument("--powtorzenia", type=int, default=3)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    vcfg = B.VARIANTS[args.wariant]
    areas, srs = _geometries(vcfg["roi"][args.stan])
    print("Wariant %s, raster %s, stan %s" % (args.wariant, args.raster, args.stan))
    print("Obszarów treningowych: %d\n" % len(areas))

    configs = [
        (label, steps) for label, steps, _, _ in
        B._build_configs(1, B.ALL_METHODS)
    ]
    print("Metod do zmierzenia: %d, powtórzeń pomiaru: %d\n"
          % (len(configs), args.powtorzenia))

    session = remotior_sensus.Session(
        n_processes=B.N_PROCESSES, available_ram=B.RAM_MB
    )
    band_catalog = session.bandset_catalog()
    band_catalog.create_bandset(paths=B.RASTERS[args.raster], bandset_number=1)
    B._set_wavelengths(band_catalog)
    paths = band_catalog.get(1).get_absolute_paths()
    source = filters.build_multiband_source(paths)
    ogr_driver = ogr.GetDriverByName("GPKG")

    print("Wycinanie fragmentów obrazu (poza pomiarem) ...")
    stacks = {}
    for roi_id, (geom, class_id) in areas.items():
        result = warp_roi_stack(source, geom, srs, roi_id, ogr_driver)
        if result is None:
            continue
        stacks[roi_id] = (result[0], class_id)
    print("  wycięto %d fragmentów\n" % len(stacks))

    pools = collections.defaultdict(list)
    for roi_id, (stack, class_id) in stacks.items():
        pools[class_id].append(stack)

    rows = []
    for label, steps in configs:
        roi_times = _measure_roi(stacks, steps, args.powtorzenia)
        class_times = _measure_class(pools, steps, args.powtorzenia)
        if not roi_times and not class_times:
            continue
        rows.append({
            "Metoda": label,
            "ROI_mediana_ms": "%.1f" % (1000 * st.median(roi_times)) if roi_times else "",
            "ROI_maks_ms": "%.1f" % (1000 * max(roi_times)) if roi_times else "",
            "Klasa_mediana_ms": "%.1f" % (1000 * st.median(class_times)) if class_times else "",
            "Klasa_maks_ms": "%.1f" % (1000 * max(class_times)) if class_times else "",
            "n_obszarow": len(roi_times),
            "n_klas": len(class_times),
        })
        print("  %-20s ROI %8.1f ms   klasa %8.1f ms"
              % (label,
                 1000 * st.median(roi_times) if roi_times else float("nan"),
                 1000 * st.median(class_times) if class_times else float("nan")))

    _save(rows, args)

def _measure_roi(stacks, steps, repeats):
    times = []
    for stack, _ in stacks.values():
        best = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            try:
                filters.run_pipeline(stack, steps)
            except Exception:
                best = None
                break
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        if best is not None:
            times.append(best)
    return times

def _measure_class(pools, steps, repeats):
    times = []
    for class_stacks in pools.values():
        columns = []
        for stack in class_stacks:
            flat = stack.reshape(stack.shape[0], -1)
            valid = np.all(~np.isnan(flat), axis=0)
            columns.append(flat[:, valid])
        if not columns:
            continue
        pool = np.concatenate(columns, axis=1)[:, None, :]
        best = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            try:
                filters.run_pipeline(pool, steps)
            except Exception:
                best = None
                break
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        if best is not None:
            times.append(best)
    return times

def _save(rows, args):
    if not rows:
        raise SystemExit("Brak wyników pomiaru.")
    rows.sort(key=lambda w: -float(w["ROI_mediana_ms"] or 0))
    path = os.path.join(OUT_DIR, "czasy_detekcji.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\n  zapisano %s (%d wierszy)" % (path, len(rows)))
    print("  wariant %s, raster %s, stan %s, powtórzeń %d"
          % (args.wariant, args.raster, args.stan, args.powtorzenia))

if __name__ == "__main__":
    main()
