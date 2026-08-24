# -*- coding: utf-8 -*-
"""Zestawia udział klas w scenie z ich udziałem w próbie referencyjnej.

WEJŚCIE:      rastry klasyfikacji poziomu bazowego oraz warstwa referencyjna
WYJŚCIE:      zestawienie na konsoli
URUCHOMIENIE: python3 badania/weryfikacja/porownaj_udzialy_klas.py   (Python QGIS-LTR)
"""

import argparse
import collections
import glob
import os
import statistics as st
import sys

import numpy as np
from osgeo import gdal, ogr

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_ROOT, os.path.join(_ROOT, "badania", "wyniki")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import benchmark_outlier_removal as B

RESULTS_DIR = os.environ.get("WYNIKI_DIR", B.DATA_DIR + "/benchmark_results_v2")
CLASS_NAMES = {
    1: "pole zaorane", 2: "droga", 3: "plaża", 4: "las",
    5: "zabudowa", 6: "woda", 7: "pole z roślinnością",
}

def scene_shares(run_dir):
    shares = collections.defaultdict(list)
    pattern = os.path.join(run_dir, "classifications", "Baseline__*.tif")
    for path in sorted(glob.glob(pattern)):
        raster = gdal.Open(path)
        if raster is None:
            continue
        values = raster.GetRasterBand(1).ReadAsArray()
        raster = None
        classes, counts = np.unique(values[values > 0], return_counts=True)
        total = int(counts.sum())
        counted = dict(zip([int(c) for c in classes], [int(n) for n in counts]))
        for class_id in counted:
            shares[class_id].append(100.0 * counted[class_id] / total)
    return shares

def reference_counts(reference_path):
    dataset = ogr.Open(reference_path)
    if dataset is None:
        raise SystemExit("Nie można otworzyć %s" % reference_path)
    counts = collections.Counter()
    for feature in dataset.GetLayer():
        counts[feature.GetField(B.ROI_CLASS_FIELD)] += 1
    dataset = None
    return counts

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wariant", default="1234567")
    parser.add_argument("--raster", default="L1C", choices=sorted(B.RASTERS))
    args = parser.parse_args()

    run_id = "%s_poprawny_%s_class" % (args.wariant, args.raster)
    run_dir = os.path.join(RESULTS_DIR, run_id)
    if not os.path.isdir(run_dir):
        raise SystemExit("Brak katalogu przebiegu %s" % run_dir)

    shares = scene_shares(run_dir)
    if not shares:
        raise SystemExit("Brak map poziomu bazowego w %s" % run_dir)
    counts = reference_counts(B.VARIANTS[args.wariant]["reference"])
    total_points = sum(counts.values())

    print("Przebieg %s, map poziomu bazowego: %d"
          % (run_id, len(next(iter(shares.values())))))
    print("Punktów referencyjnych: %d\n" % total_points)
    print("  %-20s %10s %14s | %7s %8s | %s"
          % ("klasa", "w scenie", "zakres", "punktów", "w próbie", "stosunek"))
    for class_id in sorted(counts):
        values = shares.get(class_id, [0.0])
        scene = st.median(values)
        sample = 100.0 * counts[class_id] / total_points
        if scene > 0 and sample > scene:
            relation = "%.0f razy więcej" % (sample / scene)
        elif scene > 0:
            relation = "%.1f razy mniej" % (scene / sample)
        else:
            relation = "brak w mapach"
        print("  %-20s %9.2f%% %13s | %7d %7.1f%% | %s"
              % (CLASS_NAMES.get(class_id, class_id), scene,
                 "%.1f-%.1f%%" % (min(values), max(values)),
                 counts[class_id], sample, relation))

if __name__ == "__main__":
    main()
