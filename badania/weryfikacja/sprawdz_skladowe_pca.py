# -*- coding: utf-8 -*-
"""Wyznacza liczbę głównych składowych odpowiadającą kolejnym progom udziału wariancji, osobno dla każdej klasy.

WEJŚCIE:      warstwa obszarów treningowych wariantu oraz obraz wielospektralny
WYJŚCIE:      zestawienie na konsoli
URUCHOMIENIE: python3 badania/weryfikacja/sprawdz_skladowe_pca.py   (Python QGIS-LTR)
"""

import argparse
import os
import sys

import numpy as np
from osgeo import gdal, ogr
from sklearn.decomposition import PCA

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_ROOT, os.path.join(_ROOT, "badania", "wyniki")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import benchmark_outlier_removal as B
from core.outlier_filters import scale_data

THRESHOLDS = (0.70, 0.90, 0.95, 0.99)

def class_pixels(roi_path, image_path):
    raster = gdal.Open(image_path)
    geotransform = raster.GetGeoTransform()
    bands = [raster.GetRasterBand(i + 1).ReadAsArray()
             for i in range(raster.RasterCount)]

    dataset = ogr.Open(roi_path)
    if dataset is None:
        raise SystemExit("Nie można otworzyć %s" % roi_path)
    layer = dataset.GetLayer()

    pools = {}
    for feature in layer:
        class_id = feature.GetField(B.ROI_CLASS_FIELD)
        x_min, x_max, y_min, y_max = feature.GetGeometryRef().GetEnvelope()
        col_start = int((x_min - geotransform[0]) / geotransform[1])
        col_end = int((x_max - geotransform[0]) / geotransform[1]) + 1
        row_start = int((y_max - geotransform[3]) / geotransform[5])
        row_end = int((y_min - geotransform[3]) / geotransform[5]) + 1
        for row in range(max(row_start, 0), min(row_end, raster.RasterYSize)):
            for col in range(max(col_start, 0), min(col_end, raster.RasterXSize)):
                pools.setdefault(class_id, []).append(
                    [band[row, col] for band in bands]
                )
    return {k: np.array(v, dtype=float) for k, v in pools.items()}, raster.RasterCount

def components_for(pixels, threshold):
    scaled, _ = scale_data(pixels)
    ratios = PCA().fit(scaled).explained_variance_ratio_
    return int(np.searchsorted(np.cumsum(ratios), threshold) + 1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wariant", default="1234567")
    parser.add_argument("--raster", default="L1C", choices=sorted(B.RASTERS))
    parser.add_argument("--stan", default="poprawny",
                        choices=("poprawny", "zbledami"))
    args = parser.parse_args()

    roi_path = B.VARIANTS[args.wariant]["roi"][args.stan]
    image_path = B.RASTERS[args.raster]
    pools, band_count = class_pixels(roi_path, image_path)

    print("Wariant %s, zbiór %s, obraz %s, kanałów %d"
          % (args.wariant, args.stan, args.raster, band_count))
    print("Suma wariancji po standaryzacji równa jest liczbie kanałów, "
          "czyli %d.\n" % band_count)
    header = "  ".join("%.2f" % t for t in THRESHOLDS)
    print("  %-8s %9s   %s" % ("klasa", "pikseli", header))
    for class_id in sorted(pools):
        pixels = pools[class_id]
        if len(pixels) <= band_count:
            print("  %-8s %9d   pominięto, za mało pikseli" % (class_id, len(pixels)))
            continue
        counts = [components_for(pixels, t) for t in THRESHOLDS]
        print("  %-8s %9d   %s"
              % (class_id, len(pixels), "     ".join("%d" % c for c in counts)))

if __name__ == "__main__":
    main()
