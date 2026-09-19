# -*- coding: utf-8 -*-
"""Sprawdza równoważność dwóch sposobów wyznaczania sygnatury spektralnej.

WEJŚCIE:      dane_testowe/ROI/warianty oraz katalog sygnatur 1456_from_SCP.scpx
WYJŚCIE:      zestawienie na konsoli
URUCHOMIENIE: python3 badania/weryfikacja/weryfikuj_metode_sygnatur.py   (Python QGIS-LTR)
"""

import os
import sys

import numpy as np
import remotior_sensus
from osgeo import ogr

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_ROOT, os.path.join(_ROOT, "badania", "wyniki")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from core import outlier_filters as filters
from core.outlier_catalog import warp_roi_stack

_BADANIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DANE_DIR", os.path.join(_BADANIA, "dane_testowe"))
RESULTS_ROOT = os.environ.get("WYNIKI_BADAN_DIR",
                        os.path.join(_BADANIA, "wyniki_badan"))
IMAGE = DATA_DIR + "/L1C/L1C_przyciety.tif"
SCPX = DATA_DIR + "/ROI/warianty/1456_from_SCP.scpx"

def main():
    rs = remotior_sensus.Session(n_processes=2, available_ram=4000)
    bandset_catalog = rs.bandset_catalog()
    bandset_catalog.create_bandset(paths=IMAGE, bandset_number=1)
    bandset = bandset_catalog.get(1)

    catalog = rs.spectral_signatures_catalog(bandset=bandset)
    catalog.load(file_path=SCPX)
    table = catalog.table
    print("Sygnatur w katalogu: %d" % len(table))

    ds = ogr.Open(catalog.geometry_file)
    layer = ds.GetLayer()
    srs = layer.GetSpatialRef()
    geometries = {}
    for feat in layer:
        geometries[feat.GetField("roi_id")] = feat.GetGeometryRef().Clone()
    ds = None
    print("Geometrii w pliku: %d\n" % len(geometries))

    ogr_driver = ogr.GetDriverByName("GPKG")
    source = filters.build_multiband_source(bandset.get_absolute_paths())

    max_mean_diff = 0.0
    max_rel_diff = 0.0
    matching_px = 0
    checked = 0
    examples = []
    try:
        for i in range(len(table)):
            sig_id = table["signature_id"][i]
            geom = geometries.get(sig_id)
            if geom is None:
                continue
            warped = warp_roi_stack(source, geom, srs, sig_id, ogr_driver)
            if warped is None:
                continue
            stack = warped[0]

            valid = np.all(~np.isnan(stack), axis=0)
            pixels = stack[:, valid]
            if pixels.shape[1] == 0:
                continue
            means = np.nanmean(pixels, axis=1)

            from_library = np.array(catalog.signatures[sig_id].value, dtype=float)
            if len(from_library) != len(means):
                print("  RÓŻNA LICZBA PASM dla %s: %d vs %d"
                      % (sig_id, len(from_library), len(means)))
                continue

            difference = np.abs(means - from_library)
            relative = difference / np.maximum(np.abs(from_library), 1e-9)
            max_mean_diff = max(max_mean_diff, float(difference.max()))
            max_rel_diff = max(max_rel_diff, float(relative.max()))
            if int(table["pixel_count"][i]) == int(pixels.shape[1]):
                matching_px += 1
            checked += 1
            if len(examples) < 3:
                examples.append(
                    (sig_id, int(table["pixel_count"][i]),
                     int(pixels.shape[1]), from_library[:3], means[:3])
                )
    finally:
        filters.release_multiband_source(source)

    print("=" * 70)
    print("Sprawdzonych sygnatur: %d" % checked)
    print("Zgodna liczba pikseli: %d z %d" % (matching_px, checked))
    print("Maksymalna różnica bezwzględna średniej: %.6f" % max_mean_diff)
    print("Maksymalna różnica względna:             %.3e" % max_rel_diff)
    print()
    print("Przykłady (pierwsze trzy pasma):")
    for sig_id, px_bib, px_bez, values_library, values_direct in examples:
        print("  %s  piksele: biblioteka=%d, bezpośrednia=%d" %
              (sig_id, px_bib, px_bez))
        print("     biblioteka:   %s" % np.array2string(values_library, precision=4))
        print("     bezpośrednia: %s" % np.array2string(values_direct, precision=4))

if __name__ == "__main__":
    main()
