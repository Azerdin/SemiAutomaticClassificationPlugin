# -*- coding: utf-8 -*-
"""Wyznacza dokładność klasyfikacji sprzed naprawy błędu wtyczki SCP.

WEJŚCIE:      rastry klasyfikacji sprzed naprawy oraz warstwa referencyjna wariantu 1456
WYJŚCIE:      zestawienie na konsoli
URUCHOMIENIE: python3 badania/wyniki/oblicz_oa_przed_poprawka.py   (wymaga GDAL i numpy)
"""

import os
import numpy as np
from osgeo import gdal, ogr

gdal.UseExceptions()
ogr.UseExceptions()

_BADANIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANE = os.environ.get("DANE_DIR", os.path.join(_BADANIA, "dane_testowe"))
WYNIKI = os.environ.get("WYNIKI_BADAN_DIR",
                        os.path.join(_BADANIA, "wyniki_badan"))
REF = DANE + "/referencja2/warianty/referencja_1456.shp"
CLASS_FIELD = "klasa"
BEFORE_DIR = WYNIKI + "/klasyfikacja przed poprawka"

AFTER_SVM = (WYNIKI + "/benchmark_results_v2/1456_poprawny_L1C_class/"
             "classifications/Baseline__support_vector_machine.tif")

BEFORE = [
    ("maszyny wektorów nośnych", "SVM_blad.tif"),
    ("perceptronu wielowarstwowego", "MLP_blad.tif"),
    ("mapowania kąta spektralnego", "SAM_blad.tif"),
    ("minimalnej odległości", "MD_blad.tif"),
    ("lasu losowego", "RF_blad.tif"),
]

def _ref_points():
    vds = ogr.Open(REF)
    lyr = vds.GetLayer()
    pts = []
    for feat in lyr:
        cls = feat.GetField(CLASS_FIELD)
        if cls in (None, 0):
            continue
        g = feat.GetGeometryRef()
        if g is None:
            continue
        c = g.Centroid() if g.GetGeometryName() != "POINT" else g
        pts.append((c.GetX(), c.GetY(), int(cls)))
    return pts

def oa(raster_path, pts):
    ds = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    xs, ys = ds.RasterXSize, ds.RasterYSize
    correct = total = 0
    for x, y, cls in pts:
        col = int((x - gt[0]) / gt[1])
        row = int((y - gt[3]) / gt[5])
        if not (0 <= col < xs and 0 <= row < ys):
            continue
        total += 1
        if int(arr[row, col]) == cls:
            correct += 1
    return 100.0 * correct / total if total else 0.0, correct, total

def main():
    pts = _ref_points()
    print("Punktów referencyjnych (wariant 1456): %d\n" % len(pts))

    if os.path.exists(AFTER_SVM):
        v, c, t = oa(AFTER_SVM, pts)
        print("WALIDACJA (SVM po naprawie): OA = %.2f%%  (%d/%d) "
              "-- oczekiwane ~94,94\n" % (v, c, t))
    else:
        print("WALIDACJA pominięta (brak rastra po naprawie)\n")

    print("=== OA PRZED naprawą ===")
    print("%-30s %8s  %s" % ("Algorytm", "OA [%]", "poprawne/wszystkie"))
    for name, fname in BEFORE:
        path = os.path.join(BEFORE_DIR, fname)
        if not os.path.exists(path):
            print("%-30s   BRAK PLIKU (%s)" % (name, fname))
            continue
        v, c, t = oa(path, pts)
        print("%-30s %7.2f  %d/%d" % (name, v, c, t))
    print("\nAlgorytm maksymalnego prawdopodobieństwa: przed naprawą błąd "
          "wykonania (brak rastra).")

if __name__ == "__main__":
    main()
