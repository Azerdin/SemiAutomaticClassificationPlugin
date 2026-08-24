# -*- coding: utf-8 -*-
"""Wyznacza odległość Jeffriesa-Matusity między klasami zbioru uczącego.

WEJŚCIE:      dane_testowe/ROI/warianty oraz raster L1C
WYJŚCIE:      zestawienie na konsoli
URUCHOMIENIE: python3 badania/wyniki/oblicz_separowalnosc_roi.py [warstwa.shp]   (wymaga GDAL i numpy)
"""

import os
import itertools
import sys
import numpy as np
from osgeo import gdal, ogr

gdal.UseExceptions()
ogr.UseExceptions()

_BADANIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANE = os.environ.get("DANE_DIR", os.path.join(_BADANIA, "dane_testowe"))
WYNIKI = os.environ.get("WYNIKI_BADAN_DIR",
                        os.path.join(_BADANIA, "wyniki_badan"))
RASTER = DANE + "/L1C/L1C_przyciety.tif"
ROI = sys.argv[1] if len(sys.argv) > 1 else DANE + "/ROI/warianty/roi_1234567.shp"
CLASS_FIELD = "klasa"

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
         "B9", "B10", "B11", "B12"]
DROP = {0, 9, 10}
USE = [i for i in range(len(BANDS)) if i not in DROP]

CLASS_NAMES = {1: "pole zaorane", 2: "droga", 3: "plaża", 4: "las",
               5: "zabudowa", 6: "woda", 7: "pole z roślinnością"}
DOSKONALA = 1.90

def sample_roi_pixels():
    ds = gdal.Open(RASTER)
    gt, proj = ds.GetGeoTransform(), ds.GetProjection()
    xs, ys = ds.RasterXSize, ds.RasterYSize
    stack = ds.ReadAsArray().astype(np.float64)
    nodata = ds.GetRasterBand(1).GetNoDataValue()

    vds = ogr.Open(ROI)
    lyr = vds.GetLayer()
    srs = lyr.GetSpatialRef()
    classes = sorted({f.GetField(CLASS_FIELD) for f in lyr
                      if f.GetField(CLASS_FIELD) not in (None, 0)})
    lyr.ResetReading()

    by_class = {}
    mem_v = ogr.GetDriverByName("Memory")
    mem_r = gdal.GetDriverByName("MEM")
    for cls in classes:
        vsrc = mem_v.CreateDataSource("m")
        vlyr = vsrc.CreateLayer("c", srs=srs, geom_type=ogr.wkbMultiPolygon)
        vlyr.CreateField(ogr.FieldDefn("k", ogr.OFTInteger))
        defn = vlyr.GetLayerDefn()
        lyr.ResetReading()
        for feat in lyr:
            if feat.GetField(CLASS_FIELD) != cls:
                continue
            g = feat.GetGeometryRef()
            if g is None:
                continue
            nf = ogr.Feature(defn)
            nf.SetGeometry(g.Clone())
            vlyr.CreateFeature(nf)
        mds = mem_r.Create("", xs, ys, 1, gdal.GDT_Byte)
        mds.SetGeoTransform(gt)
        mds.SetProjection(proj)
        gdal.RasterizeLayer(mds, [1], vlyr, burn_values=[1])
        mask = mds.ReadAsArray().astype(bool)
        px = stack[:, mask].T
        if nodata is not None:
            px = px[~np.any(px == nodata, axis=1)]
        px = px[~np.any(~np.isfinite(px), axis=1)]
        by_class[int(cls)] = px[:, USE]
    return by_class

def bhattacharyya(mi, Ci, mj, Cj):
    S = (Ci + Cj) / 2.0
    dm = (mi - mj).reshape(-1, 1)
    term1 = 0.125 * float((dm.T @ np.linalg.inv(S) @ dm)[0, 0])
    _, logdetS = np.linalg.slogdet(S)
    _, logdetCi = np.linalg.slogdet(Ci)
    _, logdetCj = np.linalg.slogdet(Cj)
    term2 = 0.5 * (logdetS - 0.5 * (logdetCi + logdetCj))
    return term1 + term2

def jm(mi, Ci, mj, Cj):
    return 2.0 * (1.0 - np.exp(-bhattacharyya(mi, Ci, mj, Cj)))

def main():
    print("Pasma robocze (%d): %s" % (len(USE), ", ".join(BANDS[i] for i in USE)))
    data = sample_roi_pixels()
    classes = sorted(data)
    print("Pikseli na klasę: %s\n" % {c: len(data[c]) for c in classes})

    means = {c: data[c].mean(axis=0) for c in classes}
    covs = {c: np.cov(data[c], rowvar=False) for c in classes}

    pairs = []
    for i, j in itertools.combinations(classes, 2):
        v = jm(means[i], covs[i], means[j], covs[j])
        pairs.append((v, i, j))
    pairs.sort()

    print("Odległość Jeffriesa-Matusity dla wszystkich par klas (rosnąco):")
    for v, i, j in pairs:
        print("  %-22s -- %-22s  JM = %.2f"
              % (CLASS_NAMES[i], CLASS_NAMES[j], v))

    n_doskonala = sum(1 for v, _, _ in pairs if v > DOSKONALA)
    print("\nPar łącznie: %d, powyżej %.2f: %d, minimum: %.2f"
          % (len(pairs), DOSKONALA, n_doskonala, pairs[0][0]))

    M = np.vstack([means[c] for c in classes])
    correct = total = 0
    for c in classes:
        for row in data[c]:
            d = np.linalg.norm(M - row, axis=1)
            if classes[int(np.argmin(d))] == c:
                correct += 1
            total += 1
    print("Pikseli najbliżej centroidu własnej klasy: %d/%d = %.1f%%"
          % (correct, total, 100.0 * correct / total))

if __name__ == "__main__":
    main()
