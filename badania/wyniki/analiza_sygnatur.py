# -*- coding: utf-8 -*-
"""Wyznacza jednorodność obszarów treningowych oraz uśrednione sygnatury klas dla obu stanów zbioru uczącego.

WEJŚCIE:      dane_testowe/ROI oraz ROI_blad, raster L1C
WYJŚCIE:      wyniki_badan/agregacja_v2/cv_*.csv, spectral_signatures_bledy.csv oraz wyniki_badan/img/*.png
URUCHOMIENIE: python3 badania/wyniki/analiza_sygnatur.py   (wymaga GDAL, numpy i matplotlib)
"""

import csv
import os

import numpy as np
from osgeo import gdal, ogr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BADANIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DANE_DIR", os.path.join(_BADANIA, "dane_testowe"))
RESULTS_ROOT = os.environ.get("WYNIKI_BADAN_DIR",
                        os.path.join(_BADANIA, "wyniki_badan"))
RASTER = DATA_DIR + "/L1C/L1C_przyciety.tif"
SHP = DATA_DIR + "/ROI_blad/warianty/roi_1234567_bledy.shp"
SHP_CLEAN = DATA_DIR + "/ROI/warianty/roi_1234567.shp"
CLASS_FIELD = "klasa"

OUT_DIR = os.environ.get("AGREGACJA_DIR", RESULTS_ROOT + "/agregacja_v2")

IMG_DIR = os.path.join(RESULTS_ROOT, "img")

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
         "B9", "B10", "B11", "B12"]
WAVELENGTHS = [0.443, 0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865,
               0.945, 1.375, 1.61, 2.19]
CV_EXCLUDE = {9, 10}

CLASS_NAMES = {1: "pole zaorane", 2: "droga", 3: "plaża", 4: "las",
               5: "zabudowa", 6: "woda", 7: "pole z roślinnością"}

def sample_polygons(roi_path=SHP):
    ds = gdal.Open(RASTER)
    gt = ds.GetGeoTransform()
    bands = [ds.GetRasterBand(i + 1) for i in range(ds.RasterCount)]
    vds = ogr.Open(roi_path)
    lyr = vds.GetLayer()
    out = []
    for feat in lyr:
        cls = int(feat.GetField(CLASS_FIELD))
        g = feat.GetGeometryRef()
        if g is None:
            continue
        x0, x1, y0, y1 = g.GetEnvelope()
        c0 = int((x0 - gt[0]) / gt[1]); c1 = int((x1 - gt[0]) / gt[1])
        r0 = int((y1 - gt[3]) / gt[5]); r1 = int((y0 - gt[3]) / gt[5])
        vecs = []
        for r in range(min(r0, r1), max(r0, r1) + 1):
            for c in range(min(c0, c1), max(c0, c1) + 1):
                if not (0 <= c < ds.RasterXSize and 0 <= r < ds.RasterYSize):
                    continue
                px = gt[0] + (c + 0.5) * gt[1]
                py = gt[3] + (r + 0.5) * gt[5]
                pt = ogr.Geometry(ogr.wkbPoint); pt.AddPoint(px, py)
                if g.Contains(pt):
                    vecs.append([float(b.ReadAsArray(c, r, 1, 1)[0][0])
                                 for b in bands])
        a = np.array(vecs)
        out.append((cls, len(a), a.mean(axis=0), a.std(axis=0, ddof=0)))
    return out

def cv_max(means, stds):
    cvs = [(100.0 * stds[i] / means[i], i) for i in range(len(means))
           if i not in CV_EXCLUDE and means[i] != 0]
    val, idx = max(cvs)
    return val, BANDS[idx]

def _save(name, header, rows):
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

def gen_cv_csv(polys, out_name):
    by_class = {}
    for cls, n, means, stds in polys:
        by_class.setdefault(cls, []).append((n, means, stds))
    rows, mean_cv = [], {}
    for cls in sorted(by_class):
        cvals = []
        for i, (n, means, stds) in enumerate(by_class[cls], 1):
            v, band = cv_max(means, stds)
            cvals.append(v)
            rows.append([cls, CLASS_NAMES[cls], i, n, "%.12g" % v, band])
        mean_cv[cls] = float(np.mean(cvals))
        print("  klasa %d (%s): śr. CV_max = %.2f%%  (poligony: %d)"
              % (cls, CLASS_NAMES[cls], mean_cv[cls], len(cvals)))
    _save(out_name,
          ["klasa", "nazwa_klasy", "poligon", "liczba_pikseli",
           "cv_max_proc", "pasmo"], rows)
    return mean_cv

def gen_comparison_csv(clean_cv, mean_cv):
    _save("cv_porownanie.csv",
          ["klasa", "nazwa_klasy", "cv_max_poprawny_proc",
           "cv_max_zbledami_proc"],
          [[cls, CLASS_NAMES[cls], "%.12g" % clean_cv[cls],
            "%.12g" % mean_cv[cls]] for cls in range(1, 8)])

def class_mean_signatures(polys):
    acc = {}
    for cls, n, means, _stds in polys:
        w, s = acc.get(cls, (0.0, np.zeros(len(means))))
        acc[cls] = (w + n, s + n * means)
    return {cls: s / w for cls, (w, s) in acc.items()}

def gen_signature_csv(sig):
    _save("spectral_signatures_bledy.csv",
          ["pasmo", "dlugosc_fali_um"] + ["klasa_%d" % c for c in range(1, 8)],
          [[BANDS[bi], "%.12g" % WAVELENGTHS[bi]]
           + ["%.12g" % sig[c][bi] for c in range(1, 8)]
           for bi in range(len(BANDS))])

REFLECTANCE_SCALE = 10000.0
CLASS_COLOURS = {1: "tab:brown", 2: "tab:gray", 3: "gold", 4: "darkgreen",
                 5: "tab:red", 6: "tab:blue", 7: "yellowgreen"}

def gen_signature_figure(sig, file_name):
    os.makedirs(IMG_DIR, exist_ok=True)
    positions = list(range(len(BANDS)))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for class_id in range(1, 8):
        values = [v / REFLECTANCE_SCALE for v in sig[class_id]]
        ax.plot(positions, values, marker="o", ms=3,
                color=CLASS_COLOURS[class_id],
                label="%d %s" % (class_id,
                                 CLASS_NAMES[class_id]))
    ax.set_xticks(positions)
    ax.set_xticklabels(["%s\n%.3f" % (b, w) for b, w in zip(BANDS, WAVELENGTHS)],
                       fontsize=8)
    ax.set_xlabel("Pasmo Sentinel-2 (długość fali [µm])")
    ax.set_ylabel("Współczynnik odbicia TOA")
    ax.set_ylim(0.08, 0.58)
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, file_name), dpi=150)
    plt.close(fig)

def main():
    print("Próbkowanie ROI z błędami z rastra L1C ...")
    polys = sample_polygons(SHP)
    print("Obszarów treningowych: %d" % len(polys))
    print("Jednorodność (śr. CV_max per klasa):")
    mean_cv = gen_cv_csv(polys, "cv_bledy.csv")
    sig = class_mean_signatures(polys)
    gen_signature_csv(sig)
    gen_signature_figure(sig, "spectral_signatures_bledy.png")

    print("\nPróbkowanie ROI poprawnych z rastra L1C ...")
    clean = sample_polygons(SHP_CLEAN)
    print("Obszarów treningowych: %d" % len(clean))
    print("Jednorodność (śr. CV_max per klasa):")
    clean_cv = gen_cv_csv(clean, "cv_poprawny.csv")
    gen_signature_figure(class_mean_signatures(clean), "spectral_signatures.png")
    gen_comparison_csv(clean_cv, mean_cv)

    print("\nZestawienia CSV -> %s" % OUT_DIR)
    print("Wykresy -> %s/spectral_signatures.png oraz "
          "spectral_signatures_bledy.png" % IMG_DIR)

if __name__ == "__main__":
    main()
