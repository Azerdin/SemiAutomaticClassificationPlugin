# -*- coding: utf-8 -*-
"""Wytwarza osiem wariantów zestawu klas z pełnej warstwy obszarów treningowych.

WEJŚCIE:      dane_testowe/ROI/roi_1234567_duzeSHP.shp
WYJŚCIE:      dane_testowe/ROI/warianty/roi_<wariant>.shp
URUCHOMIENIE: python3 badania/dane/generuj_warianty_roi.py   (wymaga GDAL/OGR)
"""

import argparse
import itertools
import os
import sys

_qgis = "/Applications/QGIS-LTR.app/Contents"
if sys.executable.startswith(_qgis) and os.path.isdir(_qgis):
    os.environ.setdefault("PROJ_LIB", _qgis + "/Resources/proj")
    os.environ.setdefault("GDAL_DATA", _qgis + "/Resources/gdal")

from osgeo import ogr

ogr.UseExceptions()

_BADANIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANE = os.environ.get("DANE_DIR", os.path.join(_BADANIA, "dane_testowe"))
WYNIKI = os.environ.get("WYNIKI_BADAN_DIR",
                        os.path.join(_BADANIA, "wyniki_badan"))

SRC = DANE + "/ROI/roi_1234567_duzeSHP.shp"
OUT_DIR = DANE + "/ROI/warianty"
PREFIX = "roi_"
SUFFIX = ""
CLASS_FIELD = "class_id"
COPY_FIELDS = ["roi_id", "macroclass", "class_id", "c_name"]

OUT_FIELD = "klasa"
BASE_DIGITS = "1456"
OPTIONAL_DIGITS = ["2", "3", "7"]

def all_variants():
    base = set(BASE_DIGITS)
    out = []
    for bits in itertools.product((False, True), repeat=len(OPTIONAL_DIGITS)):
        digits = set(base)
        for present, d in zip(bits, OPTIONAL_DIGITS):
            if present:
                digits.add(d)
        out.append("".join(sorted(digits)))
    return sorted(out)

def merge_class(c, variant):
    if c == 3 and "3" not in variant:
        return None
    if c == 7 and "7" not in variant:
        return 1
    if c == 2 and "2" not in variant:
        return 5
    return c

def split_to_variants(src_path, out_dir, prefix, suffix, class_field, copy_fields):
    src_ds = ogr.Open(src_path)
    if src_ds is None:
        raise SystemExit("BŁĄD: nie mogę otworzyć źródła: %s" % src_path)
    src_layer = src_ds.GetLayer()
    src_defn = src_layer.GetLayerDefn()
    srs = src_layer.GetSpatialRef()
    geom_type = src_layer.GetGeomType()

    if src_layer.FindFieldIndex(class_field, 1) < 0:
        raise SystemExit("BŁĄD: brak pola klasy '%s' w %s" % (class_field, src_path))

    copy_defns = []
    for name in copy_fields:
        idx = src_defn.GetFieldIndex(name)
        if idx < 0:
            raise SystemExit("BŁĄD: brak pola '%s' w %s" % (name, src_path))
        fd = src_defn.GetFieldDefn(idx)
        new_fd = ogr.FieldDefn(fd.GetName(), fd.GetType())
        new_fd.SetWidth(fd.GetWidth())
        new_fd.SetPrecision(fd.GetPrecision())
        copy_defns.append(new_fd)

    os.makedirs(out_dir, exist_ok=True)
    driver = ogr.GetDriverByName("ESRI Shapefile")

    for variant in all_variants():
        name = "%s%s%s" % (prefix, variant, suffix)
        out_path = os.path.join(out_dir, "%s.shp" % name)
        if os.path.exists(out_path):
            driver.DeleteDataSource(out_path)
        out_ds = driver.CreateDataSource(out_path)
        out_layer = out_ds.CreateLayer(name, srs, geom_type)
        for d in copy_defns:
            out_layer.CreateField(d)
        out_layer.CreateField(ogr.FieldDefn(OUT_FIELD, ogr.OFTInteger))
        out_defn = out_layer.GetLayerDefn()

        written = dropped = 0
        src_layer.ResetReading()
        for feat in src_layer:
            class_id = merge_class(int(feat.GetField(class_field)), variant)
            if class_id is None:
                dropped += 1
                continue
            out_feat = ogr.Feature(out_defn)
            geom = feat.GetGeometryRef()
            if geom is not None:
                out_feat.SetGeometry(geom.Clone())
            for fname in copy_fields:
                out_feat.SetField(fname, feat.GetField(fname))
            out_feat.SetField(OUT_FIELD, class_id)
            out_layer.CreateFeature(out_feat)
            out_feat = None
            written += 1

        out_ds = None
        print("  %-22s zapisano %4d  (usunięto plażę: %d)"
              % (name, written, dropped))

    src_ds = None

def main():
    parser = argparse.ArgumentParser(
        description="Generuje 8 wariantów zestawu klas z warstwy ROI (poprawnej).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Tylko wypisz plan i sprawdź istnienie źródła, nic nie zapisuj.")
    args = parser.parse_args()

    variants = all_variants()
    print("Warianty (%d): %s\n" % (len(variants), ", ".join(variants)))

    if args.dry_run:
        ok = os.path.exists(SRC)
        print("ROI: źródło %s -> %s | pole klasy: %s | %s"
              % (SRC, OUT_DIR, CLASS_FIELD, "OK" if ok else "BRAK PLIKU"))
        print("\n--dry-run: nic nie zapisano.")
        return

    print("=== ROI: %s -> %s ===" % (os.path.basename(SRC), OUT_DIR))
    split_to_variants(SRC, OUT_DIR, PREFIX, SUFFIX, CLASS_FIELD, COPY_FIELDS)

if __name__ == "__main__":
    main()
