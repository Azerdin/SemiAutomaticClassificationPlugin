# -*- coding: utf-8 -*-
"""Zapisuje katalogi sygnatur .scpx dla każdej pary obszarów treningowych i obrazu.

WEJŚCIE:      dane_testowe/ROI oraz ROI_blad, rastry L1C i L2A
WYJŚCIE:      dane_testowe/scpx/roi_<wariant>_<stan>_<raster>.scpx
URUCHOMIENIE: python3 badania/dane/generuj_scpx_wariantow.py   (Python QGIS-LTR)
"""

import argparse
import os
import sys

import remotior_sensus

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_ROOT, os.path.join(_ROOT, "badania", "wyniki")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import benchmark_outlier_removal as B

OUT_DIR = B.DATA_DIR + "/scpx"

def _argumenty():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", choices=sorted(B.RASTERS), default=None,
                        help="tylko wskazany raster (domyślnie wszystkie)")
    parser.add_argument("--stan", choices=["poprawny", "zbledami"], default=None,
                        help="tylko wskazany stan zbioru uczącego")
    parser.add_argument("--wariant", default=None,
                        help="tylko wskazany wariant, np. 1456")
    parser.add_argument("--force", action="store_true",
                        help="nadpisz istniejące pliki")
    return parser.parse_args()

def _generate(rs, roi_path, image_path, out_path):
    bandset_catalog = rs.bandset_catalog()
    bandset_catalog.create_bandset(paths=image_path, bandset_number=1)
    B._set_wavelengths(bandset_catalog)
    bandset = bandset_catalog.get(1)

    catalog = rs.spectral_signatures_catalog(bandset=bandset)
    catalog.import_vector(
        file_path=roi_path,
        macroclass_field=B.ROI_MACROCLASS_FIELD,
        class_field=B.ROI_CLASS_FIELD,
        macroclass_name_field=B.ROI_CLASS_NAME_FIELD,
        class_name_field=B.ROI_CLASS_NAME_FIELD,
        calculate_signature=True,
    )
    if catalog.table is None or len(catalog.table) == 0:
        return None
    catalog.save(out_path)
    return len(catalog.table)

def main():
    args = _argumenty()
    os.makedirs(OUT_DIR, exist_ok=True)

    rasters = [args.raster] if args.raster else sorted(B.RASTERS)
    states = [args.stan] if args.stan else ["poprawny", "zbledami"]
    variants = [args.wariant] if args.wariant else sorted(B.VARIANTS)

    print("=" * 78)
    print("GENEROWANIE KATALOGÓW SYGNATUR")
    print("=" * 78)
    print("  warianty: %s" % ", ".join(variants))
    print("  stany:    %s" % ", ".join(states))
    print("  rastry:   %s" % ", ".join(rasters))
    print("  wyjście:  %s\n" % OUT_DIR)

    rs = remotior_sensus.Session(
        n_processes=B.N_PROCESSES, available_ram=B.RAM_MB
    )

    n_ok = n_error = n_pominieto = 0
    for variant in variants:
        vcfg = B.VARIANTS.get(variant)
        if vcfg is None:
            print("  POMINIĘTO wariant %s (brak w konfiguracji)" % variant)
            continue
        for state in states:
            roi_path = vcfg["roi"].get(state)
            if not roi_path or not os.path.exists(roi_path):
                print("  POMINIĘTO %s/%s (brak pliku ROI)" % (variant, state))
                n_pominieto += 1
                continue
            for raster in rasters:
                image_path = B.RASTERS[raster]
                name = "roi_%s_%s_%s.scpx" % (variant, state, raster)
                out_path = os.path.join(OUT_DIR, name)
                if os.path.exists(out_path) and not args.force:
                    print("  POMINIĘTO %s (już istnieje)" % name)
                    n_pominieto += 1
                    continue
                try:
                    count = _generate(rs, roi_path, image_path, out_path)
                except Exception as err:
                    print("  BŁĄD %s: %s" % (name, err))
                    n_error += 1
                    continue
                if count is None:
                    print("  BŁĄD %s: import nie dał sygnatur" % name)
                    n_error += 1
                else:
                    print("  OK   %-38s %3d sygnatur" % (name, count))
                    n_ok += 1

    print("\n" + "=" * 78)
    print("KONIEC: utworzono %d, błędów %d, pominięto %d"
          % (n_ok, n_error, n_pominieto))
    print("Pliki w: %s" % OUT_DIR)

if __name__ == "__main__":
    main()
