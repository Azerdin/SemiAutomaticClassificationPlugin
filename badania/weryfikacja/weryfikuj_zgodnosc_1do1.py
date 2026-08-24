# -*- coding: utf-8 -*-
"""Porównuje sygnatury wyznaczone przez wtyczkę SCP oraz przez skrypt badawczy.

WEJŚCIE:      katalog sygnatur wtyczki oraz obszary treningowe wariantu
WYJŚCIE:      zestawienie na konsoli
URUCHOMIENIE: python3 badania/weryfikacja/weryfikuj_zgodnosc_1do1.py   (Python QGIS-LTR)
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

PARAMS = {
    "MAD": {"threshold": 3.5},
    "IQR": {"factor": 1.5},
    "Percentile": {"lower_pct": 0.01, "upper_pct": 0.99},
    "Robust Mahalanobis": {"alpha": 0.025},
    "Mahalanobis": {"alpha": 0.025},
}

def _load(rs, path, bandset):
    catalog = rs.spectral_signatures_catalog(bandset=bandset)
    catalog.load(file_path=path)
    return catalog

def _signatures(catalog):
    out = {}
    table = catalog.table
    for i in range(len(table)):
        sig_id = table["signature_id"][i]
        out[sig_id] = (
            int(table["class_id"][i]),
            int(table["pixel_count"][i]),
            [round(float(v), 4) for v in catalog.signatures[sig_id].value],
        )
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zrodlo", help="katalog wejściowy .scpx (przed czyszczeniem)")
    parser.add_argument("wynik_wtyczki", help="katalog oczyszczony we wtyczce SCP")
    parser.add_argument("--metoda", default="MAD")
    parser.add_argument("--zakres", default="class", choices=["roi", "class"])
    parser.add_argument("--raster", default="L1C", choices=sorted(B.RASTERS))
    args = parser.parse_args()

    rs = remotior_sensus.Session(
        n_processes=B.N_PROCESSES, available_ram=B.RAM_MB
    )
    bandset_catalog = rs.bandset_catalog()
    bandset_catalog.create_bandset(
        paths=B.RASTERS[args.raster], bandset_number=1
    )
    B._set_wavelengths(bandset_catalog)
    bandset = bandset_catalog.get(1)

    steps = [(args.metoda, PARAMS.get(args.metoda, {}))]
    script_dir, removed = B._build_cleaned_catalog(
        rs, args.zrodlo, bandset, steps, False, 1, scope=args.zakres
    )
    skrypt = _signatures(script_dir)
    plugin = _signatures(_load(rs, args.wynik_wtyczki, bandset))

    print("\n" + "=" * 74)
    print("PORÓWNANIE: metoda %s, zakres %s, raster %s"
          % (args.metoda, args.zakres, args.raster))
    print("=" * 74)
    print("  sygnatur po czyszczeniu:  wtyczka %d, skrypt %d"
          % (len(plugin), len(skrypt)))
    print("  usuniętych pikseli (skrypt): %d" % removed)

    common = set(plugin) & set(skrypt)
    print("  sygnatur wspólnych (po identyfikatorze): %d" % len(common))
    tylko_w = set(plugin) - set(skrypt)
    tylko_s = set(skrypt) - set(plugin)
    if tylko_w:
        print("  tylko we wtyczce: %d" % len(tylko_w))
    if tylko_s:
        print("  tylko w skrypcie: %d" % len(tylko_s))

    identical = rozne_px = rozne_wart = 0
    examples = []
    for sig_id in sorted(common):
        kw, pw, ww = plugin[sig_id]
        ks, ps, ws = skrypt[sig_id]
        difference = max((abs(a - b) for a, b in zip(ww, ws)), default=0.0)
        if pw == ps and difference < 1e-9:
            identical += 1
        else:
            if pw != ps:
                rozne_px += 1
            if difference >= 1e-9:
                rozne_wart += 1
            if len(examples) < 8:
                examples.append((sig_id, kw, pw, ps, difference))

    print("\n  IDENTYCZNE (piksele i wartości): %d z %d" % (identical, len(common)))
    print("  różna liczba pikseli:            %d" % rozne_px)
    print("  różne wartości sygnatury:        %d" % rozne_wart)
    if examples:
        print("\n  klasa | piksele wtyczka/skrypt | maks. różnica wartości")
        for sig_id, class_id, pw, ps, difference in examples:
            print("   %4d  | %6d / %-6d        | %.6f" % (class_id, pw, ps, difference))

    if identical == len(common) and not tylko_w and not tylko_s:
        print("\n  WYNIK: pełna zgodność 1:1")
    else:
        print("\n  WYNIK: wykryto różnice (szczegóły powyżej)")

if __name__ == "__main__":
    main()
