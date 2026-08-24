# -*- coding: utf-8 -*-
"""Przelicza macierz eksperymentu dla lasu losowego w strojonej konfiguracji.

WEJŚCIE:      konfiguracja z bloku CONFIG w benchmark_outlier_removal.py
WYJŚCIE:      wyniki_badan/rf_poprawiony/<przebieg>/
URUCHOMIENIE: python3 badania/wyniki/uruchom_rf_poprawiony.py   (Python QGIS-LTR)
"""

import sys
from types import SimpleNamespace

import benchmark_outlier_removal as B

OUT_DIR = B.DATA_DIR + "/rf_poprawiony"
ALGORITHM = "random forest"
NUMBER_TREES = 500
MAX_FEATURES = "sqrt"

def main():
    B.OUTPUT_DIR = OUT_DIR
    B.ALGORITHMS = [ALGORITHM]
    B.RF_NUMBER_TREES = NUMBER_TREES
    B.RF_MAX_FEATURES = MAX_FEATURES
    B.KEEP_CLASSIFICATIONS = True
    B.SKIP_DONE = True

    print("=" * 82)
    print("PRZELICZENIE LASU LOSOWEGO W POPRAWIONEJ KONFIGURACJI")
    print("=" * 82)
    print("  algorytm:        %s" % ALGORITHM)
    print("  liczba drzew:    %d (domyślnie we wtyczce %d)"
          % (NUMBER_TREES, 10))
    print("  cechy na podział: %s (domyślnie we wtyczce wszystkie pasma)"
          % MAX_FEATURES)
    print("  katalog wyjściowy: %s" % OUT_DIR)
    print("  rastry klasyfikacji: zachowywane (około 5 GB)")
    print()

    args = SimpleNamespace(
        only=None,
        dry_run="--dry-run" in sys.argv,
        force="--force" in sys.argv,
    )
    B._run_matrix(args)

if __name__ == "__main__":
    main()
