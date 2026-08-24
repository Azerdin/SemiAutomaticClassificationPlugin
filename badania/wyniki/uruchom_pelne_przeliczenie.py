# -*- coding: utf-8 -*-
"""Sterownik trybu macierzy: ustawia katalog wyjściowy oraz przełączniki, których nie przyjmuje wiersz poleceń.

WEJŚCIE:      konfiguracja z bloku CONFIG w benchmark_outlier_removal.py
WYJŚCIE:      wyniki_badan/benchmark_results_v2/<przebieg>/
URUCHOMIENIE: wywoływany przez uruchom_pelne_przeliczenie.sh   (Python QGIS-LTR)
"""

import sys
import time
from types import SimpleNamespace

import benchmark_outlier_removal as B

OUTPUT_NAME = "benchmark_results_v2"

def main():
    B.OUTPUT_DIR = B.DATA_DIR + "/" + OUTPUT_NAME
    B.KEEP_CLASSIFICATIONS = True
    B.SKIP_DONE = True

    print("=" * 82)
    print("PEŁNE PRZELICZENIE MACIERZY")
    print("=" * 82)
    print("  katalog wyjściowy: %s" % B.OUTPUT_DIR)
    print("  rastry klasyfikacji:  zachowywane")
    print("  wznawianie:           gotowe przebiegi pomijane")
    print("  start:                %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print()

    args = SimpleNamespace(
        only=None,
        dry_run="--dry-run" in sys.argv,
        force="--force" in sys.argv,
    )
    if "--only" in sys.argv:
        args.only = sys.argv[sys.argv.index("--only") + 1]

    t0 = time.time()
    B._run_matrix(args)
    if not args.dry_run:
        print("\n  koniec: %s (czas łączny %.1f h)"
              % (time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - t0) / 3600.0))

if __name__ == "__main__":
    main()
