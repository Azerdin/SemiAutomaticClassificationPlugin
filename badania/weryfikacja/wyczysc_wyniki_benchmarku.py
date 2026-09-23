# -*- coding: utf-8 -*-
"""Usuwa dane pośrednie przebiegów, pozostawiając rastry klasyfikacji oraz zestawienia.

WEJŚCIE:      katalog wyników przebiegów
WYJŚCIE:      ten sam katalog, po usunięciu plików pośrednich
URUCHOMIENIE: python3 badania/weryfikacja/wyczysc_wyniki_benchmarku.py <katalog>
"""

import argparse
import os
import shutil

KEEP_DIR = "classifications"

def human(nbytes):
    size = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return "%.1f %s" % (size, unit)
        size /= 1024.0

def dir_size(path):
    total = 0
    for dp, _dn, fn in os.walk(path):
        for f in fn:
            fp = os.path.join(dp, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total

def find_run_dirs(root, keep_name):
    runs = set()
    for dirpath, dirnames, _filenames in os.walk(root):
        if keep_name in dirnames:
            runs.add(dirpath)
            dirnames[:] = [d for d in dirnames if d != keep_name]
    return sorted(runs)

def has_rasters(keep_path):
    if not os.path.isdir(keep_path):
        return False
    return any(f.lower().endswith(".tif") for f in os.listdir(keep_path))

def clean_run(run_dir, keep_name, apply):
    keep_path = os.path.join(run_dir, keep_name)
    if not has_rasters(keep_path):
        print("  POMINIĘTO (brak rastrów w %s/): %s" % (keep_name, run_dir))
        return None, 0

    targets = [e for e in sorted(os.listdir(run_dir)) if e != keep_name]
    if not targets:
        print("  bez zmian (tylko rastry): %s" % run_dir)
        return 0, 0

    n_rasters = sum(1 for f in os.listdir(keep_path) if f.lower().endswith(".tif"))
    freed = 0
    print("  %s  (zostają rastry: %d)" % (run_dir, n_rasters))
    for entry in targets:
        path = os.path.join(run_dir, entry)
        size = dir_size(path) if os.path.isdir(path) else os.path.getsize(path)
        freed += size
        kind = "kat " if os.path.isdir(path) else "plik"
        print("      %s %-30s %s%s"
              % ("USUŃ" if apply else "usunąłbym", entry + ("/" if os.path.isdir(path) else ""),
                 human(size), "" if apply else ""))
        if apply:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    return len(targets), freed

def main():
    parser = argparse.ArgumentParser(
        description="Usuwa z wyników benchmarku pliki oceny dokładności, "
        "zostawiając tylko rastry klasyfikacji (folder classifications/).")
    parser.add_argument(
        "root", metavar="KATALOG_WYNIKOW",
        help="Katalog przebiegu benchmarku lub katalog nadrzędny z wieloma "
        "przebiegami.")
    parser.add_argument(
        "--apply", action="store_true",
        help="Faktycznie usuń pliki. Bez tej flagi tylko podgląd (dry-run).")
    parser.add_argument(
        "--keep-dir", default=KEEP_DIR,
        help="Nazwa folderu z rastrami do zachowania (domyślnie: %s)." % KEEP_DIR)
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit("BŁĄD: katalog nie istnieje: %s" % args.root)

    runs = find_run_dirs(args.root, args.keep_dir)
    if not runs:
        raise SystemExit(
            "Nie znaleziono żadnego przebiegu (folderu z podkatalogiem '%s/') "
            "w: %s" % (args.keep_dir, args.root))

    mode = "USUWANIE (--apply)" if args.apply else "PODGLĄD (dry-run — nic nie usuwam)"
    print("Tryb: %s" % mode)
    print("Znaleziono przebiegów: %d\n" % len(runs))

    total_items = total_freed = 0
    cleaned = skipped = 0
    for run in runs:
        items, freed = clean_run(run, args.keep_dir, args.apply)
        if items is None:
            skipped += 1
        else:
            cleaned += 1
            total_items += items
            total_freed += freed

    print("\n" + "=" * 60)
    print("Przebiegi wyczyszczone: %d,  pominięte (brak rastrów): %d" % (cleaned, skipped))
    print("%s: %d pozycji, %s"
          % ("Usunięto" if args.apply else "Do usunięcia", total_items, human(total_freed)))
    if not args.apply and total_items:
        print("\nTo był tylko podgląd. Aby faktycznie usunąć, dodaj flagę --apply.")

if __name__ == "__main__":
    main()
