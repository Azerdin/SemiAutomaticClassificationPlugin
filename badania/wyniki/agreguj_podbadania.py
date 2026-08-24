# -*- coding: utf-8 -*-
"""Agreguje wyniki badań uzupełniających do zestawień w układzie przestawnym.

WEJŚCIE:      wyniki_badan/podbadania_v2/*/csv_per_algorytm/wyniki_<algorytm>.csv
WYJŚCIE:      wyniki_badan/agregacja_v2/podbadania_*.csv
URUCHOMIENIE: python3 badania/wyniki/agreguj_podbadania.py
"""

import csv
import os

_BADANIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANE = os.environ.get("DANE_DIR", os.path.join(_BADANIA, "dane_testowe"))
WYNIKI = os.environ.get("WYNIKI_BADAN_DIR",
                        os.path.join(_BADANIA, "wyniki_badan"))
DATA = os.environ.get("PODBADANIA_DIR", WYNIKI + "/podbadania_v2")
OUT = os.environ.get("AGREGACJA_DIR", WYNIKI + "/agregacja_v2")

def _load(run_dir, algo_file):
    base = os.path.join(DATA, run_dir)
    if not os.path.isdir(base):
        return {}
    subdirs = [d for d in os.listdir(base)
               if os.path.isdir(os.path.join(base, d)) and "L1C_class" in d]
    if not subdirs:
        return {}
    path = os.path.join(base, subdirs[0], "csv_per_algorytm", algo_file)
    if not os.path.exists(path):
        return {}
    out = {}
    for row in csv.DictReader(open(path, encoding="utf-8-sig")):
        try:
            out[row["Metoda"]] = float(row["DA_%"])
        except (KeyError, ValueError):
            pass
    return out

def _pivot(loaders):
    configs = sorted(set().union(*[d.keys() for d in loaders]) if loaders else [])
    rows = []
    for c in configs:
        vals = [d.get(c) for d in loaders]
        present = [v for v in vals if v is not None]
        best = max(present) if present else -1
        rows.append((best, c, vals))
    rows.sort(key=lambda x: -x[0])
    return [(c, *vals) for (best, c, vals) in rows]

def _write_csv(fname, header, rows):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, fname)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r[0]] + ["" if v is None else "%.4f" % v for v in r[1:]])
    print("  zapisano %s (%d wierszy)" % (fname, len(rows)))

def gen_gridsearch():
    algos = ["spectral_angle_mapping", "support_vector_machine",
             "multi-layer_perceptron"]
    cols = ["spectral angle mapping", "support vector machine",
            "multi-layer perceptron"]
    loaders = [_load("gridsearch_zbledami", "wyniki_%s.csv" % a) for a in algos]
    _write_csv("podbadania_gridsearch.csv", ["Konfiguracja"] + cols,
               _pivot(loaders))

def gen_potoki(state, fname):
    algos = ["maximum_likelihood", "random_forest", "minimum_distance",
             "spectral_angle_mapping", "support_vector_machine",
             "multi-layer_perceptron"]
    cols = ["maximum likelihood", "random forest", "minimum distance",
            "spectral angle mapping", "support vector machine",
            "multi-layer perceptron"]
    loaders = [_load("maxk2_%s" % state, "wyniki_%s.csv" % a) for a in algos]
    _write_csv(fname, ["Konfiguracja"] + cols, _pivot(loaders))

def gen_rf():
    loaders = [_load("rf_strojony_poprawny", "wyniki_random_forest.csv"),
               _load("rf_strojony_zbledami", "wyniki_random_forest.csv")]
    _write_csv("podbadania_rf.csv", ["Konfiguracja", "poprawny", "zbledami"],
               _pivot(loaders))

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    gen_gridsearch()
    gen_potoki("poprawny", "podbadania_potoki_poprawny.csv")
    gen_potoki("zbledami", "podbadania_potoki_zbledami.csv")
    gen_rf()
    print("Gotowe (CSV w %s)." % OUT)
