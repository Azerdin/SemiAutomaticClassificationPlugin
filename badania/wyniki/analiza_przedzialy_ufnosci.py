# -*- coding: utf-8 -*-
"""Wyznacza przedziały ufności dokładności oraz jej przyrostu, a także średnią makro miary F1.

WEJŚCIE:      wyniki_badan/benchmark_results_v2
WYJŚCIE:      wyniki_badan/agregacja_v2/przedzialy_ufnosci.csv oraz podsumowanie_ci.txt
URUCHOMIENIE: python3 badania/wyniki/analiza_przedzialy_ufnosci.py   (wymaga numpy)
"""

import argparse
import csv
import glob
import os
from math import sqrt

_BADANIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANE = os.environ.get("DANE_DIR", os.path.join(_BADANIA, "dane_testowe"))
WYNIKI = os.environ.get("WYNIKI_BADAN_DIR",
                        os.path.join(_BADANIA, "wyniki_badan"))
RESULTS_DIR = os.environ.get("WYNIKI_DIR", WYNIKI + "/benchmark_results_v2")
OUT_DIR = os.environ.get("AGREGACJA_DIR", WYNIKI + "/agregacja_v2")

Z = 1.959963984540054

ALGORITHMS = [
    "maximum_likelihood", "random_forest", "multi-layer_perceptron",
    "support_vector_machine", "spectral_angle_mapping", "minimum_distance",
]
DISPLAY_NAME = {
    "maximum_likelihood": "maximum likelihood",
    "random_forest": "random forest",
    "multi-layer_perceptron": "multi-layer perceptron",
    "support_vector_machine": "support vector machine",
    "spectral_angle_mapping": "spectral angle mapping",
    "minimum_distance": "minimum distance",
}

def wilson(trafienia, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = trafienia / n
    denominator = 1.0 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denominator
    half = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denominator
    return p, max(0.0, centre - half), min(1.0, centre + half)

def paired_ci(b, c, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    difference = (c - b) / n
    variance = (b + c - (c - b) ** 2 / n) / (n * n)
    half = Z * sqrt(max(variance, 0.0))
    return difference, difference - half, difference + half

def _load_predictions(accuracy_dir):
    out = {}
    for path in glob.glob(os.path.join(accuracy_dir, "*_pred.csv")):
        name = os.path.basename(path)[:-len("_pred.csv")]
        algorithm = None
        for candidate in sorted(ALGORITHMS, key=len, reverse=True):
            if name.endswith("__" + candidate):
                algorithm = candidate
                method = name[:-(len(candidate) + 2)]
                break
        if algorithm is None:
            continue
        with open(path, encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        trafienia = [int(w["mapa"]) == int(w["referencja"]) for w in rows]
        out[(method, algorithm)] = trafienia
    return out

def _f1_makro(matrix_path):
    if not os.path.exists(matrix_path):
        return None
    rows = list(csv.reader(open(matrix_path, encoding="utf-8-sig")))
    if not rows:
        return None
    header = rows[0][1:]
    M = {}
    classes = []
    for w in rows[1:]:
        if not w or not w[0].strip():
            continue
        try:
            map_class = int(w[0])
        except ValueError:
            continue
        classes.append(map_class)
        for j, h in enumerate(header):
            try:
                M[(map_class, int(h))] = float(w[j + 1] or 0)
            except (ValueError, IndexError):
                pass
    classes = sorted(set(classes) & {k[1] for k in M})
    if not classes:
        return None
    total = 0.0
    for c in classes:
        tp = M.get((c, c), 0.0)
        ref = sum(M.get((m, c), 0.0) for m in classes)
        map_class = sum(M.get((c, r), 0.0) for r in classes)
        pa = tp / ref if ref else 0.0
        ua = tp / map_class if map_class else 0.0
        total += 2 * pa * ua / (pa + ua) if (pa + ua) else 0.0
    return 100.0 * total / len(classes)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wyniki", default=RESULTS_DIR)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    directories = sorted(glob.glob(os.path.join(args.wyniki, "*", "accuracy")))
    if not directories:
        raise SystemExit("Brak podkatalogów accuracy/ w %s" % args.wyniki)

    rows = []
    for directory in directories:
        run_id = os.path.basename(os.path.dirname(directory))
        pred = _load_predictions(directory)
        if not pred:
            continue
        for algorithm in ALGORITHMS:
            baseline = pred.get(("Baseline", algorithm))
            if baseline is None:
                continue
            candidates = [(m, t) for (m, a), t in pred.items()
                         if a == algorithm and m != "Baseline"]
            if not candidates:
                continue
            method, best_rows = max(candidates, key=lambda kt: sum(kt[1]))

            n = len(baseline)
            oa_b, lo_b, hi_b = wilson(sum(baseline), n)
            oa_m, lo_m, hi_m = wilson(sum(best_rows), n)
            b = sum(1 for x, y in zip(baseline, best_rows) if x and not y)
            c = sum(1 for x, y in zip(baseline, best_rows) if y and not x)
            d, lo_d, hi_d = paired_ci(b, c, n)

            f1_b = _f1_makro(os.path.join(
                directory, "Baseline__%s_macierz.csv" % algorithm))
            f1_m = _f1_makro(os.path.join(
                directory, "%s__%s_macierz.csv"
                % (method.replace(" ", "_"), algorithm)))

            rows.append({
                "Run_ID": run_id, "Algorytm": DISPLAY_NAME[algorithm],
                "Metoda": method, "n_punktow": n,
                "OA_bazowa_%": 100 * oa_b,
                "OA_bazowa_CI_dol_%": 100 * lo_b,
                "OA_bazowa_CI_gora_%": 100 * hi_b,
                "OA_metody_%": 100 * oa_m,
                "OA_metody_CI_dol_%": 100 * lo_m,
                "OA_metody_CI_gora_%": 100 * hi_m,
                "Delta_pp": 100 * d,
                "Delta_CI_dol_pp": 100 * lo_d,
                "Delta_CI_gora_pp": 100 * hi_d,
                "CI_wyklucza_zero": "TAK" if lo_d > 0 or hi_d < 0 else "nie",
                "F1_makro_bazowa_%": f1_b, "F1_makro_metody_%": f1_m,
                "Delta_F1_makro_pp": (None if f1_b is None or f1_m is None
                                      else f1_m - f1_b),
            })

    _save(rows)
    _summarise(rows, args.wyniki)

def _save(rows):
    if not rows:
        raise SystemExit("Brak danych do zapisania.")
    pola = list(rows[0].keys())
    path = os.path.join(OUT_DIR, "przedzialy_ufnosci.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=pola)
        writer.writeheader()
        for w in rows:
            writer.writerow({
                k: ("" if v is None else
                    ("%.4f" % v if isinstance(v, float) else v))
                for k, v in w.items()
            })
    print("  zapisano przedzialy_ufnosci.csv (%d wierszy)" % len(rows))

def _relative_path(p):
    w = os.path.relpath(os.path.abspath(p), _BADANIA)
    if w.startswith(os.pardir):
        return os.path.basename(os.path.abspath(p))
    return w

def _summarise(rows, source):
    import statistics as st
    excludes = [w for w in rows if w["CI_wyklucza_zero"] == "TAK"]
    dodatnie = [w for w in excludes if w["Delta_pp"] > 0]
    width = [w["Delta_CI_gora_pp"] - w["Delta_CI_dol_pp"] for w in rows]
    f1 = [w["Delta_F1_makro_pp"] for w in rows
          if w["Delta_F1_makro_pp"] is not None]
    oa = [w["Delta_pp"] for w in rows]

    lines = [
        "PRZEDZIAŁY UFNOŚCI I F1 MAKRO (poziom 95%)",
        "=" * 70,
        "źródło danych: %s" % _relative_path(source),
        "porównań (przebieg x algorytm): %d" % len(rows),
        "punktów referencyjnych w przebiegu: %d" % rows[0]["n_punktow"],
        "",
        "PRZYROST DOKŁADNOŚCI",
        "  średni przyrost:            %+.2f pp" % st.mean(oa),
        "  przedział wyklucza zero:    %d z %d" % (len(excludes), len(rows)),
        "     w tym przyrosty:         %d" % len(dodatnie),
        "     w tym spadki:            %d" % (len(excludes) - len(dodatnie)),
        "  średnia szerokość przedziału: %.2f pp" % st.mean(width),
        "",
        "F1 MAKRO (kontrola dominacji klasy najliczniejszej)",
        "  porównań z macierzami:      %d" % len(f1),
        "  średni przyrost F1 makro:   %+.2f pp" % (st.mean(f1) if f1 else 0),
        "  przyrost dodatni w:         %d z %d"
        % (sum(1 for x in f1 if x > 0), len(f1)),
    ]
    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "podsumowanie_ci.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text)
    print("\nWyniki w: %s" % OUT_DIR)

if __name__ == "__main__":
    main()
