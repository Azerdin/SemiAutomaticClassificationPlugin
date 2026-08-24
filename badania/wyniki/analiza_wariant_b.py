# -*- coding: utf-8 -*-
"""Stosuje korekcję Holma nad całą przestrzenią przeszukiwania, uwzględniając wybór metody.

WEJŚCIE:      wyniki_badan/benchmark_results_v2
WYJŚCIE:      wyniki_badan/analiza_wariant_b_v2/*.csv oraz podsumowanie.txt
URUCHOMIENIE: python3 badania/wyniki/analiza_wariant_b.py   (wymaga numpy i scipy)
"""

import argparse
import csv
import glob
import os
from math import comb

import numpy as np

_BADANIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANE = os.environ.get("DANE_DIR", os.path.join(_BADANIA, "dane_testowe"))
WYNIKI = os.environ.get("WYNIKI_BADAN_DIR",
                        os.path.join(_BADANIA, "wyniki_badan"))
DEFAULT_RESULTS_DIR = os.environ.get("WYNIKI_DIR", WYNIKI + "/benchmark_results_v2")
OUT_DIR = WYNIKI + "/analiza_wariant_b_v2"

ALGORITHMS = [
    "maximum_likelihood", "random_forest", "multi-layer_perceptron",
    "support_vector_machine", "spectral_angle_mapping", "minimum_distance",
]
ALGORITHM_NAME = {
    "maximum_likelihood": "maximum likelihood",
    "random_forest": "random forest",
    "multi-layer_perceptron": "multi-layer perceptron",
    "support_vector_machine": "support vector machine",
    "spectral_angle_mapping": "spectral angle mapping",
    "minimum_distance": "minimum distance",
}

def _mcnemar(baseline_hits, method_hits):
    b = int(np.sum(baseline_hits & ~method_hits))
    c = int(np.sum(~baseline_hits & method_hits))
    n = b + c
    if n == 0:
        return b, c, 1.0
    if n < 25:
        p = min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
        return b, c, p
    chi = (abs(b - c) - 1) ** 2 / n
    from scipy.stats import chi2
    return b, c, float(chi2.sf(chi, 1))

def _holm(p_values):
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    corrected = [0.0] * m
    previous = 0.0
    for ranga, i in enumerate(order):
        value = max(previous, min(1.0, p_values[i] * (m - ranga)))
        corrected[i] = value
        previous = value
    return corrected

def _load_predictions(accuracy_dir):
    results = {}
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
        reference = np.array([int(w["referencja"]) for w in rows])
        map_class = np.array([int(w["mapa"]) for w in rows])
        results[(method, algorithm)] = map_class == reference
    return results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wyniki", default=DEFAULT_RESULTS_DIR,
                        help="katalog z przebiegami benchmarku")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    directories = sorted(
        p for p in glob.glob(os.path.join(args.wyniki, "*", "accuracy"))
    )
    if not directories:
        raise SystemExit("Brak podkatalogów accuracy/ w %s" % args.wyniki)

    all_rows = []
    best_rows = []
    for directory in directories:
        run_id = os.path.basename(os.path.dirname(directory))
        predykcje = _load_predictions(directory)
        if not predykcje:
            continue

        testy = []
        for algorithm in ALGORITHMS:
            baseline = predykcje.get(("Baseline", algorithm))
            if baseline is None:
                continue
            for (method, alg), trafienia in predykcje.items():
                if alg != algorithm or method == "Baseline":
                    continue
                b, c, p = _mcnemar(baseline, trafienia)
                testy.append({
                    "Run_ID": run_id, "Algorytm": ALGORITHM_NAME[algorithm],
                    "Metoda": method,
                    "DA_bazowa_%": 100.0 * baseline.mean(),
                    "DA_metody_%": 100.0 * trafienia.mean(),
                    "Delta_DA_pp": 100.0 * (trafienia.mean() - baseline.mean()),
                    "b": b, "c": c, "p": p,
                })
        if not testy:
            continue

        corrected = _holm([t["p"] for t in testy])
        for test, p_holm in zip(testy, corrected):
            test["p_Holm_rodzina"] = p_holm
            test["Istotne_005"] = "TAK" if p_holm <= 0.05 else "nie"
            test["Rodzina_n"] = len(testy)
        all_rows.extend(testy)

        for algorithm in ALGORITHMS:
            candidates = [t for t in testy
                         if t["Algorytm"] == ALGORITHM_NAME[algorithm]]
            if not candidates:
                continue
            best = max(candidates, key=lambda t: t["DA_metody_%"])
            best_rows.append(best)

    _save_csv(os.path.join(OUT_DIR, "wszystkie_porownania.csv"), all_rows)
    _save_csv(os.path.join(OUT_DIR, "najlepsze_konfiguracje.csv"), best_rows)
    _summary(all_rows, best_rows, args.wyniki)

def _save_csv(path, rows):
    if not rows:
        return
    pola = ["Run_ID", "Algorytm", "Metoda", "DA_bazowa_%", "DA_metody_%",
            "Delta_DA_pp", "b", "c", "p", "p_Holm_rodzina", "Rodzina_n",
            "Istotne_005"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=pola)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: ("%.4f" % v if isinstance(v, float) else v)
                for k, v in row.items() if k in pola
            })
    print("  zapisano %s (%d wierszy)" % (os.path.basename(path), len(rows)))

def _relative_path(p):
    w = os.path.relpath(os.path.abspath(p), _BADANIA)
    if w.startswith(os.pardir):
        return os.path.basename(os.path.abspath(p))
    return w

def _summary(all_rows, best_rows, source):
    significant_all = [t for t in all_rows if t["Istotne_005"] == "TAK"]
    poprawy_w = [t for t in significant_all if t["Delta_DA_pp"] > 0]
    significant_best = [t for t in best_rows if t["Istotne_005"] == "TAK"]
    poprawy_n = [t for t in significant_best if t["Delta_DA_pp"] > 0]

    lines = [
        "WARIANT B — korekcja Holma nad pełną rodziną porównań",
        "=" * 70,
        "źródło danych: %s" % _relative_path(source),
        "",
        "A) WSZYSTKIE PORÓWNANIA (metoda x algorytm)",
        "   porównań razem:        %d" % len(all_rows),
        "   istotnych:             %d" % len(significant_all),
        "      w tym poprawy:      %d" % len(poprawy_w),
        "      w tym pogorszenia:  %d" % (len(significant_all) - len(poprawy_w)),
        "",
        "B) NAJLEPSZA METODA W OBRĘBIE ALGORYTMU (struktura jak w pracy)",
        "   porównań razem:        %d" % len(best_rows),
        "   istotnych:             %d" % len(significant_best),
        "      w tym poprawy:      %d" % len(poprawy_n),
        "      w tym pogorszenia:  %d" % (len(significant_best) - len(poprawy_n)),
    ]
    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "podsumowanie.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text)
    print("\nWyniki w: %s" % OUT_DIR)

if __name__ == "__main__":
    main()
