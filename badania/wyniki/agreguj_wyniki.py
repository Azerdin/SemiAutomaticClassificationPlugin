# -*- coding: utf-8 -*-
"""Agreguje wyniki badania głównego do zestawień zbiorczych oraz rysunków.

WEJŚCIE:      wyniki_badan/benchmark_results_v2/*/konfirmacja_mcnemar.csv
WYJŚCIE:      wyniki_badan/agregacja_v2/T*.csv, pelne_wyniki_384.csv oraz wyniki_badan/img/F*.png
URUCHOMIENIE: python3 badania/wyniki/agreguj_wyniki.py   (wymaga numpy i matplotlib >= 3.4)
"""

import csv
import glob
import os
import statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BADANIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANE = os.environ.get("DANE_DIR", os.path.join(_BADANIA, "dane_testowe"))
WYNIKI = os.environ.get("WYNIKI_BADAN_DIR",
                        os.path.join(_BADANIA, "wyniki_badan"))
RESULTS_DIR = os.environ.get("WYNIKI_DIR", WYNIKI + "/benchmark_results_v2")
OUT_DIR = os.environ.get("AGREGACJA_DIR", WYNIKI + "/agregacja_v2")
IMG_DIR = os.path.join(WYNIKI, "img")

ALGO_ORDER = [
    "maximum likelihood", "random forest", "multi-layer perceptron",
    "support vector machine", "spectral angle mapping", "minimum distance",
]
ALGO_SHORT = {
    "maximum likelihood": "MLC", "random forest": "RF",
    "multi-layer perceptron": "MLP", "support vector machine": "SVM",
    "spectral angle mapping": "SAM", "minimum distance": "MD",
}

def _f(x):
    return float(str(x).replace("+", "").strip())

def load_konfirmacje(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "konfirmacja_mcnemar.csv"))):
        with open(path, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    if not rows:
        raise SystemExit("Brak plików konfirmacja_mcnemar.csv w %s" % results_dir)
    return rows

def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")

def _sig(r):
    return r["Istotne_005_lokalnie"].strip().upper() == "TAK"

def _write_csv(path, header, table):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(table)

def table_T1(rows, out_dir):
    scopes = ["roi", "class"]
    header = ["Algorytm", "delta_roi_pp", "delta_class_pp",
              "istotnych_roi", "istotnych_class", "istotnych_razem", "n_na_zakres"]
    table = []
    for algo in ALGO_ORDER:
        cells = {s: [_f(r["Delta_DA_pp"]) for r in rows
                     if r["Algorytm"] == algo and r["Zakres"] == s] for s in scopes}
        sig = {s: sum(_sig(r) for r in rows
                      if r["Algorytm"] == algo and r["Zakres"] == s) for s in scopes}
        table.append([
            algo, round(_mean(cells["roi"]), 2), round(_mean(cells["class"]), 2),
            sig["roi"], sig["class"], sig["roi"] + sig["class"], len(cells["roi"]),
        ])
    _write_csv(os.path.join(out_dir, "T1_algorytm_x_zakres.csv"), header, table)
    return header, table

def table_T2(rows, out_dir):
    header = ["Stan", "Zakres", "sr_delta_pp", "istotnych", "n"]
    table = []
    for state in ["poprawny", "zbledami"]:
        for scope in ["roi", "class"]:
            g = [r for r in rows if r["Stan"] == state and r["Zakres"] == scope]
            table.append([state, scope, round(_mean([_f(r["Delta_DA_pp"]) for r in g]), 2),
                          sum(_sig(r) for r in g), len(g)])
    _write_csv(os.path.join(out_dir, "T2_stan_x_zakres.csv"), header, table)
    return header, table

def table_T3(rows, out_dir):
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["Najlepsza_konfiguracja"]].append(r)
    header = ["Metoda", "razy_najlepsza", "udzial_%", "sr_delta_pp",
              "mediana_delta_pp", "istotnych"]
    table = []
    total = len(rows)
    for method, g in by_method.items():
        deltas = [_f(r["Delta_DA_pp"]) for r in g]
        table.append([method, len(g), round(100.0 * len(g) / total, 1),
                      round(_mean(deltas), 2), round(st.median(deltas), 2),
                      sum(_sig(r) for r in g)])
    table.sort(key=lambda row: (-row[1], -row[3]))
    _write_csv(os.path.join(out_dir, "T3_ranking_metod.csv"), header, table)
    return header, table

def table_T4(rows, out_dir):
    header = ["Raster", "Zakres", "sr_delta_pp", "istotnych", "n"]
    table = []
    for raster in sorted(set(r["Raster"] for r in rows)):
        for scope in ["roi", "class"]:
            g = [r for r in rows if r["Raster"] == raster and r["Zakres"] == scope]
            table.append([raster, scope, round(_mean([_f(r["Delta_DA_pp"]) for r in g]), 2),
                          sum(_sig(r) for r in g), len(g)])
    _write_csv(os.path.join(out_dir, "T4_raster_x_zakres.csv"), header, table)
    return header, table

def _baseline_map(rows):
    return {(r["Wariant"], r["Raster"], r["Stan"], r["Algorytm"]):
            _f(r["DA_bazowa_%"]) for r in rows}

def table_baseline_correct_vs_errors(rows, out_dir):
    bm = _baseline_map(rows)
    variants = sorted(set(k[0] for k in bm))
    rasters = sorted(set(k[1] for k in bm))
    header = ["Algorytm", "OA_baz_poprawny", "OA_baz_zbledami",
              "Delta_zbl_minus_pop", "zbl_lepszy", "n"]
    table = []
    for a in ALGO_ORDER:
        pop, zbl, better = [], [], 0
        for v in variants:
            for ra in rasters:
                p = bm.get((v, ra, "poprawny", a))
                z = bm.get((v, ra, "zbledami", a))
                if p is None or z is None:
                    continue
                pop.append(p); zbl.append(z)
                if z > p:
                    better += 1
        table.append([a, round(_mean(pop), 2), round(_mean(zbl), 2),
                      round(_mean(zbl) - _mean(pop), 2), better, len(pop)])
    table.sort(key=lambda r: -r[3])
    _write_csv(os.path.join(out_dir, "T5_baseline_poprawny_vs_bledy.csv"),
               header, table)
    return header, table

def _removal_map(rows):
    m = {}
    for r in rows:
        k = (r["Wariant"], r["Raster"], r["Stan"], r["Algorytm"])
        v = _f(r["DA_najlepsza_%"])
        if v is not None and (k not in m or v > m[k]):
            m[k] = v
    return m

def table_removal_correct_vs_errors(rows, out_dir):
    bm = _removal_map(rows)
    variants = sorted(set(k[0] for k in bm))
    rasters = sorted(set(k[1] for k in bm))
    header = ["Algorytm", "OA_po_poprawny", "OA_po_zbledami",
              "Delta_zbl_minus_pop", "zbl_lepszy", "n"]
    table = []
    for a in ALGO_ORDER:
        pop, zbl, better = [], [], 0
        for v in variants:
            for ra in rasters:
                p = bm.get((v, ra, "poprawny", a))
                z = bm.get((v, ra, "zbledami", a))
                if p is None or z is None:
                    continue
                pop.append(p); zbl.append(z)
                if z > p:
                    better += 1
        table.append([a, round(_mean(pop), 2), round(_mean(zbl), 2),
                      round(_mean(zbl) - _mean(pop), 2), better, len(pop)])
    table.sort(key=lambda r: -r[3])
    _write_csv(os.path.join(out_dir, "T5b_removal_poprawny_vs_bledy.csv"),
               header, table)
    return header, table

def _ranking(rows, unit_keys, value_fn):
    units = defaultdict(dict)
    for r in rows:
        units[tuple(r[k] for k in unit_keys)][r["Algorytm"]] = value_fn(r)
    ranks, vals = defaultdict(list), defaultdict(list)
    for algo_val in units.values():
        for pos, (a, v) in enumerate(
                sorted(algo_val.items(), key=lambda kv: -kv[1]), 1):
            ranks[a].append(pos); vals[a].append(v)
    return {a: (_mean(ranks[a]), _mean(vals[a]), len(ranks[a])) for a in ranks}

def _write_ranking(rows, out_dir, name, unit_keys, value_fn, oa_col):
    rk = _ranking(rows, unit_keys, value_fn)
    header = ["Miejsce", "Algorytm", "sr_pozycja_1-6", oa_col, "n"]
    table = [[i, a, round(v[0], 2), round(v[1], 2), v[2]]
             for i, (a, v) in enumerate(
                 sorted(rk.items(), key=lambda kv: kv[1][0]), 1)]
    _write_csv(os.path.join(out_dir, name + ".csv"), header, table)
    return header, table

def table_ranking_baseline(rows, out_dir):
    return _write_ranking(rows, out_dir, "T6_ranking_baseline",
                          ("Wariant", "Stan", "Raster"),
                          lambda r: _f(r["DA_bazowa_%"]), "sr_OA_baz")

def table_ranking_after_removal(rows, out_dir):
    return _write_ranking(rows, out_dir, "T7_ranking_po_usuwaniu",
                          ("Wariant", "Stan", "Raster", "Zakres"),
                          lambda r: _f(r["DA_najlepsza_%"]), "sr_OA_najlepsza")

def _load_baseline_kappa(results_dir):
    out = {}
    for f in glob.glob(os.path.join(results_dir, "*", "wyniki_zbiorcze.csv")):
        with open(f, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("Metoda", "").strip() in ("Poziom bazowy", "Baseline"):
                    out[(r["Run_ID"], r["Algorytm"])] = r.get("Kappa", "")
    return out

def table_pelne(rows, out_dir):
    algo_rank = {a: i for i, a in enumerate(ALGO_ORDER)}
    raster_rank = {"L1C": 0, "L2A": 1}
    state_rank = {"poprawny": 0, "zbledami": 1}
    scope_rank = {"roi": 0, "class": 1}

    def keyf(r):
        return (r["Wariant"], raster_rank.get(r["Raster"], 9),
                state_rank.get(r["Stan"], 9), scope_rank.get(r["Zakres"], 9),
                algo_rank.get(r["Algorytm"], 99))

    srt = sorted(rows, key=keyf)
    bkap = _load_baseline_kappa(RESULTS_DIR)

    def kap_csv(r):
        v = bkap.get((r["Run_ID"], r["Algorytm"]), "")
        try:
            return "%.4f" % float(v)
        except (ValueError, TypeError):
            return ""

    fields = ["Wariant", "Stan", "Raster", "Zakres", "Algorytm",
              "OA_bazowe_%", "Kappa_bazowe", "Najlepsza_metoda", "Delta_OA_pp",
              "Istotne_lok"]
    csv_rows = [[r["Wariant"], r["Stan"], r["Raster"], r["Zakres"], r["Algorytm"],
                 "%.2f" % _f(r["DA_bazowa_%"]), kap_csv(r),
                 r["Najlepsza_konfiguracja"],
                 "%+.2f" % _f(r["Delta_DA_pp"]), r["Istotne_005_lokalnie"]]
                for r in srt]
    _write_csv(os.path.join(out_dir, "pelne_wyniki_384.csv"), fields, csv_rows)
    return fields, csv_rows

def _save_fig(fig, fname):
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMG_DIR, fname), dpi=150)
    plt.close(fig)

def fig_F1(rows, out_dir):
    labels = [ALGO_SHORT[a] for a in ALGO_ORDER]
    roi = [_mean([_f(r["Delta_DA_pp"]) for r in rows
                  if r["Algorytm"] == a and r["Zakres"] == "roi"]) for a in ALGO_ORDER]
    cls = [_mean([_f(r["Delta_DA_pp"]) for r in rows
                  if r["Algorytm"] == a and r["Zakres"] == "class"]) for a in ALGO_ORDER]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, roi, w, label="zakres: poligon (roi)", color="#4C72B0")
    b2 = ax.bar(x + w / 2, cls, w, label="zakres: klasa (class)", color="#DD8452")
    ax.set_ylabel("średnia ΔOA [pp]")
    ax.set_title("Średnia poprawa dokładności po usunięciu wartości odstających")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="k", lw=0.8)
    ax.legend()
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    _save_fig(fig, "F1_delta_algorytm_zakres.png")

def fig_F2(rows, out_dir):
    data = [[_f(r["Delta_DA_pp"]) for r in rows if r["Algorytm"] == a] for a in ALGO_ORDER]
    labels = [ALGO_SHORT[a] for a in ALGO_ORDER]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.axhline(0, color="r", lw=0.8, ls="--")
    ax.set_ylabel("ΔOA [pp]")
    ax.set_title("Rozrzut poprawy dokładności wg algorytmu (64 przebiegi)")
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    _save_fig(fig, "F2_rozrzut_delta_algorytm.png")

def fig_F3(rows, out_dir):
    variants = sorted(set(r["Wariant"] for r in rows))
    mat = np.full((len(variants), len(ALGO_ORDER)), np.nan)
    for i, v in enumerate(variants):
        for j, a in enumerate(ALGO_ORDER):
            vals = [_f(r["Delta_DA_pp"]) for r in rows
                    if r["Wariant"] == v and r["Algorytm"] == a]
            if vals:
                mat[i, j] = _mean(vals)
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(ALGO_ORDER)))
    ax.set_xticklabels([ALGO_SHORT[a] for a in ALGO_ORDER])
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    ax.set_xlabel("algorytm")
    ax.set_ylabel("wariant (zestaw klas)")
    ax.set_title("Średnia ΔOA [pp] wg wariantu i algorytmu")
    for i in range(len(variants)):
        for j in range(len(ALGO_ORDER)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, "%.1f" % mat[i, j], ha="center", va="center",
                        fontsize=8, color="black")
    fig.colorbar(im, ax=ax, label="ΔOA [pp]")
    fig.tight_layout()
    _save_fig(fig, "F3_heatmapa_wariant_algorytm.png")

def _print_table(title, header, table):
    print("\n=== %s ===" % title)
    print("  " + " | ".join(str(h) for h in header))
    for row in table:
        print("  " + " | ".join(str(c) for c in row))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_konfirmacje(RESULTS_DIR)
    n_runs = len(set(r["Run_ID"] for r in rows))
    print("Wczytano %d wierszy konfirmacji z %d przebiegów." % (len(rows), n_runs))

    _print_table("T1 — ΔOA wg algorytmu × zakres", *table_T1(rows, OUT_DIR))
    _print_table("T2 — ΔOA wg stanu × zakres", *table_T2(rows, OUT_DIR))
    _print_table("T3 — ranking metod odstających", *table_T3(rows, OUT_DIR))
    _print_table("T4 — L1C vs L2A × zakres", *table_T4(rows, OUT_DIR))
    _print_table("T5 — baseline: poprawny vs z błędami",
                 *table_baseline_correct_vs_errors(rows, OUT_DIR))
    _print_table("T5b — po usuwaniu: poprawny vs z błędami",
                 *table_removal_correct_vs_errors(rows, OUT_DIR))
    _print_table("T6 — ranking baselinów (śr. pozycja 1-6)",
                 *table_ranking_baseline(rows, OUT_DIR))
    _print_table("T7 — ranking wyników po usuwaniu (śr. pozycja 1-6)",
                 *table_ranking_after_removal(rows, OUT_DIR))

    _, pelne = table_pelne(rows, OUT_DIR)
    print("\nPełna tabela: %d wierszy -> pelne_wyniki_384.csv" % len(pelne))

    fig_F1(rows, OUT_DIR)
    fig_F2(rows, OUT_DIR)
    fig_F3(rows, OUT_DIR)
    print("\nCSV (dane) i rysunki (.png) zapisane w:\n  %s" % OUT_DIR)

if __name__ == "__main__":
    main()
