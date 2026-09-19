#!/usr/bin/env bash

# Wyznacza dokładność z gotowych rastrów klasyfikacji, bez powtarzania klasyfikacji.
#
# WEJŚCIE:      wyniki_badan/podbadania_v2 oraz warstwy referencyjne wariantów
# WYJŚCIE:      pliki oceny dokładności w katalogach przebiegów
# URUCHOMIENIE: ./badania/wyniki/ocen_podbadania.sh   (nadpisanie: OUT=... REF_DIR=...)

QGISPY="${QGISPY:-/Applications/QGIS-LTR.app/Contents/MacOS/bin/python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$SCRIPT_DIR/benchmark_outlier_removal.py"
BADANIA="$(cd "$SCRIPT_DIR/.." && pwd)"
DANE="${DANE_DIR:-$BADANIA/dane_testowe}"
WYNIKI="${WYNIKI_BADAN_DIR:-$BADANIA/wyniki_badan}"
OUT="${OUT:-$WYNIKI/podbadania_v2}"
LOGDIR="$OUT/logi_ocena"
REF_DIR="${REF_DIR:-$DANE/referencja2/warianty}"
REF_FIELD="${REF_FIELD:-klasa}"

DEFAULT_STEPS=(gridsearch_zbledami maxk2_zbledami maxk2_poprawny \
               rf_strojony_zbledami rf_strojony_poprawny)

N_OK=0; N_FAIL=0; N_SKIP=0; FAILED_STEPS=""

_eval() {
  local name="$1"
  local dir="$OUT/$name"
  local log="$LOGDIR/${name}.log"
  echo "=================================================================="
  echo "[$(date '+%F %T')] OCENA ${name}"
  echo "  katalog: ${dir}"
  echo "  log:     ${log}"

  if [ ! -d "$dir" ]; then
    echo "  POMINIĘTO (brak katalogu)"
    N_SKIP=$((N_SKIP + 1)); return 0
  fi
  if ! find "$dir" -type d -name classifications 2>/dev/null | grep -q .; then
    echo "  POMINIĘTO (brak rastrów w classifications/ — uruchom pod-badanie"
    echo "             z --keep-classifications)"
    N_SKIP=$((N_SKIP + 1)); return 0
  fi

  local t0; t0=$(date +%s)
  "$QGISPY" "$BENCH" \
    --evaluate-results-dir "$dir" \
    --reference-dir "$REF_DIR" --ref-field "$REF_FIELD" >"$log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then
    echo "  OK  (czas ${dt}s)"
    N_OK=$((N_OK + 1))
  else
    echo "  BŁĄD (kod ${rc}, czas ${dt}s) — szczegóły w logu: ${log}"
    N_FAIL=$((N_FAIL + 1)); FAILED_STEPS="${FAILED_STEPS} ${name}"
  fi
  return 0
}

main() {
  mkdir -p "$LOGDIR"

  if [ "$#" -gt 0 ]; then STEPS=("$@"); else STEPS=("${DEFAULT_STEPS[@]}"); fi

  echo "Ocena pod-badań (bez klasyfikacji): ${STEPS[*]}"
  echo "Referencja: $REF_DIR (pole: $REF_FIELD)"
  echo "Start: $(date '+%F %T')"

  for step in "${STEPS[@]}"; do _eval "$step"; done

  echo "=================================================================="
  echo "KONIEC: $(date '+%F %T')"
  echo "  ocenione OK: $N_OK,  błędy: $N_FAIL,  pominięte: $N_SKIP"
  [ -n "$FAILED_STEPS" ] && echo "  katalogi z błędem:${FAILED_STEPS}"
}

main "$@"
