#!/usr/bin/env bash
# Uruchamia komplet obliczeń badania głównego oraz przegląd parametrów dla zbioru poprawnego.
#
# WEJŚCIE:      dane_testowe: obszary treningowe, warstwa referencyjna oraz raster L1C
# WYJŚCIE:      wyniki_badan/benchmark_results_v2 oraz wyniki_badan/podbadania_v2/gridsearch_poprawny
# URUCHOMIENIE: ./badania/wyniki/uruchom_pelne_przeliczenie.sh [--dry-run | macierz | grid]

set -u

QGISPY="${QGISPY:-/Applications/QGIS-LTR.app/Contents/MacOS/bin/python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$SCRIPT_DIR/benchmark_outlier_removal.py"
DRIVER="$SCRIPT_DIR/uruchom_pelne_przeliczenie.py"
BADANIA="$(cd "$SCRIPT_DIR/.." && pwd)"
DANE="${DANE_DIR:-$BADANIA/dane_testowe}"
WYNIKI="${WYNIKI_BADAN_DIR:-$BADANIA/wyniki_badan}"

OUT_MACIERZ="$WYNIKI/benchmark_results_v2"
OUT_GRID="$WYNIKI/podbadania_v2/gridsearch_poprawny"
LOGDIR="$WYNIKI/logi_v2"

REF="$DANE/referencja2/warianty/referencja_1234567.shp"
IMG="$DANE/L1C/L1C_przyciety.tif"
ROI_OK="$DANE/ROI/warianty/roi_1234567.shp"

NPROC="${NPROC:-8}"
RAM="${RAM:-36000}"
WYMAGANE_GB="${WYMAGANE_GB:-35}"

DRY=0
STEPS=(macierz grid)
ARGS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    macierz|grid) ARGS+=("$a") ;;
    *) echo "Nieznany argument: $a"; exit 2 ;;
  esac
done
[ ${#ARGS[@]} -gt 0 ] && STEPS=("${ARGS[@]}")

mkdir -p "$LOGDIR"

echo "=============================================================================="
echo "PEŁNE PRZELICZENIE WYNIKÓW"
echo "=============================================================================="
echo "  start:   $(date '+%F %T')"
echo "  kroki:   ${STEPS[*]}"
echo "  logi:    $LOGDIR"
echo

if [ ! -x "$QGISPY" ]; then
  echo "BŁĄD: brak Pythona QGIS-LTR: $QGISPY"
  echo "      Ustaw ścieżkę zmienną QGISPY=... i uruchom ponownie."
  exit 1
fi
for f in "$BENCH" "$DRIVER" "$REF" "$IMG" "$ROI_OK"; do
  if [ ! -e "$f" ]; then echo "BŁĄD: brak pliku $f"; exit 1; fi
done

WOLNE_GB=$(df -g "$WYNIKI" | awk 'NR==2 {print $4}')
echo "  wolne miejsce: ${WOLNE_GB} GB (wymagane około ${WYMAGANE_GB} GB)"
if [ "$WOLNE_GB" -lt "$WYMAGANE_GB" ]; then
  echo
  echo "PRZERWANO: za mało miejsca na dysku."
  echo "  Komplet wyników potrzebuje około 30 GB, ponieważ rastry klasyfikacji"
  echo "  są zachowywane, zgodnie z wymaganiem powtarzalności."
  echo
  echo "  Możliwości:"
  echo "   * zwolnić miejsce poza katalogiem badania,"
  echo "   * wymusić start mimo ostrzeżenia: WYMAGANE_GB=0 $0"
  exit 1
fi
echo

N_OK=0; N_FAIL=0; FAILED=""

_krok() {
  local id="$1"; local log="$2"; shift 2
  echo "=============================================================================="
  echo "[$(date '+%F %T')] KROK ${id}"
  echo "  log: ${log}"
  local t0; t0=$(date +%s)
  caffeinate -is "$@" >"$log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then
    printf "  OK  (czas %d h %02d min)\n" $((dt/3600)) $(((dt%3600)/60))
    N_OK=$((N_OK+1))
  else
    printf "  BŁĄD (kod %d, czas %d h %02d min), szczegóły w logu\n" "$rc" $((dt/3600)) $(((dt%3600)/60))
    N_FAIL=$((N_FAIL+1)); FAILED="${FAILED} ${id}"
  fi
}

step_macierz() {
  if [ "$DRY" = "1" ]; then
    "$QGISPY" "$DRIVER" --dry-run
    return
  fi
  _krok macierz "$LOGDIR/macierz.log" "$QGISPY" "$DRIVER"
}

step_grid() {
  if [ "$DRY" = "1" ]; then
    echo "[dry-run] przegląd parametrów dla zbioru poprawnego -> $OUT_GRID"
    return
  fi
  if find "$OUT_GRID" -name wyniki_zbiorcze.csv 2>/dev/null | grep -q .; then
    echo "KROK grid: POMINIĘTO (wynik już istnieje w $OUT_GRID)"
    return
  fi
  _krok grid "$LOGDIR/grid_poprawny.log" \
    "$QGISPY" "$BENCH" \
    --image "$IMG" \
    --signatures "$ROI_OK" \
    --reference "$REF" --ref-field klasa \
    --keep-classifications \
    --scope class --grid-search \
    --methods Percentile "Robust Mahalanobis" GMM OneClassSVM EllipticEnvelope IsolationForest \
    --algorithm "spectral angle mapping" "support vector machine" "multi-layer perceptron" \
    --variant 1234567 --state poprawny --raster L1C \
    --output "$OUT_GRID" --n-processes "$NPROC" --ram "$RAM"
}

for s in "${STEPS[@]}"; do
  "step_$s"
done

echo "=============================================================================="
echo "[$(date '+%F %T')] PODSUMOWANIE"
if [ "$DRY" = "1" ]; then
  echo "  --dry-run: powyżej plan; nic nie policzono."
else
  echo "  kroków zakończonych powodzeniem: $N_OK"
  echo "  kroków zakończonych błędem:      $N_FAIL${FAILED:+ ($FAILED)}"
  echo
  echo "  wyniki badania głównego: $OUT_MACIERZ"
  echo "  przegląd parametrów:     $OUT_GRID"
fi
echo "=============================================================================="
[ "$N_FAIL" -eq 0 ]
