#!/usr/bin/env bash
# Uruchamia badania uzupełniające, każde w obu stanach zbioru uczącego.
#
# WEJŚCIE:      dane_testowe: obszary treningowe, warstwa referencyjna oraz raster L1C
# WYJŚCIE:      wyniki_badan/podbadania_v2/<krok>/
# URUCHOMIENIE: ./badania/wyniki/uruchom_podbadania_v2.sh [--dry-run | A A_poprawny B C D_zbledami D_poprawny]

set -u

QGISPY="${QGISPY:-/Applications/QGIS-LTR.app/Contents/MacOS/bin/python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$SCRIPT_DIR/benchmark_outlier_removal.py"
BADANIA="$(cd "$SCRIPT_DIR/.." && pwd)"
DANE="${DANE_DIR:-$BADANIA/dane_testowe}"
WYNIKI="${WYNIKI_BADAN_DIR:-$BADANIA/wyniki_badan}"

OUT="$WYNIKI/podbadania_v2"
LOGDIR="$OUT/logi"

REF="$DANE/referencja2/warianty/referencja_1234567.shp"
IMG="$DANE/L1C/L1C_przyciety.tif"
ROI_BLED="$DANE/ROI_blad/warianty/roi_1234567_bledy.shp"
ROI_OK="$DANE/ROI/warianty/roi_1234567.shp"

NPROC="${NPROC:-8}"
RAM="${RAM:-36000}"
FORCE="${FORCE:-0}"
WYMAGANE_GB="${WYMAGANE_GB:-6}"

DEFAULT_STEPS=(D_zbledami D_poprawny B C A A_poprawny)

DRY=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    A|A_poprawny|B|C|D_zbledami|D_poprawny) ARGS+=("$a") ;;
    *) echo "Nieznany krok: $a"
       echo "  dostępne: A A_poprawny B C D_zbledami D_poprawny"; exit 2 ;;
  esac
done
STEPS=("${DEFAULT_STEPS[@]}")
[ ${#ARGS[@]} -gt 0 ] && STEPS=("${ARGS[@]}")

mkdir -p "$LOGDIR"

N_OK=0; N_FAIL=0; N_SKIP=0; FAILED=""

echo "=============================================================================="
echo "POD-BADANIA, PRZELICZENIE"
echo "=============================================================================="
echo "  start:      $(date '+%F %T')"
echo "  kroki:      ${STEPS[*]}"
echo "  wyjście:    $OUT"
echo "  referencja: $REF"
echo "  zasoby:     n-processes=$NPROC, ram=$RAM MB, FORCE=$FORCE"
echo

if [ ! -x "$QGISPY" ]; then
  echo "BŁĄD: brak Pythona QGIS-LTR: $QGISPY"; exit 1
fi
for f in "$BENCH" "$REF" "$IMG" "$ROI_BLED" "$ROI_OK"; do
  [ -e "$f" ] || { echo "BŁĄD: brak pliku $f"; exit 1; }
done

WOLNE_GB=$(df -g "$WYNIKI" | awk 'NR==2 {print $4}')
echo "  wolne miejsce: ${WOLNE_GB} GB (wymagane około ${WYMAGANE_GB} GB)"
if [ "$WOLNE_GB" -lt "$WYMAGANE_GB" ]; then
  echo
  echo "PRZERWANO: za mało miejsca na dysku."
  echo "  Badania uzupełniające zapisują rastry klasyfikacji, łącznie około 2,8 GB."
  echo
  echo "  Wymuszenie startu mimo ostrzeżenia: WYMAGANE_GB=0 $0"
  exit 1
fi
echo

_krok() {
  local id="$1"; local outdir="$2"; shift 2
  local log="$LOGDIR/${id}.log"
  echo "=============================================================================="
  echo "[$(date '+%F %T')] KROK ${id}"
  echo "  wyjście: ${outdir}"
  echo "  log:     ${log}"

  if [ "$FORCE" != "1" ] && find "$outdir" -name wyniki_zbiorcze.csv 2>/dev/null | grep -q .; then
    echo "  POMINIĘTO (wynik już istnieje; FORCE=1 wymusza przeliczenie)"
    N_SKIP=$((N_SKIP+1)); return 0
  fi
  if [ "$DRY" = "1" ]; then
    echo "  [dry-run] nic nie uruchamiam"
    return 0
  fi

  local t0; t0=$(date +%s)
  caffeinate -is "$QGISPY" "$BENCH" "$@" >"$log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then
    printf "  OK  (czas %d h %02d min)\n" $((dt/3600)) $(((dt%3600)/60))
    N_OK=$((N_OK+1))
  else
    printf "  BŁĄD (kod %d, czas %d h %02d min), szczegóły w logu\n" "$rc" $((dt/3600)) $(((dt%3600)/60))
    N_FAIL=$((N_FAIL+1)); FAILED="${FAILED} ${id}"
  fi
  return 0
}

step_A_poprawny() {
  _krok A_poprawny "$OUT/gridsearch_poprawny" \
    --image "$IMG" \
    --signatures "$ROI_OK" \
    --reference "$REF" --ref-field klasa \
    --keep-classifications \
    --scope class --grid-search \
    --methods Percentile "Robust Mahalanobis" GMM OneClassSVM EllipticEnvelope IsolationForest \
    --algorithm "spectral angle mapping" "support vector machine" "multi-layer perceptron" \
    --variant 1234567 --state poprawny --raster L1C \
    --output "$OUT/gridsearch_poprawny" --n-processes "$NPROC" --ram "$RAM"
}

step_A() {
  _krok A "$OUT/gridsearch_zbledami" \
    --image "$IMG" \
    --signatures "$ROI_BLED" \
    --reference "$REF" --ref-field klasa \
    --keep-classifications \
    --scope class --grid-search \
    --methods Percentile "Robust Mahalanobis" GMM OneClassSVM EllipticEnvelope IsolationForest \
    --algorithm "spectral angle mapping" "support vector machine" "multi-layer perceptron" \
    --variant 1234567 --state zbledami --raster L1C \
    --output "$OUT/gridsearch_zbledami" --n-processes "$NPROC" --ram "$RAM"
}

step_B() {
  _krok B "$OUT/maxk2_zbledami" \
    --image "$IMG" \
    --signatures "$ROI_BLED" \
    --reference "$REF" --ref-field klasa \
    --keep-classifications \
    --scope class --max-k 2 \
    --methods PCA "Robust Mahalanobis" MAD Percentile SAM \
    --variant 1234567 --state zbledami --raster L1C \
    --output "$OUT/maxk2_zbledami" --n-processes "$NPROC" --ram "$RAM"
}

step_C() {
  _krok C "$OUT/maxk2_poprawny" \
    --image "$IMG" \
    --signatures "$ROI_OK" \
    --reference "$REF" --ref-field klasa \
    --keep-classifications \
    --scope class --max-k 2 \
    --methods PCA "Robust Mahalanobis" MAD Percentile SAM \
    --variant 1234567 --state poprawny --raster L1C \
    --output "$OUT/maxk2_poprawny" --n-processes "$NPROC" --ram "$RAM"
}

step_D_zbledami() {
  _krok D_zbledami "$OUT/rf_strojony_zbledami" \
    --image "$IMG" \
    --signatures "$ROI_BLED" \
    --reference "$REF" --ref-field klasa \
    --keep-classifications \
    --scope class --algorithm "random forest" \
    --rf-max-features sqrt --rf-number-trees 500 \
    --variant 1234567 --state zbledami --raster L1C \
    --output "$OUT/rf_strojony_zbledami" --n-processes "$NPROC" --ram "$RAM"
}

step_D_poprawny() {
  _krok D_poprawny "$OUT/rf_strojony_poprawny" \
    --image "$IMG" \
    --signatures "$ROI_OK" \
    --reference "$REF" --ref-field klasa \
    --keep-classifications \
    --scope class --algorithm "random forest" \
    --rf-max-features sqrt --rf-number-trees 500 \
    --variant 1234567 --state poprawny --raster L1C \
    --output "$OUT/rf_strojony_poprawny" --n-processes "$NPROC" --ram "$RAM"
}

for s in "${STEPS[@]}"; do
  "step_$s"
done

echo "=============================================================================="
echo "[$(date '+%F %T')] PODSUMOWANIE"
if [ "$DRY" = "1" ]; then
  echo "  --dry-run: powyżej plan; nic nie policzono."
else
  echo "  powodzenie: $N_OK    pominięte: $N_SKIP    błędy: $N_FAIL${FAILED:+ ($FAILED)}"
  echo
  echo "  wyniki: $OUT"
  echo
  echo "  Ocena dokładności na gotowych rastrach:"
  echo "    OUT=\"$OUT\" ./badania/wyniki/ocen_podbadania.sh"
fi
echo "=============================================================================="
[ "$N_FAIL" -eq 0 ]
