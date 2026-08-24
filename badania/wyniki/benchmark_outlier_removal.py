# -*- coding: utf-8 -*-

"""Rdzeń badania: wykonuje detekcję wartości odstających, przelicza sygnatury, klasyfikuje scenę i ocenia dokładność.

WEJŚCIE:      dane_testowe: obszary treningowe, warstwa referencyjna oraz raster
WYJŚCIE:      wyniki_badan/benchmark_results_v2/<przebieg>/ z macierzami błędów i zestawieniami
URUCHOMIENIE: python3 badania/wyniki/benchmark_outlier_removal.py   (Python QGIS-LTR)
"""

import argparse
import csv
import itertools
import logging
import multiprocessing
import multiprocessing.pool
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

_qgis = "/Applications/QGIS-LTR.app/Contents"
if os.path.isdir(_qgis):
    os.environ.setdefault("PROJ_LIB", _qgis + "/Resources/proj")
    os.environ.setdefault("GDAL_DATA", _qgis + "/Resources/gdal")

import numpy as np

try:
    from osgeo import gdal, ogr

    gdal.UseExceptions()
except ImportError:
    sys.exit(
        "BŁĄD: brak biblioteki GDAL (osgeo). Uruchom w konsoli Pythona programu QGIS "
        "or install 'gdal' in your environment."
    )

try:
    import sklearn
except ImportError:
    sys.exit("BŁĄD: brak biblioteki scikit-learn. Instalacja: pip install scikit-learn")

try:
    import scipy
except ImportError:
    sys.exit("BŁĄD: brak biblioteki scipy. Instalacja: pip install scipy")

try:
    import remotior_sensus
except ImportError:
    sys.exit("BŁĄD: brak biblioteki remotior_sensus. Instalacja: pip install remotior-sensus")

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, *_HERE.parents):
    if (_cand / "core" / "outlier_filters.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break
try:
    from core.outlier_filters import (
        stack_from_dataset,
        warp_multiband_to_memory as _warp_multiband_to_memory,
        mask_to_union_geometry,
    )
    from core.outlier_catalog import (
        apply_removal_to_catalog as _apply_removal_shared,
    )
except ImportError as _err:
    sys.exit(
        "BŁĄD: nie można zaimportować core.outlier_filters (%s). Uruchom skrypt "
        "from the SCP plugin repository" % _err
    )

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger("benchmark")

_BADANIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANE = os.environ.get("DANE_DIR", os.path.join(_BADANIA, "dane_testowe"))
WYNIKI = os.environ.get("WYNIKI_BADAN_DIR",
                        os.path.join(_BADANIA, "wyniki_badan"))

RASTERS = {
    "L1C": DANE + "/L1C/L1C_przyciety.tif",
    "L2A": DANE + "/L2A/L2A_przyciety.tif",
}

_ROI = DANE + "/ROI/warianty"
_ROIB = DANE + "/ROI_blad/warianty"
_REF = DANE + "/referencja2/warianty"
VARIANTS = {
    v: {
        "reference": "%s/referencja_%s.shp" % (_REF, v),
        "roi": {
            "poprawny": "%s/roi_%s.shp" % (_ROI, v),
            "zbledami": "%s/roi_%s_bledy.shp" % (_ROIB, v),
        },
    }
    for v in ["123456", "1234567", "12456", "124567",
              "13456", "134567", "1456", "14567"]
}

RASTERS_TO_RUN = ["L1C", "L2A"]
STATES_TO_RUN = ["poprawny", "zbledami"]
SCOPES_TO_RUN = ["roi", "class"]

ALGORITHMS = None
METHODS = None
GRID_SEARCH = False
MAX_K = 1
MACROCLASS = False
WAVELENGTHS = {
    13: [0.443, 0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865,
         0.945, 1.375, 1.61, 2.19],
    12: [0.443, 0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865,
         0.945, 1.61, 2.19],
}
WAVELENGTH_UNIT = "µm (1 E-6m)"
REF_FIELD = "klasa"

RF_NUMBER_TREES = 10
RF_MAX_FEATURES = None

ROI_MACROCLASS_FIELD = "klasa"
ROI_CLASS_FIELD = "klasa"
ROI_CLASS_NAME_FIELD = "c_name"

OUTPUT_DIR = WYNIKI + "/benchmark_results_v2"
N_PROCESSES = 8
RAM_MB = 36000
WORKERS = 1
KEEP_CLASSIFICATIONS = True
SKIP_DONE = True

_METHOD_DESCRIPTIONS = {
    "MAD": ("progowa", "Medianowe odchylenie bezwzględne — piksel odstający, "
            "gdy zmodyfikowany wynik Z liczony względem mediany przekracza próg."),
    "Band Z-score": ("progowa", "Klasyczny wynik Z liczony niezależnie w każdym "
                     "paśmie — odrzuca piksele oddalone od średniej o więcej niż "
                     "próg (w odchyleniach standardowych)."),
    "IQR": ("progowa", "Reguła rozstępu międzykwartylowego (Tukeya) — odstający "
            "poza przedziałem [Q1 - f*IQR, Q3 + f*IQR]."),
    "Mahalanobis": ("progowa", "Odległość Mahalanobisa od środka klasy z klasyczną "
                    "kowariancją; próg z rozkładu chi-kwadrat na poziomie alpha."),
    "Robust Mahalanobis": ("progowa", "Odległość Mahalanobisa z odporną estymacją "
                           "kowariancji (MCD), mniej wrażliwa na same odstające; "
                           "próg chi-kwadrat."),
    "HotellingT2": ("progowa", "Wielowymiarowy test T-kwadrat Hotellinga — odrzuca "
                    "obserwacje o statystyce powyżej wartości krytycznej (alpha)."),
    "PCA": ("progowa", "Odległość Mahalanobisa w przestrzeni głównych składowych "
            "(zachowana wariancja variance_ratio); próg chi-kwadrat."),
    "Percentile": ("frakcyjna", "Przycięcie percentylowe — usuwa piksele poza "
                   "zadanym zakresem percentyli w którymkolwiek paśmie."),
    "PCA Reconstruction": ("frakcyjna", "Błąd rekonstrukcji PCA — odrzuca ułamek "
                           "pikseli o największym błędzie odtworzenia z głównych "
                           "składowych."),
    "GMM": ("frakcyjna", "Mieszanina rozkładów Gaussa — odrzuca ułamek pikseli o "
            "najniższej gęstości prawdopodobieństwa modelu."),
    "EllipticEnvelope": ("frakcyjna", "Dopasowanie elipsoidy (odporna kowariancja) "
                         "— odrzuca zadany ułamek pikseli poza elipsoidą."),
    "IsolationForest": ("frakcyjna", "Las izolacyjny — odrzuca ułamek pikseli o "
                        "najkrótszej ścieżce izolacji (najłatwiej separowalnych)."),
    "LOF": ("frakcyjna", "Lokalny współczynnik odstawania — porównuje gęstość "
            "lokalną punktu z gęstością jego sąsiadów; odrzuca ułamek."),
    "kNN": ("frakcyjna", "Odległość do k-tego najbliższego sąsiada — odrzuca "
            "ułamek pikseli o największej odległości."),
    "OneClassSVM": ("frakcyjna", "Jednoklasowa maszyna wektorów nośnych — uczy "
                    "granicy obszaru gęstego; parametr nu steruje ułamkiem "
                    "odrzuconych pikseli."),
    "SAM": ("frakcyjna", "Spectral Angle Mapper — kąt spektralny do średniej "
            "sygnatury klasy; odrzuca ułamek pikseli o największym kącie."),
    "Erosion": ("morfologiczna", "Erozja morfologiczna maski poligonu — usuwa "
                "piksele brzegowe ROI, ograniczając wpływ pikseli mieszanych na "
                "krawędziach."),
}

_DEFAULT_PARAMS = {
    "MAD": {"threshold": 3.5},
    "Band Z-score": {"threshold": 3.0},
    "Percentile": {"lower_pct": 0.01, "upper_pct": 0.99},
    "IQR": {"factor": 1.5},
    "Mahalanobis": {"alpha": 0.025},
    "Robust Mahalanobis": {"alpha": 0.025},
    "HotellingT2": {"alpha": 0.05},
    "PCA": {"variance_ratio": 0.95, "n_components": 3, "alpha": 0.025},
    "PCA Reconstruction": {
        "variance_ratio": 0.95, "n_components": 3, "contamination": 0.01,
    },
    "GMM": {"n_components": 2, "contamination": 0.01},
    "EllipticEnvelope": {"contamination": 0.01},
    "IsolationForest": {"contamination": 0.01},
    "LOF": {"n_neighbors": 20, "contamination": 0.01},
    "kNN": {"n_neighbors": 5, "contamination": 0.01},
    "OneClassSVM": {"nu": 0.01},
    "SAM": {"contamination": 0.01},
    "Erosion": {"iterations": 1},
}

_FRAC = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]

_PARAM_GRID = {
    "Percentile": [
        {"lower_pct": lo, "upper_pct": hi}
        for lo, hi in [(0.005, 0.995), (0.01, 0.99), (0.02, 0.98),
                       (0.05, 0.95), (0.10, 0.90), (0.20, 0.80)]
    ],
    "Mahalanobis": [{"alpha": a} for a in [0.01, 0.025, 0.05, 0.10]],
    "Robust Mahalanobis": [{"alpha": a} for a in [0.01, 0.025, 0.05, 0.10]],
    "HotellingT2": [{"alpha": a} for a in [0.01, 0.05, 0.10, 0.20]],
    "PCA": [
        {"variance_ratio": v, "alpha": a}
        for v in [0.90, 0.95, 0.99]
        for a in [0.01, 0.025, 0.05, 0.10]
    ],
    "PCA Reconstruction": [
        {"variance_ratio": v, "contamination": c}
        for v in [0.90, 0.95, 0.99]
        for c in _FRAC
    ],
    "GMM": [
        {"n_components": n, "contamination": c}
        for n in [2, 3]
        for c in _FRAC
    ],
    "EllipticEnvelope": [{"contamination": c} for c in _FRAC],
    "IsolationForest": [{"contamination": c} for c in _FRAC],
    "LOF": [
        {"n_neighbors": n, "contamination": c}
        for n in [10, 20, 30]
        for c in _FRAC
    ],
    "kNN": [
        {"n_neighbors": n, "contamination": c}
        for n in [5, 10, 20]
        for c in _FRAC
    ],
    "OneClassSVM": [{"nu": nu} for nu in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]],
    "SAM": [{"contamination": c} for c in _FRAC],
}

ALL_METHODS = list(_DEFAULT_PARAMS.keys())

ALL_ALGORITHMS = [
    "minimum distance",
    "maximum likelihood",
    "spectral angle mapping",
    "random forest",
    "support vector machine",
    "multi-layer perceptron",
]

_TAG_FIELDNAMES = ["Run_ID", "Wariant", "Stan", "Raster", "Zakres"]

_PL_FIELDNAMES = [
    "Lp.",
    "Metoda",
    "Algorytm",
    "DA_%",
    "Kappa",
    "Makro_PA_%",
    "Makro_UA_%",
    "Makro_F1_%",
    "Niesklasyfikowane",
    "Usuniete_px",
    "Czas_s",
    "Blad",
]

_PL_PER_CLASS_FIELDNAMES = [
    "Lp.",
    "Metoda",
    "Algorytm",
    "Klasa",
    "PA_%",
    "UA_%",
    "F1_%",
]

_DIR_CLASSIFICATIONS = "classifications"
_DIR_ACCURACY = "accuracy"
_DIR_CSV_PER_ALGO = "csv_per_algorytm"
_DIR_CHARTS_PER_ALGO = "wykresy_per_algorytm"

_FILE_SUMMARY_CSV = "wyniki_zbiorcze.csv"
_FILE_PER_CLASS_CSV = "wyniki_per_klasa.csv"
_FILE_CSV_PER_ALGO = "wyniki_%s.csv"
_FILE_MCNEMAR_CSV = "konfirmacja_mcnemar.csv"

_CHART_HEATMAP = "wykres_mapa_ciepla.png"
_CHART_DELTA_DA = "wykres_zmiana_da.png"
_CHART_PIXELS_VS_DA = "wykres_piksele_vs_zmiana_da.png"
_CHART_TOP_METHODS = "wykres_top_metod.png"
_CHART_PER_CLASS = "wykres_per_klasa.png"

_CHART_ALGO_DA = "wykres_%s_da.png"
_CHART_ALGO_DELTA_DA = "wykres_%s_zmiana_da.png"
_CHART_ALGO_PIXELS_DA = "wykres_%s_piksele_da.png"
_CHART_ALGO_PER_CLASS = "wykres_%s_per_klasa.png"

def _tmp(suffix=""):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    os.remove(path)
    return path

def _shp_to_gpkg(shp_path):
    src_ds = ogr.Open(shp_path)
    if src_ds is None:
        raise ValueError("Cannot open shapefile: %s" % shp_path)

    src_layer = src_ds.GetLayer(0)
    src_defn = src_layer.GetLayerDefn()
    srs = src_layer.GetSpatialRef()
    geom_type = src_layer.GetGeomType()

    tmp_path = _tmp(".gpkg")
    drv = ogr.GetDriverByName("GPKG")
    dst_ds = drv.CreateDataSource(tmp_path)
    if dst_ds is None:
        raise ValueError("Cannot create GPKG: %s" % tmp_path)

    dst_layer = dst_ds.CreateLayer("reference", srs=srs, geom_type=geom_type)

    skip_src_indices = set()
    for i in range(src_defn.GetFieldCount()):
        fld = src_defn.GetFieldDefn(i)
        if fld.GetName().upper() == "FID":
            skip_src_indices.add(i)
            continue
        dst_layer.CreateField(fld)

    dst_defn = dst_layer.GetLayerDefn()

    src_layer.ResetReading()
    for src_feat in src_layer:
        dst_feat = ogr.Feature(dst_defn)
        geom = src_feat.GetGeometryRef()
        if geom is not None:
            dst_feat.SetGeometry(geom.Clone())
        dst_fi = 0
        for src_fi in range(src_defn.GetFieldCount()):
            if src_fi in skip_src_indices:
                continue
            dst_feat.SetField(dst_fi, src_feat.GetField(src_fi))
            dst_fi += 1
        dst_layer.CreateFeature(dst_feat)

    dst_ds.FlushCache()
    dst_ds = None
    src_ds = None

    check = ogr.Open(tmp_path)
    if check is None:
        raise ValueError("Converted GPKG is not readable: %s" % tmp_path)
    check = None

    return tmp_path

def _update_roi_geometry(geometry_file, sig_id, clean_mask, geotransform, projection):
    union_geom = mask_to_union_geometry(clean_mask, geotransform, projection)
    if union_geom is None:
        return

    ds_geom = ogr.Open(geometry_file, 1)
    if ds_geom is None:
        return
    layer = ds_geom.GetLayer()
    layer.ResetReading()
    for feat in layer:
        if feat.GetField("roi_id") == sig_id:
            feat.SetGeometry(union_geom)
            layer.SetFeature(feat)
            break
    ds_geom.FlushCache()
    ds_geom = None

_ROI_STACK_CACHE = {}
_CATALOG_STACK_CACHE = {}

def _covariance_singular_reason(pixels):
    bands, n = pixels.shape
    if n <= bands + 1:
        return "too few pixels (n=%d <= bands+1=%d)" % (n, bands + 1)
    try:
        cov_matrix = np.ma.cov(np.ma.masked_invalid(pixels), ddof=1)
        inv = np.linalg.inv(cov_matrix)
        if np.isnan(inv[0, 0]):
            return "covariance inverse contains NaN"
    except Exception as err:
        return "%s" % err
    return None

def _log_ml_diagnostic(sig_id, mc_id, cl_id, pixels, stage):
    reason = _covariance_singular_reason(pixels)
    if reason is not None:
        log.warning(
            "[ML-diag] sig=%s macroclass=%s class=%s (%s, n=%d px): %s "
            "— Maximum Likelihood will skip this signature",
            sig_id, mc_id, cl_id, stage, pixels.shape[1], reason,
        )

def _log_ml_baseline_diagnostics(catalog, band_paths):
    geometry_file = catalog.geometry_file
    ds_read = ogr.Open(geometry_file)
    if ds_read is None:
        return
    layer = ds_read.GetLayer()
    srs = layer.GetSpatialRef()
    tbl = catalog.table
    ogr_driver = ogr.GetDriverByName("GPKG")
    features = []
    for feat in layer:
        sig_id = feat.GetField("roi_id")
        row_mask = tbl["signature_id"] == sig_id
        if not row_mask.any() or tbl[row_mask]["geometry"][0] != 1:
            continue
        geom = feat.GetGeometryRef()
        if geom is not None:
            features.append(
                (sig_id, feat.GetField("macroclass_id"),
                 feat.GetField("class_id"), geom.Clone())
            )
    ds_read = None
    for sig_id, mc_id, cl_id, geom in features:
        cached = _ensure_cached_stack(sig_id, geom, srs, band_paths, ogr_driver)
        if cached is None:
            continue
        stack = cached[0]
        valid = np.all(~np.isnan(stack), axis=0)
        _log_ml_diagnostic(sig_id, mc_id, cl_id, stack[:, valid], "baseline")

def _ensure_cached_stack(sig_id, geom, srs, band_paths, ogr_driver):
    cached = _ROI_STACK_CACHE.get(sig_id)
    if cached is not None:
        return cached
    roi_gpkg = _tmp(".gpkg")
    roi_ds = ogr_driver.CreateDataSource(roi_gpkg)
    roi_layer = roi_ds.CreateLayer(
        "roi", srs=srs, geom_type=ogr.wkbMultiPolygon
    )
    feat = ogr.Feature(roi_layer.GetLayerDefn())
    feat.SetGeometry(ogr.ForceToMultiPolygon(geom))
    roi_layer.CreateFeature(feat)
    roi_ds.FlushCache()
    roi_ds = None

    ds_raster = _warp_multiband_to_memory(band_paths, roi_gpkg)
    try:
        os.remove(roi_gpkg)
    except OSError:
        pass

    if ds_raster is None:
        log.warning("Nie udało się wyciąć obszaru sig_id=%s, pominięto", sig_id)
        return None

    cached = (
        stack_from_dataset(ds_raster),
        ds_raster.GetGeoTransform(),
        ds_raster.GetProjection(),
    )
    _ROI_STACK_CACHE[sig_id] = cached
    return cached

def _apply_removal_to_catalog(catalog, band_paths, steps, use_voting,
                              vote_threshold, scope="roi"):
    modified, _deleted, report = _apply_removal_shared(
        catalog,
        lambda *_: True,
        steps,
        use_voting,
        vote_threshold,
        scope=scope,
        temp_path_fn=lambda: _tmp(".gpkg"),
        stack_cache=_CATALOG_STACK_CACHE,
        warn_missing=False,
        log_error=log.error,
        log_info=log.debug,
    )
    return modified, report

def _load_fresh_catalog(rs, scpx_path, bandset):
    catalog = rs.spectral_signatures_catalog(bandset=bandset)
    catalog.load(file_path=scpx_path)
    return catalog

_PREPARED_SCPX = {}

def _scpx_path(sig_path, image_path, output_dir):
    directory = os.path.join(output_dir, "sygnatury")
    os.makedirs(directory, exist_ok=True)
    roi = os.path.splitext(os.path.basename(sig_path))[0]
    raster = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(directory, "%s__%s.scpx" % (roi, raster))

def _prepare_signatures_scpx(rs, bandset, sig_path, image_path, output_dir):
    ext = os.path.splitext(sig_path)[1].lower()
    if ext == ".scpx":
        return sig_path
    key = (os.path.abspath(sig_path), os.path.abspath(image_path))
    cached = _PREPARED_SCPX.get(key)
    if cached and os.path.exists(cached):
        return cached
    print("  Importuję ROI %s i liczę sygnatury z rastra ..."
          % os.path.basename(sig_path))
    catalog = rs.spectral_signatures_catalog(bandset=bandset)
    catalog.import_vector(
        file_path=sig_path,
        macroclass_field=ROI_MACROCLASS_FIELD,
        class_field=ROI_CLASS_FIELD,
        macroclass_name_field=ROI_CLASS_NAME_FIELD,
        class_name_field=ROI_CLASS_NAME_FIELD,
        calculate_signature=True,
    )
    if catalog.table is None or len(catalog.table) == 0:
        sys.exit("BŁĄD: import obszarów %s nie dał żadnych sygnatur "
                 "(sprawdź pola macroclass/class_id/c_name)." % sig_path)
    out = _scpx_path(sig_path, image_path, output_dir)
    catalog.save(out)
    _PREPARED_SCPX[key] = out
    print("  => %d sygnatur; katalog zapisany w %s"
          % (len(catalog.table), out))
    return out

def _build_cleaned_catalog(
    rs, scpx_path, bandset, steps, use_voting, vote_threshold, scope="roi",
):
    catalog = _load_fresh_catalog(rs, scpx_path, bandset)
    band_paths = bandset.get_absolute_paths()
    _, report = _apply_removal_to_catalog(
        catalog, band_paths, steps, use_voting, vote_threshold, scope=scope
    )
    total_removed = sum(r["removed_pixels"] for r in report)
    return catalog, total_removed

def _parse_rf_max_features(value):
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("none", ""):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v

def _classify(rs, bandset_catalog, catalog, out_path, algorithm, macroclass):
    return rs.band_classification(
        input_bands=bandset_catalog.get(1),
        output_path=out_path,
        spectral_signatures=catalog,
        macroclass=macroclass,
        algorithm_name=algorithm,
        bandset_catalog=bandset_catalog,
        threshold=False,
        signature_raster=False,
        cross_validation=False,
        input_normalization=None,
        class_weight=None,
        find_best_estimator=False,
        rf_number_trees=_g_rf_number_trees,
        rf_min_samples_split=2,
        rf_max_features=_g_rf_max_features,
        svm_c=1.0,
        svm_gamma="scale",
        svm_kernel="rbf",
        mlp_training_portion=0.9,
        mlp_hidden_layer_sizes=[100],
        mlp_alpha=0.01,
        mlp_learning_rate_init=0.001,
        mlp_max_iter=200,
        mlp_batch_size="auto",
        mlp_activation="relu",
        classification_confidence=False,
    )

_REF_POINTS_CACHE = {}

def _load_reference_points(reference_path, ref_field):
    key = (reference_path, ref_field)
    if key in _REF_POINTS_CACHE:
        return _REF_POINTS_CACHE[key]
    ds = ogr.Open(reference_path)
    if ds is None:
        raise ValueError("Cannot open reference: %s" % reference_path)
    layer = ds.GetLayer()
    if layer.FindFieldIndex(ref_field, 1) < 0:
        raise ValueError(
            "Reference field '%s' not found in %s" % (ref_field, reference_path)
        )
    points = []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        centroid = geom.Centroid()
        points.append(
            (feat.GetFID(), int(feat.GetField(ref_field)),
             centroid.GetX(), centroid.GetY())
        )
    ds = None
    points.sort()
    if not points:
        raise ValueError("Reference has no usable features: %s" % reference_path)
    _REF_POINTS_CACHE[key] = points
    return points

def _extract_predictions(classification_path, points):
    ds = gdal.Open(classification_path)
    if ds is None:
        raise ValueError("Cannot open classification: %s" % classification_path)
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    nodata = band.GetNoDataValue()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize

    pred = np.zeros(len(points), dtype=np.int64)
    for i, (_fid, _ref_cls, x, y) in enumerate(points):
        col = int((x - gt[0]) / gt[1])
        row = int((y - gt[3]) / gt[5])
        if not (0 <= col < xsize and 0 <= row < ysize):
            continue
        value = band.ReadAsArray(col, row, 1, 1)
        if value is None:
            continue
        v = float(value[0][0])
        if nodata is not None and v == nodata:
            continue
        if v > 0:
            pred[i] = int(v)
    ds = None
    return pred

def _pixel_accuracy(ref, pred):
    ref = np.asarray(ref)
    pred = np.asarray(pred)
    n = int(ref.size)
    ref_classes = sorted(set(int(c) for c in ref))
    labels = sorted(set(ref_classes) | set(int(c) for c in pred))

    matrix = {
        (int(i), int(j)): int(np.sum((pred == i) & (ref == j)))
        for i in labels
        for j in labels
    }
    diag = sum(matrix[(k, k)] for k in labels)
    oa = 100.0 * diag / n

    p_o = diag / n
    p_e = sum(
        (int(np.sum(pred == k)) * int(np.sum(ref == k))) for k in labels
    ) / float(n * n)
    kappa = (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else None

    per_class = []
    for k in ref_classes:
        col_sum = int(np.sum(ref == k))
        row_sum = int(np.sum(pred == k))
        correct = matrix[(k, k)]
        pa = 100.0 * correct / col_sum if col_sum else None
        ua = 100.0 * correct / row_sum if row_sum else None
        f1 = (
            2.0 * pa * ua / (pa + ua)
            if pa is not None and ua is not None and (pa + ua) > 0
            else None
        )
        per_class.append({"class": k, "pa": pa, "ua": ua, "f1": f1})

    def _macro(key):
        vals = [pc[key] for pc in per_class if pc[key] is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "oa": oa,
        "kappa": kappa,
        "per_class": per_class,
        "macro_pa": _macro("pa"),
        "macro_ua": _macro("ua"),
        "macro_f1": _macro("f1"),
        "n": n,
        "n_unclassified": int(np.sum(pred == 0)),
        "matrix": [(i, j, c) for (i, j), c in sorted(matrix.items()) if c > 0],
    }

def _write_matrix_csv(path, matrix_entries, ref_classes):
    labels = sorted(
        set(i for i, _j, _c in matrix_entries)
        | set(j for _i, j, _c in matrix_entries)
        | set(ref_classes)
    )
    counts = {(i, j): c for i, j, c in matrix_entries}
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["mapa\\referencja"] + labels)
        for i in labels:
            writer.writerow([i] + [counts.get((i, j), 0) for j in labels])

def _write_predictions_csv(path, points, pred):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["fid", "referencja", "mapa"])
        for (fid, ref_cls, _x, _y), p in zip(points, pred):
            writer.writerow([fid, ref_cls, int(p)])

def _mcnemar(correct_a, correct_b):
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return b, c, 1.0, "brak punktów spornych"
    if b + c < 25:
        try:
            from scipy.stats import binomtest
            p = float(binomtest(min(b, c), b + c, 0.5).pvalue)
        except ImportError:
            from scipy.stats import binom_test
            p = float(binom_test(min(b, c), b + c, 0.5))
        return b, c, p, "dwumianowy"
    from scipy.stats import chi2 as _chi2_dist

    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    p = float(_chi2_dist.sf(stat, 1))
    return b, c, p, "chi2 (poprawka na ciągłość)"

def _holm(p_values):
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted

_AUTOTEST_DIAGONAL = [260, 68, 73, 73, 53, 85, 66]
_AUTOTEST_OA = 74.92
_AUTOTEST_KAPPA = 0.6890
_AUTOTEST_PA = [74.1, 70.8, 76.0, 86.9, 65.4, 95.5, 61.1]
_AUTOTEST_UA = [91.5, 70.1, 98.6, 64.6, 73.6, 97.7, 37.1]

def _run_autotest(classification_path, reference_path, ref_field):
    print("AUTOTEST procedury oceny dokładności")
    print("  klasyfikacja: %s" % classification_path)
    print("  referencja:   %s (pole '%s')" % (reference_path, ref_field))
    points = _load_reference_points(reference_path, ref_field)
    pred = _extract_predictions(classification_path, points)
    ref = np.array([p[1] for p in points])
    acc = _pixel_accuracy(ref, pred)

    ref_classes = sorted(set(int(c) for c in ref))
    diag = [
        next(c for i, j, c in acc["matrix"] if i == k and j == k)
        if any(i == k and j == k for i, j, _c in acc["matrix"])
        else 0
        for k in ref_classes
    ]
    pa = [pc["pa"] for pc in acc["per_class"]]
    ua = [pc["ua"] for pc in acc["per_class"]]

    failures = []
    if diag != _AUTOTEST_DIAGONAL:
        failures.append("przekątna %s != %s" % (diag, _AUTOTEST_DIAGONAL))
    if abs(acc["oa"] - _AUTOTEST_OA) > 0.1:
        failures.append("OA %.4f != %.2f" % (acc["oa"], _AUTOTEST_OA))
    if abs((acc["kappa"] or 0) - _AUTOTEST_KAPPA) > 0.001:
        failures.append("kappa %.4f != %.4f" % (acc["kappa"], _AUTOTEST_KAPPA))
    for name, got, expected in (("PA", pa, _AUTOTEST_PA), ("UA", ua, _AUTOTEST_UA)):
        for k, (g, e) in enumerate(zip(got, expected)):
            if g is None or abs(g - e) > 0.1:
                failures.append(
                    "%s klasa %d: %s != %.1f"
                    % (name, ref_classes[k], "%.2f" % g if g is not None else "-", e)
                )

    print("  n=%d  niesklasyfikowane=%d" % (acc["n"], acc["n_unclassified"]))
    print("  OA=%.4f%%  kappa=%.4f" % (acc["oa"], acc["kappa"]))
    print("  przekątna: %s" % diag)
    if failures:
        print("\nAUTOTEST: NIEZALICZONY")
        for msg in failures:
            print("  - %s" % msg)
        return 1
    print("\nAUTOTEST: ZALICZONY (wszystkie wartości zgodne z kontrolnymi)")
    return 0

def _params_label(params):
    return ",".join("%s=%s" % (k, v) for k, v in params.items())

def _build_configs(max_k, methods, grid_search=False):
    defaults = _DEFAULT_PARAMS

    for m in methods:
        param_list = (
            _PARAM_GRID.get(m, [defaults[m]])
            if grid_search
            else [defaults[m]]
        )
        for params in param_list:
            label = "%s(%s)" % (m, _params_label(params)) if grid_search else m
            yield (label, [(m, params)], False, 1)

    for k in range(2, max_k + 1):
        for combo in itertools.permutations(methods, k):
            steps = [(m, defaults[m]) for m in combo]
            yield (" → ".join(combo), steps, False, 1)

        vote_thr = (k + 1) // 2
        for combo in itertools.combinations(methods, k):
            steps = [(m, defaults[m]) for m in combo]
            yield ("Ensemble[%s]" % ", ".join(combo), steps, True, vote_thr)

_LABEL_MAP_CACHE = None

def _sanitize_label(label):
    return re.sub(r"[^\w\-]", "_", label)[:80]

def _label_map():
    global _LABEL_MAP_CACHE
    if _LABEL_MAP_CACHE is None:
        m = {}
        for grid in (False, True):
            for label, *_ in _build_configs(2, ALL_METHODS, grid_search=grid):
                m[_sanitize_label(label)] = label
        for a, b in itertools.permutations(ALL_METHODS, 2):
            lbl = "Ensemble[%s, %s]" % (a, b)
            m[_sanitize_label(lbl)] = lbl
        _LABEL_MAP_CACHE = m
    return _LABEL_MAP_CACHE

def _rename_pca(label):
    return re.sub(r"\bPCA\b(?!\s+(?:Reconstruction|Mahalanobis))",
                  "PCA Mahalanobis", label)

def _method_display(label):
    if label in ("Baseline", "Poziom bazowy"):
        return label
    lm = _label_map()
    orig = (lm.get(label) or lm.get(_sanitize_label(label))
            or label.replace("_", " "))
    return _rename_pca(orig)

def _to_pl_row(rank, row):
    return {
        "Lp.": rank,
        "Metoda": "Poziom bazowy" if row["label"] == "Baseline"
                  else _method_display(row["label"]),
        "Algorytm": row.get("algorithm", ""),
        "DA_%": ("%.4f" % row["oa"]) if row.get("oa") is not None else "",
        "Kappa": ("%.4f" % row["kappa"]) if row.get("kappa") is not None else "",
        "Makro_PA_%": _fmt_metric(row.get("macro_pa")),
        "Makro_UA_%": _fmt_metric(row.get("macro_ua")),
        "Makro_F1_%": _fmt_metric(row.get("macro_f1")),
        "Niesklasyfikowane": (
            row.get("n_unclassified") if row.get("n_unclassified") is not None else ""
        ),
        "Usuniete_px": row.get("total_removed", ""),
        "Czas_s": row.get("time_s", ""),
        "Blad": row.get("error", ""),
    }

def _write_polish_csv(path, valid_rows, invalid_rows, tags=None):
    tags = tags or {}
    fieldnames = _TAG_FIELDNAMES + _PL_FIELDNAMES
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(valid_rows, 1):
            writer.writerow({**tags, **_to_pl_row(rank, row)})
        for row in invalid_rows:
            writer.writerow({**tags, **_to_pl_row("—", row)})

def _fmt_metric(value):
    return ("%.4f" % value) if value is not None else ""

def _write_per_class_csv(path, valid_rows, tags=None):
    tags = tags or {}
    fieldnames = _TAG_FIELDNAMES + _PL_PER_CLASS_FIELDNAMES
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(valid_rows, 1):
            method = (
                "Poziom bazowy" if row["label"] == "Baseline"
                else _method_display(row["label"])
            )
            for pc in row.get("per_class") or []:
                writer.writerow(
                    {
                        **tags,
                        "Lp.": rank,
                        "Metoda": method,
                        "Algorytm": row.get("algorithm", ""),
                        "Klasa": pc.get("class", ""),
                        "PA_%": _fmt_metric(pc.get("pa")),
                        "UA_%": _fmt_metric(pc.get("ua")),
                        "F1_%": _fmt_metric(pc.get("f1")),
                    }
                )

def _generate_charts(results, algorithms, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("UWAGA: brak biblioteki matplotlib, pomijam generowanie wykresów.")
        return

    valid = [r for r in results if r.get("oa") is not None]
    if not valid:
        print("  No valid results — skipping chart generation.")
        return

    labels = list(dict.fromkeys(r["label"] for r in results))

    baseline_oa_by_algo = {
        r["algorithm"]: r["oa"] for r in valid if r["label"] == "Baseline"
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
        }
    )

    oa_matrix = []
    for lbl in labels:
        row = []
        for algo in algorithms:
            m = next(
                (r for r in valid if r["label"] == lbl and r["algorithm"] == algo), None
            )
            row.append(m["oa"] if m else float("nan"))
        oa_matrix.append(row)

    oa_arr = np.array(oa_matrix, dtype=float)

    fig_h = max(5, len(labels) * 0.38 + 2)
    fig_w = max(6, len(algorithms) * 1.6 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmin, vmax = np.nanmin(oa_arr), np.nanmax(oa_arr)
    im = ax.imshow(oa_arr, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Dokładność Ogólna [%]")

    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(
        "Dokładność Ogólna [%] — metoda usuwania wartości odstających × algorytm klasyfikacji",
        pad=10,
    )
    ax.set_xlabel("Algorytm klasyfikacji")
    ax.set_ylabel("Metoda usuwania wartości odstających")

    span = vmax - vmin if vmax > vmin else 1.0
    for i in range(len(labels)):
        for j in range(len(algorithms)):
            val = oa_arr[i, j]
            if not np.isnan(val):
                brightness = (val - vmin) / span
                color = "black" if 0.25 < brightness < 0.75 else "white"
                ax.text(
                    j,
                    i,
                    "%.1f" % val,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )

    plt.tight_layout()
    path = str(out_dir / _CHART_HEATMAP)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Heatmap: %s" % path)

    non_base_labels = [l for l in labels if l != "Baseline"]
    if non_base_labels and baseline_oa_by_algo:
        n_methods = len(non_base_labels)
        n_algos = len(algorithms)
        fig_w2 = max(6, n_algos * max(3.5, n_methods * 0.25 + 1.5))
        fig, axes = plt.subplots(
            1, n_algos, figsize=(fig_w2, max(5, n_methods * 0.4 + 2)), sharey=True
        )
        if n_algos == 1:
            axes = [axes]

        for ax, algo in zip(axes, algorithms):
            base = baseline_oa_by_algo.get(algo)
            deltas = []
            for lbl in non_base_labels:
                m = next(
                    (r for r in valid if r["label"] == lbl and r["algorithm"] == algo),
                    None,
                )
                deltas.append(
                    (m["oa"] - base) if (m and base is not None) else float("nan")
                )

            colors = [
                "#2ca02c" if (not np.isnan(d) and d > 0) else "#d62728" for d in deltas
            ]
            y_pos = range(n_methods)
            ax.barh(y_pos, deltas, color=colors, edgecolor="white", height=0.65)
            ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(non_base_labels)
            ax.set_title(algo, fontsize=8)
            ax.set_xlabel("ΔDA [pp]", fontsize=8)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)

        fig.suptitle(
            "Zmiana Dokładności Ogólnej względem poziomu bazowego [punkty procentowe]",
            fontsize=10,
            y=1.01,
        )
        plt.tight_layout()
        path = str(out_dir / _CHART_DELTA_DA)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print("  Delta OA: %s" % path)

    scatter_pts = []
    for r in valid:
        if r["label"] == "Baseline":
            continue
        base = baseline_oa_by_algo.get(r["algorithm"])
        removed = r.get("total_removed")
        if base is None or removed is None:
            continue
        scatter_pts.append(
            {
                "label": r["label"],
                "algorithm": r["algorithm"],
                "removed": removed,
                "delta": r["oa"] - base,
            }
        )

    cmap = plt.cm.get_cmap("tab10", len(algorithms))
    algo_color = {a: cmap(i) for i, a in enumerate(algorithms)}

    if scatter_pts:
        fig, ax = plt.subplots(figsize=(8, 5))
        for pt in scatter_pts:
            ax.scatter(
                pt["removed"],
                pt["delta"],
                color=algo_color.get(pt["algorithm"], "gray"),
                alpha=0.75,
                s=55,
                edgecolors="white",
                linewidths=0.4,
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        legend_handles = [
            mpatches.Patch(color=algo_color[a], label=a) for a in algorithms
        ]
        ax.legend(
            handles=legend_handles,
            fontsize=8,
            title="Algorytm klasyfikacji",
            title_fontsize=9,
            loc="best",
        )
        ax.set_xlabel("Liczba usuniętych pikseli wartości odstających")
        ax.set_ylabel("ΔDA [pp]")
        ax.set_title(
            "Agresywność usuwania wartości odstających a zmiana Dokładności Ogólnej"
        )
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = str(out_dir / _CHART_PIXELS_VS_DA)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print("  Pixels vs delta OA: %s" % path)

    if non_base_labels and baseline_oa_by_algo:
        n_top = min(10, len(non_base_labels))
        top_labels_per_algo = {}
        for algo in algorithms:
            ranked = sorted(
                [
                    (
                        lbl,
                        next(
                            (
                                r["oa"]
                                for r in valid
                                if r["label"] == lbl and r["algorithm"] == algo
                            ),
                            None,
                        ),
                    )
                    for lbl in non_base_labels
                ],
                key=lambda x: -(x[1] or 0),
            )
            top_labels_per_algo[algo] = [lbl for lbl, _ in ranked[:n_top]]

        all_top = list(
            dict.fromkeys(l for lbls in top_labels_per_algo.values() for l in lbls)
        )
        x = np.arange(len(all_top))
        width = 0.8 / max(len(algorithms), 1)
        offsets = np.linspace(
            -(len(algorithms) - 1) * width / 2,
            (len(algorithms) - 1) * width / 2,
            len(algorithms),
        )

        fig, ax = plt.subplots(figsize=(max(10, len(all_top) * 0.7 + 2), 5))
        cmap2 = plt.cm.get_cmap("tab10", len(algorithms))
        for i, (algo, offset) in enumerate(zip(algorithms, offsets)):
            oas = []
            for lbl in all_top:
                m = next(
                    (
                        r["oa"]
                        for r in valid
                        if r["label"] == lbl and r["algorithm"] == algo
                    ),
                    None,
                )
                oas.append(m if m is not None else 0)
            ax.bar(
                x + offset,
                oas,
                width=width * 0.9,
                label=algo,
                color=cmap2(i),
                edgecolor="white",
                linewidth=0.5,
            )

        for algo in algorithms:
            base = baseline_oa_by_algo.get(algo)
            if base is not None:
                ax.axhline(
                    base,
                    linestyle="--",
                    linewidth=0.8,
                    color=algo_color.get(algo, "gray"),
                    alpha=0.6,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(all_top, rotation=30, ha="right")
        ax.set_ylabel("Dokładność Ogólna [%]")
        ax.set_title(
            "Dokładność Ogólna dla najlepszych metod (linia przerywana = poziom bazowy)"
        )
        ax.legend(title="Algorytm klasyfikacji", fontsize=8, title_fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ymin = max(0, np.nanmin([r["oa"] for r in valid]) - 2)
        ax.set_ylim(bottom=ymin)
        plt.tight_layout()
        path = str(out_dir / _CHART_TOP_METHODS)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print("  Top methods: %s" % path)

    def _per_class_chart(best_row, base_row, algo, path):
        per_class = best_row["per_class"]
        classes = [str(pc["class"]) for pc in per_class]
        base_map = (
            {str(pc["class"]): pc for pc in base_row["per_class"]}
            if base_row
            else {}
        )

        def _nan(v):
            return float("nan") if v is None else v

        x = np.arange(len(classes))
        w = 0.38
        panels = (
            ("pa", "PA [%] (dokładność producenta)", (0, 100)),
            ("ua", "UA [%] (dokładność użytkownika)", (0, 100)),
            ("f1", "F1 [%] (per klasa)", (0, 100)),
        )
        fig, axes = plt.subplots(
            1, 3, figsize=(max(13, len(classes) * 1.2 + 4), 5)
        )
        for ax, (key, title, ylim) in zip(axes, panels):
            best_vals = [_nan(pc.get(key)) for pc in per_class]
            if base_row:
                base_vals = [
                    _nan(base_map[c].get(key)) if c in base_map else float("nan")
                    for c in classes
                ]
                ax.bar(
                    x - w / 2, base_vals, w, label="Poziom bazowy",
                    color="#9bb7d4",
                )
                ax.bar(
                    x + w / 2, best_vals, w, label=best_row["label"][:24],
                    color="#d48a6a",
                )
            else:
                ax.bar(
                    x, best_vals, w * 1.6, label=best_row["label"][:24],
                    color="#d48a6a",
                )
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(classes)
            ax.set_xlabel("Klasa")
            ax.set_ylim(*ylim)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Dokładność per klasa — %s" % algo, fontsize=12)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def _baseline_per_class(rows, algo):
        return next(
            (
                r
                for r in rows
                if r["label"] == "Baseline"
                and r["algorithm"] == algo
                and r.get("per_class")
            ),
            None,
        )

    best_pc = next(
        (r for r in valid if r["label"] != "Baseline" and r.get("per_class")),
        None,
    )
    if best_pc is not None:
        path = str(out_dir / _CHART_PER_CLASS)
        _per_class_chart(
            best_pc, _baseline_per_class(valid, best_pc["algorithm"]),
            best_pc["algorithm"], path,
        )
        print("  Dokładność w podziale na klasy: %s" % path)

    per_algo_dir = out_dir / _DIR_CHARTS_PER_ALGO
    per_algo_dir.mkdir(exist_ok=True)

    for algo in algorithms:
        algo_results = [r for r in valid if r["algorithm"] == algo]
        if not algo_results:
            continue

        base_oa = next(
            (r["oa"] for r in algo_results if r["label"] == "Baseline"), None
        )
        non_base = sorted(
            [r for r in algo_results if r["label"] != "Baseline"],
            key=lambda r: -(r["oa"] or 0),
        )
        if not non_base:
            continue

        algo_best_pc = next((r for r in non_base if r.get("per_class")), None)
        if algo_best_pc is not None:
            pc_path = str(
                per_algo_dir
                / (_CHART_ALGO_PER_CLASS % re.sub(r"[^\w\-]", "_", algo))
            )
            _per_class_chart(
                algo_best_pc, _baseline_per_class(algo_results, algo),
                algo, pc_path,
            )

        method_labels = [r["label"] for r in non_base]
        oas = [r["oa"] or 0 for r in non_base]
        deltas = [
            (
                (r["oa"] - base_oa)
                if (r["oa"] is not None and base_oa is not None)
                else float("nan")
            )
            for r in non_base
        ]
        removed = [r.get("total_removed") or 0 for r in non_base]

        algo_safe = re.sub(r"[^\w\-]", "_", algo)
        x = np.arange(len(method_labels))
        ymin_global = max(0, min(oas + ([base_oa] if base_oa else [])) - 3)

        fig, ax = plt.subplots(figsize=(max(8, len(method_labels) * 0.65 + 2), 5))
        bar_colors = ["#1f77b4"] * len(method_labels)
        ax.bar(x, oas, color=bar_colors, edgecolor="white", width=0.7)
        if base_oa is not None:
            ax.axhline(
                base_oa,
                color="#d62728",
                linewidth=1.5,
                linestyle="--",
                label="Poziom bazowy: %.2f%%" % base_oa,
            )
            ax.legend(fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=35, ha="right")
        ax.set_ylabel("Dokładność Ogólna [%]")
        ax.set_title("Dokładność Ogólna — algorytm: %s" % algo)
        ax.set_ylim(bottom=ymin_global)
        ax.grid(axis="y", alpha=0.3)
        for xi, oa_val in zip(x, oas):
            ax.text(
                xi, oa_val + 0.05, "%.2f" % oa_val, ha="center", va="bottom", fontsize=7
            )
        plt.tight_layout()
        path = str(per_algo_dir / (_CHART_ALGO_DA % algo_safe))
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(max(8, len(method_labels) * 0.65 + 2), 5))
        delta_colors = [
            "#2ca02c" if (not np.isnan(d) and d > 0) else "#d62728" for d in deltas
        ]
        ax.bar(x, deltas, color=delta_colors, edgecolor="white", width=0.7)
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=35, ha="right")
        ax.set_ylabel("ΔDA [pp]")
        ax.set_title("Zmiana DA względem poziomu bazowego — algorytm: %s" % algo)
        ax.grid(axis="y", alpha=0.3)
        for xi, d in zip(x, deltas):
            if not np.isnan(d):
                va = "bottom" if d >= 0 else "top"
                ax.text(xi, d, "%+.2f" % d, ha="center", va=va, fontsize=7)
        plt.tight_layout()
        path = str(per_algo_dir / (_CHART_ALGO_DELTA_DA % algo_safe))
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        if any(r > 0 for r in removed):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sc = ax.scatter(
                removed,
                oas,
                c=deltas,
                cmap="RdYlGn",
                s=70,
                edgecolors="white",
                linewidths=0.5,
                vmin=min(d for d in deltas if not np.isnan(d)),
                vmax=max(d for d in deltas if not np.isnan(d)),
            )
            plt.colorbar(sc, ax=ax, label="ΔDA [pp]")
            if base_oa is not None:
                ax.axhline(
                    base_oa,
                    color="#d62728",
                    linewidth=1.2,
                    linestyle="--",
                    label="Poziom bazowy",
                )
                ax.legend(fontsize=8)
            ax.set_xlabel("Liczba usuniętych pikseli")
            ax.set_ylabel("Dokładność Ogólna [%]")
            ax.set_title("Usunięte piksele a DA — algorytm: %s" % algo)
            ax.set_ylim(bottom=ymin_global)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            path = str(per_algo_dir / (_CHART_ALGO_PIXELS_DA % algo_safe))
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()

    print("  Wykresy wg algorytmów: %s" % per_algo_dir)

class _NoDaemonProcess(multiprocessing.Process):

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class _NoDaemonPool(multiprocessing.pool.Pool):

    @staticmethod
    def Process(ctx, *args, **kwargs):
        proc = ctx.Process(*args, **kwargs)
        proc.__class__ = _NoDaemonProcess
        return proc

_g_rs = None
_g_bandset_catalog = None
_g_bandset = None
_g_scpx_path = None
_g_keep_classifications = False
_g_rf_number_trees = 10
_g_rf_max_features = None

def _set_wavelengths(bandset_catalog, bandset_number=1):
    band_count = bandset_catalog.get(bandset_number).get_band_count()
    wavelengths = WAVELENGTHS.get(band_count)
    if wavelengths is None:
        log.warning(
            "Nie ustawiono długości fal: nieznana liczba kanałów (%s). "
            "Uzupełnij słownik WAVELENGTHS.", band_count
        )
        return
    try:
        bandset_catalog.set_wavelength(
            wavelength_list=wavelengths, unit=WAVELENGTH_UNIT,
            bandset_number=bandset_number,
        )
    except Exception as err:
        log.warning("Nie ustawiono długości fal (%d kanałów): %s",
                    band_count, err)

def _worker_init(
    image_path, scpx_path, rs_n_processes, rs_ram, keep_classifications=False,
    rf_number_trees=10, rf_max_features=None,
):
    global _g_rs, _g_bandset_catalog, _g_bandset, _g_scpx_path
    global _g_keep_classifications, _g_rf_number_trees, _g_rf_max_features
    _g_scpx_path = scpx_path
    _g_keep_classifications = keep_classifications
    _g_rf_number_trees = rf_number_trees
    _g_rf_max_features = rf_max_features
    _g_rs = remotior_sensus.Session(n_processes=rs_n_processes, available_ram=rs_ram)
    _g_bandset_catalog = _g_rs.bandset_catalog()
    _g_bandset_catalog.create_bandset(paths=image_path, bandset_number=1)
    _set_wavelengths(_g_bandset_catalog)
    _g_bandset = _g_bandset_catalog.get(1)

def _run_one_config(task):
    (
        label,
        steps,
        use_voting,
        vote_threshold,
        reference_path,
        ref_field,
        algorithms,
        macroclass,
        scope,
        class_dir_str,
        accuracy_dir_str,
    ) = task

    rs = _g_rs
    bandset_catalog = _g_bandset_catalog
    bandset = _g_bandset
    scpx_path = _g_scpx_path

    if rs is None:
        return [
            {
                "label": label,
                "algorithm": algo,
                "oa": None,
                "kappa": None,
                "total_removed": None,
                "time_s": 0.0,
                "error": "worker not initialized",
            }
            for algo in algorithms
        ]

    class_dir = Path(class_dir_str)
    accuracy_dir = Path(accuracy_dir_str)
    safe_label = re.sub(r"[^\w\-]", "_", label)[:80]
    results = []
    t0 = time.time()

    total_removed = 0
    try:
        if steps:
            catalog, total_removed = _build_cleaned_catalog(
                rs,
                scpx_path,
                bandset,
                steps,
                use_voting,
                vote_threshold,
                scope=scope,
            )
        else:
            catalog = _load_fresh_catalog(rs, scpx_path, bandset)
            if any("likelihood" in a for a in algorithms):
                _log_ml_baseline_diagnostics(
                    catalog, bandset.get_absolute_paths()
                )
    except Exception as err:
        for algorithm in algorithms:
            results.append(
                {
                    "label": label,
                    "algorithm": algorithm,
                    "oa": None,
                    "kappa": None,
                    "total_removed": None,
                    "time_s": round(time.time() - t0, 1),
                    "error": str(err),
                }
            )
        return results

    for algorithm in algorithms:
        algo_safe = re.sub(r"[^\w\-]", "_", algorithm)
        file_key = "%s__%s" % (safe_label, algo_safe)
        t0_algo = time.time()

        class_path = str(class_dir / ("%s.tif" % file_key))
        if os.path.exists(class_path):
            os.remove(class_path)
        try:
            output = _classify(
                rs, bandset_catalog, catalog, class_path, algorithm, macroclass
            )
        except Exception as err:
            results.append(
                {
                    "label": label,
                    "algorithm": algorithm,
                    "oa": None,
                    "kappa": None,
                    "total_removed": total_removed,
                    "time_s": round(time.time() - t0_algo, 1),
                    "error": str(err),
                }
            )
            continue

        if output is None or not output.check:
            results.append(
                {
                    "label": label,
                    "algorithm": algorithm,
                    "oa": None,
                    "kappa": None,
                    "total_removed": total_removed,
                    "time_s": round(time.time() - t0_algo, 1),
                    "error": "classification check=False",
                }
            )
            continue

        actual_class_path = class_path
        try:
            if hasattr(output, "paths") and output.paths:
                actual_class_path = output.paths[0]
        except (IndexError, TypeError):
            pass
        if not os.path.exists(actual_class_path):
            results.append(
                {
                    "label": label,
                    "algorithm": algorithm,
                    "oa": None,
                    "kappa": None,
                    "total_removed": total_removed,
                    "time_s": round(time.time() - t0_algo, 1),
                    "error": "output file not found",
                }
            )
            continue

        try:
            points = _load_reference_points(reference_path, ref_field)
            pred = _extract_predictions(actual_class_path, points)
            ref = np.array([p[1] for p in points])
            acc = _pixel_accuracy(ref, pred)
        except Exception as err:
            results.append(
                {
                    "label": label,
                    "algorithm": algorithm,
                    "oa": None,
                    "kappa": None,
                    "total_removed": total_removed,
                    "time_s": round(time.time() - t0_algo, 1),
                    "error": str(err),
                }
            )
            continue

        _write_predictions_csv(
            str(accuracy_dir / ("%s_pred.csv" % file_key)), points, pred
        )
        _write_matrix_csv(
            str(accuracy_dir / ("%s_macierz.csv" % file_key)),
            acc["matrix"],
            sorted(set(int(c) for c in ref)),
        )

        if not _g_keep_classifications:
            for _p in (class_path, actual_class_path):
                try:
                    os.remove(_p)
                except OSError:
                    pass

        results.append(
            {
                "label": label,
                "algorithm": algorithm,
                "oa": acc["oa"],
                "kappa": acc["kappa"],
                "per_class": acc["per_class"],
                "macro_pa": acc["macro_pa"],
                "macro_ua": acc["macro_ua"],
                "macro_f1": acc["macro_f1"],
                "n_unclassified": acc["n_unclassified"],
                "matrix": acc["matrix"],
                "pred": [int(v) for v in pred],
                "total_removed": total_removed,
                "time_s": round(time.time() - t0_algo, 1),
                "error": "",
            }
        )

    return results

def _build_run_matrix():
    cells = []
    for variant, vcfg in VARIANTS.items():
        for state in STATES_TO_RUN:
            signatures = vcfg.get("roi", {}).get(state)
            if not signatures:
                continue
            for raster_tag in RASTERS_TO_RUN:
                for scope in SCOPES_TO_RUN:
                    cells.append(
                        {
                            "run_id": "%s_%s_%s_%s"
                            % (variant, state, raster_tag, scope),
                            "variant": variant,
                            "state": state,
                            "raster": raster_tag,
                            "scope": scope,
                            "image": RASTERS.get(raster_tag),
                            "signatures": signatures,
                            "reference": vcfg.get("reference"),
                        }
                    )
    return cells

def _preflight(cells):
    missing = []
    for c in cells:
        for key in ("image", "signatures", "reference"):
            path = c.get(key)
            if not path or not os.path.exists(path):
                missing.append((c["run_id"], key, path))
    return missing

def _run_matrix(args):
    cells = _build_run_matrix()
    if args.only:
        cells = [c for c in cells if c["run_id"] == args.only]
        if not cells:
            sys.exit("BŁĄD: --only '%s' nie pasuje do żadnej komórki." % args.only)
    if not cells:
        sys.exit(
            "BŁĄD: macierz jest pusta, uzupełnij VARIANTS oraz STATES_TO_RUN w bloku CONFIG."
        )

    print("=" * 82)
    print("TRYB MACIERZY — %d przebiegów (parametry z bloku CONFIG)" % len(cells))
    print("=" * 82)
    missing = _preflight(cells)
    missing_ids = {m[0] for m in missing}
    for c in cells:
        flag = "   <-- BRAK PLIKÓW" if c["run_id"] in missing_ids else ""
        print("  %-38s [%s]%s" % (c["run_id"], c["raster"], flag))

    if missing:
        print("\nBrakujące pliki (%d):" % len(missing))
        for run_id, key, path in missing:
            print("  [%s] %s = %s" % (run_id, key, path or "(nie ustawiono w CONFIG)"))

    if args.dry_run:
        print("\n--dry-run: powyżej plan i kontrola plików; nic nie uruchamiam.")
        return
    if missing:
        sys.exit("\nPrzerwano: uzupełnij ścieżki w CONFIG przed uruchomieniem.")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(cells, 1):
        run_dir = out_dir / c["run_id"]
        if SKIP_DONE and not args.force and (run_dir / _FILE_SUMMARY_CSV).exists():
            print("\n[%d/%d] POMINIĘTO (już policzony): %s"
                  % (i, len(cells), c["run_id"]))
            continue
        print("\n" + "#" * 82)
        print("# [%d/%d] PRZEBIEG: %s" % (i, len(cells), c["run_id"]))
        print("#" * 82)
        cell_args = SimpleNamespace(
            image=c["image"], signatures=c["signatures"], reference=c["reference"],
            ref_field=REF_FIELD, output=OUTPUT_DIR,
            algorithm=ALGORITHMS, methods=(METHODS or ALL_METHODS),
            macroclass=MACROCLASS, max_k=MAX_K, grid_search=GRID_SEARCH,
            scope=c["scope"], variant=c["variant"], state=c["state"],
            raster=c["raster"], run_id=c["run_id"],
            n_processes=N_PROCESSES, ram=RAM_MB, workers=WORKERS,
            keep_classifications=KEEP_CLASSIFICATIONS,
            rf_number_trees=RF_NUMBER_TREES,
            rf_max_features=_parse_rf_max_features(RF_MAX_FEATURES),
        )
        try:
            _run_from_args(cell_args)
        except SystemExit:
            raise
        except Exception as err:
            log.error("Przebieg %s przerwany błędem: %s", c["run_id"], err)

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark outlier-removal pipelines for SCP signatures. "
        "Accuracy assessment is pixel-based: "
        "pixel-count confusion matrix from reference-polygon centroids. "
        "Run with no --image/--signatures to execute the full CONFIG matrix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--autotest",
        metavar="CLASSIFICATION_TIF",
        help="Run the control-value check of the accuracy procedure "
        "on the given classification raster (requires --reference with the "
        "referencja_1234567 variant) and exit. No --image/--signatures needed.",
    )
    parser.add_argument(
        "--evaluate-only",
        metavar="DIR",
        help="Accuracy-only mode: skip classification and compute accuracy plus "
        "charts for ready classification rasters in DIR (named "
        "'<label>__<algorithm>.tif', e.g. the classifications/ folder of an "
        "earlier run) against --reference. Lets you test the same result rasters "
        "on different reference data. Needs --reference (and run tags); no "
        "--image/--signatures required.",
    )
    parser.add_argument(
        "--evaluate-results-dir",
        metavar="RESULTS_DIR",
        help="Batch accuracy-only mode: RESULTS_DIR holds per-run subdirectories "
        "(e.g. benchmark_results_v2, each named "
        "<variant>_<state>_<raster>_<scope> with a classifications/ folder). For "
        "EACH run the variant is parsed from the folder name, the matching "
        "reference referencja_<variant>.shp is picked from --reference-dir, and "
        "accuracy plus charts are written in place — no classification is run. "
        "Runs without rasters or without a matching reference are skipped.",
    )
    parser.add_argument(
        "--reference-dir",
        metavar="DIR",
        help="Directory with per-variant reference files referencja_<variant>.shp, "
        "used by --evaluate-results-dir (default: the referencja/warianty folder "
        "from the CONFIG block). Point it elsewhere to test the same result "
        "rasters against a different reference set.",
    )
    parser.add_argument(
        "--image", help="Path to multispectral .tif image"
    )
    parser.add_argument(
        "--signatures", help="Path to SCP training signatures (.scpx)"
    )
    parser.add_argument(
        "--reference",
        help="Reference polygons for accuracy assessment: vector .gpkg or .shp "
        "(SHP is auto-converted to a temporary GPKG). Must be the reference "
        "variant matching the ROI variant used for training "
        "(roi_X <-> referencja_X).",
    )
    parser.add_argument(
        "--ref-field",
        default="klasa",
        help="Field name in reference vector with class values "
        "(default: klasa — as in referencja_* variant shapefiles).",
    )
    parser.add_argument(
        "--output",
        default="./benchmark_results",
        help="Output directory (default: ./benchmark_results)",
    )
    parser.add_argument(
        "--algorithm",
        nargs="+",
        default=None,
        metavar="ALGO",
        choices=ALL_ALGORITHMS,
        help="Classification algorithm(s) to use. If omitted, all %d algorithms "
        "are tested. Options: %s"
        % (len(ALL_ALGORITHMS), ", ".join("'%s'" % a for a in ALL_ALGORITHMS)),
    )
    parser.add_argument(
        "--rf-number-trees",
        type=int,
        default=RF_NUMBER_TREES,
        metavar="N",
        help="Random Forest: number of trees (default: %d = SCP plugin parity). "
        "The plugin default is very low; use e.g. 500 to evaluate RF fairly "
        "(see sub-study D in the header comment)." % RF_NUMBER_TREES,
    )
    parser.add_argument(
        "--rf-max-features",
        default="none",
        metavar="VAL",
        help="Random Forest: features considered per split. 'none' = all bands "
        "(default, SCP plugin parity — causes correlated trees and depresses RF). "
        "Use 'sqrt' or 'log2' (standard RF recommendation), an integer band count, "
        "or a fraction. Default: none.",
    )
    parser.add_argument(
        "--macroclass",
        action="store_true",
        help="Use macroclass classification (default: class-level)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=1,
        metavar="K",
        help="Maximum pipeline length (1..%d, default: 1). "
        "k>1 always uses default params (grid-search applies to k=1 only). "
        "UWAGA: duże k przy wielu metodach daje silnie rosnącą liczbę konfiguracji "
        "(k=2 with 17 methods → ~320, k=3 → ~5440)." % len(ALL_METHODS),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=ALL_METHODS,
        choices=ALL_METHODS,
        metavar="METHOD",
        help="Subset of methods to test (default: all %d)." % len(ALL_METHODS),
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Test multiple parameter values per method (k=1 only). "
        "Increases configurations from ~15 to ~100 for all methods.",
    )
    parser.add_argument(
        "--scope",
        choices=("roi", "class", "macroclass"),
        default="roi",
        help="Detection reference scope (as in the plugin): 'roi' computes "
        "statistics per single ROI, 'class'/'macroclass' pools pixels of all "
        "ROIs sharing the same class/macroclass ID (an ROI outlying as a "
        "whole is then removed from the training set). Default: roi.",
    )
    parser.add_argument(
        "--variant", default="",
        help="Class-variant tag (e.g. 1234567). Written as a column in every "
        "output CSV and used to name the per-run output subdirectory.",
    )
    parser.add_argument(
        "--state", default="",
        help="Training-set state tag (e.g. poprawny / zbledami). Written as a "
        "column in every output CSV.",
    )
    parser.add_argument(
        "--raster", default="",
        help="Raster-level tag (e.g. L1C / L2A). Written as a column in every "
        "output CSV.",
    )
    parser.add_argument(
        "--run-id", default="",
        help="Explicit run identifier used as the per-run output subdirectory "
        "and Run_ID column. If omitted, it is composed from "
        "variant_state_raster_scope.",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=2,
        help="Parallel processes for remotior_sensus (default: 2)",
    )
    parser.add_argument(
        "--ram",
        type=int,
        default=2048,
        help="Available RAM in MB for remotior_sensus (default: 2048)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel worker processes (default: 1 = sequential). "
        "Each worker handles one complete config independently. "
        "RAM and n-processes are divided evenly among workers.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed progress output"
    )
    parser.add_argument(
        "--list-methods", action="store_true",
        help="Print the outlier-detection method registry (category and Polish "
        "description of each method) and exit.",
    )
    parser.add_argument(
        "--keep-classifications",
        action="store_true",
        help="Keep classification rasters on disk. By default they are deleted "
        "right after per-point predictions are extracted (only prediction and "
        "matrix CSVs are kept), to avoid filling the disk on large runs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Matrix mode: print the planned runs and check that all input "
        "files exist, without running anything.",
    )
    parser.add_argument(
        "--only", metavar="RUN_ID",
        help="Matrix mode: run only the cell with this run_id "
        "(e.g. 1234567_poprawny_L2A_roi).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Matrix mode: re-run cells even if their results already exist "
        "(default: skip finished cells, so a batch can be resumed).",
    )

    args = parser.parse_args()

    args.rf_max_features = _parse_rf_max_features(args.rf_max_features)

    if args.verbose:
        log.setLevel(logging.INFO)

    if args.list_methods:
        print("Rejestr metod detekcji wartości odstających "
              "(%d metod):\n" % len(_METHOD_DESCRIPTIONS))
        for name, (category, description) in _METHOD_DESCRIPTIONS.items():
            print("  %-20s [%s]" % (name, category))
            print("      %s" % description)
        return

    if args.autotest:
        if args.reference is None:
            sys.exit("BŁĄD: tryb --autotest wymaga podania --reference.")
        ref = args.reference
        if ref.lower().endswith(".shp"):
            ref = _shp_to_gpkg(ref)
        sys.exit(_run_autotest(args.autotest, ref, args.ref_field))

    if args.evaluate_only:
        if args.reference is None:
            sys.exit("BŁĄD: tryb --evaluate-only wymaga podania --reference.")
        _run_from_args(args)
        return

    if args.evaluate_results_dir:
        _run_evaluate_results_dir(args)
        return

    if args.image is None and args.signatures is None:
        _run_matrix(args)
        return

    if not args.image or not args.signatures:
        sys.exit("BŁĄD: pojedynczy przebieg wymaga podania --image oraz --signatures "
                 "(or omit both to run the CONFIG matrix).")
    if args.reference is None:
        sys.exit("BŁĄD: pojedynczy przebieg wymaga podania --reference.")
    if not (1 <= args.max_k <= len(ALL_METHODS)):
        sys.exit("BŁĄD: --max-k musi mieścić się w zakresie od 1 do %d." % len(ALL_METHODS))
    args.run_id = args.run_id or "_".join(
        t for t in (args.variant, args.state, args.raster, args.scope) if t
    )
    _run_from_args(args)

def _evaluate_rasters_to_results(class_dir, ref_points, ref_values, algorithms,
                                 accuracy_dir):
    algo_safe = {re.sub(r"[^\w\-]", "_", a): a for a in algorithms}
    ref_classes = sorted(set(int(c) for c in ref_values))
    results = []
    for name in sorted(os.listdir(str(class_dir))):
        if not name.endswith(".tif"):
            continue
        base = name[:-4]
        algo = None
        for asafe in sorted(algo_safe, key=len, reverse=True):
            if base.endswith("__" + asafe):
                algo = algo_safe[asafe]
                label = base[:-(len(asafe) + 2)]
                break
        if algo is None:
            continue
        label = re.sub(r"_+scope_[a-z]+_*$", "", label)
        try:
            pred = _extract_predictions(str(Path(class_dir) / name), ref_points)
            acc = _pixel_accuracy(ref_values, pred)
        except Exception as err:
            results.append({"label": label, "algorithm": algo, "oa": None,
                            "kappa": None, "total_removed": None, "time_s": 0.0,
                            "error": str(err)})
            continue
        file_key = "%s__%s" % (re.sub(r"[^\w\-]", "_", label)[:80],
                               re.sub(r"[^\w\-]", "_", algo))
        _write_predictions_csv(
            str(accuracy_dir / ("%s_pred.csv" % file_key)), ref_points, pred)
        _write_matrix_csv(
            str(accuracy_dir / ("%s_macierz.csv" % file_key)),
            acc["matrix"], ref_classes)
        results.append({
            "label": label, "algorithm": algo, "oa": acc["oa"],
            "kappa": acc["kappa"], "per_class": acc["per_class"],
            "macro_pa": acc["macro_pa"], "macro_ua": acc["macro_ua"],
            "macro_f1": acc["macro_f1"], "n": acc["n"],
            "n_unclassified": acc["n_unclassified"], "matrix": acc["matrix"],
            "pred": pred.tolist(), "total_removed": None, "time_s": 0.0})
    return results

def _finalize_run(results, run_dir, class_dir, accuracy_dir, algorithms, tags,
                  ref_values):
    valid = [r for r in results if r.get("oa") is not None]
    invalid = [r for r in results if r.get("oa") is None]
    valid.sort(key=lambda r: (-(r["oa"] or 0), -(r["kappa"] or 0)))

    summary_path = run_dir / _FILE_SUMMARY_CSV
    _write_polish_csv(summary_path, valid, invalid, tags)
    print("Zapisano: %s" % summary_path)
    _write_per_class_csv(run_dir / _FILE_PER_CLASS_CSV, valid, tags)

    csv_dir = run_dir / _DIR_CSV_PER_ALGO
    csv_dir.mkdir(exist_ok=True)
    for algo in algorithms:
        algo_safe = re.sub(r"[^\w\-]", "_", algo)
        _write_polish_csv(
            csv_dir / (_FILE_CSV_PER_ALGO % algo_safe),
            [r for r in valid if r["algorithm"] == algo],
            [r for r in invalid if r.get("algorithm") == algo], tags)

    _KEY_CELLS = [(1, 7), (7, 1), (5, 1), (5, 2), (3, 1)]
    comparisons = []
    for algo in algorithms:
        base = next((r for r in results if r["label"] == "Baseline"
                     and r["algorithm"] == algo and r.get("pred") is not None), None)
        cands = [r for r in valid if r["algorithm"] == algo
                 and r["label"] != "Baseline" and r.get("pred") is not None]
        if base is None or not cands:
            continue
        best = max(cands, key=lambda r: (r["oa"], r["kappa"] or 0))
        b, c, p, test_name = _mcnemar(np.array(base["pred"]) == ref_values,
                                      np.array(best["pred"]) == ref_values)
        bm = {(i, j): n for i, j, n in base.get("matrix") or []}
        cm = {(i, j): n for i, j, n in best.get("matrix") or []}
        changes = ["C[%d,%d]: %d->%d" % (i, j, bm.get((i, j), 0), cm.get((i, j), 0))
                   for (i, j) in _KEY_CELLS if bm.get((i, j), 0) or cm.get((i, j), 0)]
        comparisons.append({
            "Algorytm": algo, "Najlepsza_konfiguracja": _method_display(best["label"]),
            "DA_bazowa_%": "%.4f" % base["oa"], "DA_najlepsza_%": "%.4f" % best["oa"],
            "Delta_DA_pp": "%+.4f" % (best["oa"] - base["oa"]),
            "b": b, "c": c, "test": test_name, "p": p,
            "Zmiany_komorek": "; ".join(changes)})
    if comparisons:
        for row, padj in zip(comparisons, _holm([x["p"] for x in comparisons])):
            row["p_Holm"] = padj
            row["Istotne_005"] = "TAK" if padj <= 0.05 else "nie"
        fields = _TAG_FIELDNAMES + [
            "Algorytm", "Najlepsza_konfiguracja", "DA_bazowa_%", "DA_najlepsza_%",
            "Delta_DA_pp", "b", "c", "test", "p", "p_Holm_lokalny",
            "Istotne_005_lokalnie", "Zmiany_komorek"]
        with open(run_dir / _FILE_MCNEMAR_CSV, "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in comparisons:
                w.writerow({**tags, "Algorytm": row["Algorytm"],
                    "Najlepsza_konfiguracja": row["Najlepsza_konfiguracja"],
                    "DA_bazowa_%": row["DA_bazowa_%"],
                    "DA_najlepsza_%": row["DA_najlepsza_%"],
                    "Delta_DA_pp": row["Delta_DA_pp"], "b": row["b"], "c": row["c"],
                    "test": row["test"], "p": "%.6g" % row["p"],
                    "p_Holm_lokalny": "%.6g" % row["p_Holm"],
                    "Istotne_005_lokalnie": row["Istotne_005"],
                    "Zmiany_komorek": row["Zmiany_komorek"]})
        print("Testy konfirmacyjne zapisane: %s" % (run_dir / _FILE_MCNEMAR_CSV))

    print("Generowanie wykresów ...")
    _generate_charts(results, algorithms, run_dir)
    print("Zestawienie: %s | wg algorytmów: %s | dokładność: %s"
          % (summary_path, csv_dir, accuracy_dir))

def _run_evaluate_results_dir(args):
    root = Path(args.evaluate_results_dir)
    if not root.is_dir():
        sys.exit("BŁĄD: katalog wyników nie istnieje: %s" % root)
    ref_dir = args.reference_dir or _REF
    if not os.path.isdir(ref_dir):
        sys.exit("BŁĄD: katalog referencji nie istnieje: %s" % ref_dir)
    algorithms = args.algorithm if args.algorithm else ALL_ALGORITHMS

    run_dirs = sorted(p for p in root.iterdir()
                      if (p / _DIR_CLASSIFICATIONS).is_dir())
    if not run_dirs:
        sys.exit("BŁĄD: brak przebiegów z katalogiem '%s/' w %s"
                 % (_DIR_CLASSIFICATIONS, root))

    print("Wsadowa ocena (bez klasyfikacji). Przebiegów: %d" % len(run_dirs))
    print("Referencje z: %s\n" % ref_dir)

    gpkg_cache = {}
    done = skipped = 0
    for rd in run_dirs:
        run_id = rd.name
        class_dir = rd / _DIR_CLASSIFICATIONS
        tifs = [f for f in os.listdir(str(class_dir)) if f.lower().endswith(".tif")]
        if not tifs:
            print("POMIJAM (brak rastrów): %s" % run_id)
            skipped += 1
            continue
        try:
            variant, state, raster, scope = run_id.rsplit("_", 3)
        except ValueError:
            print("POMIJAM (nazwa nie pasuje do "
                  "<wariant>_<stan>_<raster>_<zakres>): %s" % run_id)
            skipped += 1
            continue

        ref_path = os.path.join(ref_dir, "referencja_%s.shp" % variant)
        if not os.path.exists(ref_path):
            print("POMIJAM (brak referencji %s): %s"
                  % (os.path.basename(ref_path), run_id))
            skipped += 1
            continue

        if ref_path not in gpkg_cache:
            gpkg_cache[ref_path] = _shp_to_gpkg(ref_path)
        ref_points = _load_reference_points(gpkg_cache[ref_path], args.ref_field)
        ref_values = np.array([p[1] for p in ref_points])

        accuracy_dir = rd / _DIR_ACCURACY
        accuracy_dir.mkdir(exist_ok=True)
        tags = {"Run_ID": run_id, "Wariant": variant, "Stan": state,
                "Raster": raster, "Zakres": scope}

        print("\n=== %s  (wariant %s, referencja %s, klasy %s) ==="
              % (run_id, variant, os.path.basename(ref_path),
                 sorted(set(int(c) for c in ref_values))))
        results = _evaluate_rasters_to_results(
            class_dir, ref_points, ref_values, algorithms, accuracy_dir)
        if not results:
            print("POMIJAM (brak rastrów '<etykieta>__<algorytm>.tif'): %s" % run_id)
            skipped += 1
            continue
        _finalize_run(results, rd, class_dir, accuracy_dir, algorithms,
                      tags, ref_values)
        done += 1

    print("\n" + "=" * 60)
    print("Ocenione przebiegi: %d,  pominięte: %d" % (done, skipped))

def _run_from_args(args):
    reference_path = args.reference
    if reference_path and reference_path.lower().endswith(".shp"):
        print("Konwersja referencji SHP do tymczasowego GPKG ...")
        try:
            reference_path = _shp_to_gpkg(reference_path)
            print("  => %s" % reference_path)
        except Exception as err:
            sys.exit("BŁĄD: nie powiodła się konwersja SHP do GPKG: %s" % err)

    algorithms = args.algorithm if args.algorithm else ALL_ALGORITHMS

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or "_".join(
        t for t in (args.variant, args.state, args.raster, args.scope) if t
    )
    tags = {
        "Run_ID": run_id,
        "Wariant": args.variant,
        "Stan": args.state,
        "Raster": args.raster,
        "Zakres": args.scope,
    }
    run_dir = (out_dir / run_id) if run_id else out_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    class_dir = run_dir / _DIR_CLASSIFICATIONS
    class_dir.mkdir(exist_ok=True)
    accuracy_dir = run_dir / _DIR_ACCURACY
    accuracy_dir.mkdir(exist_ok=True)

    if getattr(args, "evaluate_only", None):
        ref_points = _load_reference_points(reference_path, args.ref_field)
        ref_values = np.array([p[1] for p in ref_points])
        print("Tryb oceny (bez klasyfikacji). Rastry z: %s" % args.evaluate_only)
        print("Referencja: %d poligonów, klasy %s"
              % (len(ref_points), sorted(set(int(c) for c in ref_values))))
        results = _evaluate_rasters_to_results(
            args.evaluate_only, ref_points, ref_values, algorithms, accuracy_dir)
        if not results:
            sys.exit("BŁĄD: brak rastrów '<etykieta>__<algorytm>.tif' do oceny "
                     "w %s" % args.evaluate_only)
        _finalize_run(results, run_dir, class_dir, accuracy_dir, algorithms,
                      tags, ref_values)
        return

    print("Inicjalizacja biblioteki remotior_sensus ...")
    rs = remotior_sensus.Session(
        n_processes=args.n_processes,
        available_ram=args.ram,
    )

    print("Wczytywanie obrazu: %s" % args.image)
    bandset_catalog = rs.bandset_catalog()
    bandset_catalog.create_bandset(paths=args.image, bandset_number=1)
    _set_wavelengths(bandset_catalog)
    bandset = bandset_catalog.get(1)
    if bandset is None or bandset.get_band_count() == 0:
        sys.exit("BŁĄD: nie można wczytać pasm z %s" % args.image)
    print("  => loaded %d bands" % bandset.get_band_count())

    print("Przygotowanie sygnatur z: %s" % args.signatures)
    signatures_scpx = _prepare_signatures_scpx(
        rs, bandset, args.signatures, args.image, args.output
    )
    probe_catalog = _load_fresh_catalog(rs, signatures_scpx, bandset)
    sig_count = len(probe_catalog.table) if probe_catalog.table is not None else 0
    print("  => %d signatures ready" % sig_count)
    if sig_count == 0:
        sys.exit("BŁĄD: brak sygnatur w %s" % args.signatures)

    ref_field = args.ref_field
    ref_points = _load_reference_points(reference_path, ref_field)
    ref_values = np.array([p[1] for p in ref_points])
    print(
        "Referencja: %d poligonów, klasy %s"
        % (len(ref_points), sorted(set(int(c) for c in ref_values)))
    )

    configs = [("Baseline", [], False, 1)]
    configs += list(
        _build_configs(
            args.max_k, args.methods, grid_search=args.grid_search,
        )
    )

    n_k1 = sum(1 for c in configs[1:] if len(c[1]) == 1)
    n_by_k = {}
    for c in configs[1:]:
        k = len(c[1])
        n_by_k[k] = n_by_k.get(k, 0) + 1
    print(
        "\nConfigs to test: %d x %d algorithm(s) = %d runs total"
        % (len(configs), len(algorithms), len(configs) * len(algorithms))
    )
    k_summary = "  Baseline: 1  |  k=1: %d%s" % (
        n_k1,
        " (grid search)" if args.grid_search else "",
    )
    for k in sorted(k for k in n_by_k if k > 1):
        k_summary += "  |  k=%d: %d" % (k, n_by_k[k])
    print(k_summary)
    print("  Algorithms: %s" % ", ".join(algorithms))

    tasks = [
        (
            label,
            steps,
            use_voting,
            vote_threshold,
            reference_path,
            ref_field,
            algorithms,
            args.macroclass,
            args.scope,
            str(class_dir),
            str(accuracy_dir),
        )
        for label, steps, use_voting, vote_threshold in configs
    ]
    n_tasks = len(tasks)

    n_workers = max(1, min(args.workers, n_tasks))
    worker_n_proc = max(1, args.n_processes // n_workers)
    worker_ram = max(512, args.ram // n_workers)

    results = []

    if n_workers == 1:
        _worker_init(
            args.image, signatures_scpx, args.n_processes, args.ram,
            args.keep_classifications,
            args.rf_number_trees, args.rf_max_features,
        )
        for i, task in enumerate(tasks, 1):
            label = task[0]
            print("\n[%d/%d] %s" % (i, n_tasks, label))
            rows = _run_one_config(task)
            results.extend(rows)
            for r in rows:
                if r.get("oa") is not None:
                    print(
                        "  [%-26s]  OA: %.4f%%   Kappa: %.4f   time: %s s"
                        % (r["algorithm"], r["oa"], r["kappa"], r["time_s"])
                    )
                elif r.get("error"):
                    log.error("[%s] %s", r["algorithm"], r["error"])
    else:
        print(
            "\nRunning %d configs on %d parallel workers "
            "(worker_n_proc=%d, worker_ram=%d MB) ..."
            % (n_tasks, n_workers, worker_n_proc, worker_ram)
        )
        pool = _NoDaemonPool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(
                args.image, signatures_scpx, worker_n_proc, worker_ram,
                args.keep_classifications,
                args.rf_number_trees, args.rf_max_features,
            ),
        )
        done = 0
        try:
            for rows in pool.imap_unordered(_run_one_config, tasks):
                done += 1
                results.extend(rows)
                label_str = rows[0]["label"] if rows else "?"
                oa_vals = [r["oa"] for r in rows if r["oa"] is not None]
                oa_str = ("best OA: %.4f%%" % max(oa_vals)) if oa_vals else "all N/A"
                print("[%d/%d] %-50s  %s" % (done, n_tasks, label_str[:50], oa_str))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            sys.exit("\nInterrupted by user.")
        pool.close()
        pool.join()

    valid = [r for r in results if r.get("oa") is not None]
    invalid = [r for r in results if r.get("oa") is None]
    valid.sort(key=lambda r: (-(r["oa"] or 0), -(r["kappa"] or 0)))

    summary_path = run_dir / _FILE_SUMMARY_CSV
    _write_polish_csv(summary_path, valid, invalid, tags)
    print("Zapisano: %s" % summary_path)

    per_class_path = run_dir / _FILE_PER_CLASS_CSV
    _write_per_class_csv(per_class_path, valid, tags)
    print("Dokładność w podziale na klasy: %s" % per_class_path)

    csv_dir = run_dir / _DIR_CSV_PER_ALGO
    csv_dir.mkdir(exist_ok=True)
    for algo in algorithms:
        algo_valid = [r for r in valid if r["algorithm"] == algo]
        algo_invalid = [r for r in invalid if r.get("algorithm") == algo]
        algo_safe = re.sub(r"[^\w\-]", "_", algo)
        algo_path = csv_dir / (_FILE_CSV_PER_ALGO % algo_safe)
        _write_polish_csv(algo_path, algo_valid, algo_invalid, tags)
    print("Pliki wg algorytmów: %s" % csv_dir)

    baseline_oa_by_algo = {
        r["algorithm"]: r["oa"]
        for r in results
        if r["label"] == "Baseline" and r.get("oa") is not None
    }

    print("\n" + "=" * 82)
    print("ZESTAWIENIE WYNIKÓW  (malejąco według dokładności całkowitej)")
    print("=" * 82)
    print(
        "%-5s %-32s %-22s %8s %8s %12s"
        % ("Rank", "Method", "Algorithm", "OA [%]", "Kappa", "Removed px")
    )
    print("-" * 82)

    for rank, row in enumerate(valid[:20], 1):
        baseline_oa = baseline_oa_by_algo.get(row["algorithm"])
        delta = ""
        if (
            baseline_oa is not None
            and row["label"] != "Baseline"
            and row["oa"] is not None
        ):
            diff = row["oa"] - baseline_oa
            delta = " (%+.2f)" % diff
        display_label = "Baseline" if row["label"] == "Baseline" else row["label"]
        print(
            "%-5d %-32s %-22s %8.4f %8.4f %12s%s"
            % (
                rank,
                display_label[:32],
                row["algorithm"][:22],
                row["oa"] or 0,
                row["kappa"] or 0,
                str(row.get("total_removed") or "-"),
                delta,
            )
        )

    if len(valid) > 20:
        print("  ... and %d more in %s" % (len(valid) - 20, summary_path))
    if invalid:
        print("  konfiguracji zakończonych błędem: %d (kolumna 'Blad' w pliku CSV)" % len(invalid))

    best = next((r for r in valid if r.get("per_class")), None)
    if best is not None:
        best_label = (
            "Baseline" if best["label"] == "Baseline" else best["label"]
        )
        print("\n" + "=" * 82)
        print(
            "PER-CLASS ACCURACY  (best config: %s / %s)"
            % (best_label[:40], best["algorithm"])
        )
        print("-" * 82)
        print(
            "%-20s %12s %12s %14s"
            % ("Class", "PA [%]", "UA [%]", "F1 [%]")
        )
        print("-" * 82)
        for pc in best["per_class"]:
            print(
                "%-20s %12s %12s %14s"
                % (
                    str(pc["class"])[:20],
                    _fmt_metric(pc["pa"]) or "-",
                    _fmt_metric(pc["ua"]) or "-",
                    _fmt_metric(pc["f1"]) or "-",
                )
            )
        print("  Full per-class table (all configs): %s" % per_class_path)

    _KEY_CELLS = [(1, 7), (7, 1), (5, 1), (5, 2), (3, 1)]
    comparisons = []
    for algo in algorithms:
        base = next(
            (
                r
                for r in results
                if r["label"] == "Baseline"
                and r["algorithm"] == algo
                and r.get("pred") is not None
            ),
            None,
        )
        candidates = [
            r
            for r in valid
            if r["algorithm"] == algo
            and r["label"] != "Baseline"
            and r.get("pred") is not None
        ]
        if base is None or not candidates:
            continue
        best_cfg = max(candidates, key=lambda r: (r["oa"], r["kappa"] or 0))
        correct_base = np.array(base["pred"]) == ref_values
        correct_best = np.array(best_cfg["pred"]) == ref_values
        b, c, p, test_name = _mcnemar(correct_base, correct_best)

        base_matrix = {(i, j): n for i, j, n in base.get("matrix") or []}
        best_matrix = {(i, j): n for i, j, n in best_cfg.get("matrix") or []}
        cell_changes = []
        for cell in _KEY_CELLS:
            before = base_matrix.get(cell, 0)
            after = best_matrix.get(cell, 0)
            if before or after:
                cell_changes.append(
                    "C[%d,%d]: %d->%d" % (cell[0], cell[1], before, after)
                )

        comparisons.append(
            {
                "Algorytm": algo,
                "Najlepsza_konfiguracja": _method_display(best_cfg["label"]),
                "DA_bazowa_%": "%.4f" % base["oa"],
                "DA_najlepsza_%": "%.4f" % best_cfg["oa"],
                "Delta_DA_pp": "%+.4f" % (best_cfg["oa"] - base["oa"]),
                "b": b,
                "c": c,
                "test": test_name,
                "p": p,
                "Zmiany_komorek": "; ".join(cell_changes),
            }
        )

    if comparisons:
        p_adjusted = _holm([cmp_row["p"] for cmp_row in comparisons])
        for cmp_row, p_adj in zip(comparisons, p_adjusted):
            cmp_row["p_Holm"] = p_adj
            cmp_row["Istotne_005"] = "TAK" if p_adj <= 0.05 else "nie"

        print("\n" + "=" * 82)
        print(
            "KONFIRMACJA LOKALNA (tylko ten przebieg): najlepsza konfiguracja "
            "vs poziom bazowy, McNemar; m=%d" % len(comparisons)
        )
        print(
            "  UWAGA: poprawka Holma poniżej obejmuje wyłącznie %d algorytmów "
            "tego przebiegu." % len(comparisons)
        )
        print(
            "  Końcowa istotność = JEDEN Holm nad wszystkimi przebiegami "
            "(krok agregujący, jeszcze nie policzony)."
        )
        print("-" * 82)
        print(
            "%-22s %-26s %9s %5s %5s %10s %11s %8s"
            % ("Algorytm", "Konfiguracja", "ΔDA [pp]", "b", "c", "p",
               "p Holm(lok)", "istotne")
        )
        print("-" * 82)
        for cmp_row in comparisons:
            print(
                "%-22s %-26s %9s %5d %5d %10.4g %11.4g %8s"
                % (
                    cmp_row["Algorytm"][:22],
                    cmp_row["Najlepsza_konfiguracja"][:26],
                    cmp_row["Delta_DA_pp"],
                    cmp_row["b"],
                    cmp_row["c"],
                    cmp_row["p"],
                    cmp_row["p_Holm"],
                    cmp_row["Istotne_005"],
                )
            )

        mcnemar_path = run_dir / _FILE_MCNEMAR_CSV
        with open(mcnemar_path, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = _TAG_FIELDNAMES + [
                "Algorytm", "Najlepsza_konfiguracja", "DA_bazowa_%",
                "DA_najlepsza_%", "Delta_DA_pp", "b", "c", "test", "p",
                "p_Holm_lokalny", "Istotne_005_lokalnie", "Zmiany_komorek",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for cmp_row in comparisons:
                writer.writerow(
                    {
                        **tags,
                        "Algorytm": cmp_row["Algorytm"],
                        "Najlepsza_konfiguracja": cmp_row["Najlepsza_konfiguracja"],
                        "DA_bazowa_%": cmp_row["DA_bazowa_%"],
                        "DA_najlepsza_%": cmp_row["DA_najlepsza_%"],
                        "Delta_DA_pp": cmp_row["Delta_DA_pp"],
                        "b": cmp_row["b"],
                        "c": cmp_row["c"],
                        "test": cmp_row["test"],
                        "p": "%.6g" % cmp_row["p"],
                        "p_Holm_lokalny": "%.6g" % cmp_row["p_Holm"],
                        "Istotne_005_lokalnie": cmp_row["Istotne_005"],
                        "Zmiany_komorek": cmp_row["Zmiany_komorek"],
                    }
                )
        print("Testy konfirmacyjne zapisane: %s" % mcnemar_path)

    print("\nGenerowanie wykresów ...")
    _generate_charts(results, algorithms, run_dir)

    print("\nFull results (summary): %s" % summary_path)
    print("Wyniki wg algorytmów:   %s" % csv_dir)
    print("Rastry klasyfikacji:    %s" % class_dir)
    print("Tabele dokładności:     %s" % accuracy_dir)

if __name__ == "__main__":
    main()
