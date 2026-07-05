import argparse
import csv
import itertools
import logging
import multiprocessing
import multiprocessing.pool
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

_qgis = "/Applications/QGIS-LTR.app/Contents"
if os.path.isdir(_qgis):
    os.environ.setdefault("PROJ_LIB", _qgis + "/Resources/proj")
    os.environ.setdefault("GDAL_DATA", _qgis + "/Resources/gdal")

import numpy as np

try:
    from osgeo import gdal, ogr, osr

    gdal.UseExceptions()
except ImportError:
    sys.exit(
        "ERROR: GDAL (osgeo) not found. Run inside the QGIS Python console "
        "or install 'gdal' in your environment."
    )

try:
    from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance, MinCovDet
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
except ImportError:
    sys.exit("ERROR: scikit-learn not found. Install it with: pip install scikit-learn")

try:
    from scipy.ndimage import binary_erosion
    from scipy.stats import chi2, f as f_dist
except ImportError:
    sys.exit("ERROR: scipy not found. Install it with: pip install scipy")

try:
    import remotior_sensus
    from remotior_sensus.core import table_manager as _rs_tm
except ImportError:
    sys.exit("ERROR: remotior_sensus not found. Install: pip install remotior-sensus")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger("benchmark")

_RANDOM_SEED = 42


def _mahalanobis_cutoff(threshold, alpha, df):
    if alpha is not None:
        return float(chi2.ppf(1.0 - float(alpha), df))
    if threshold is None:
        threshold = 3.0
    return float(threshold) ** 2


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

# Grid-search obejmuje tylko parametry, dla których analiza wrażliwości ma sens:
# priory (contamination/nu, poziom przycięcia percentyli), poziom istotności
# alpha oraz parametry strukturalne (variance_ratio, liczba komponentów/sąsiadów).
# Progi literaturowe (MAD 3.5, Band Z-score 3.0, IQR 1.5, Erosion 1) NIE są tu
# wpisane — przy --grid-search pozostają na stałej wartości literaturowej
# (fallback do _DEFAULT_PARAMS w _build_configs), bo odchodzenie od standardu
# nie ma uzasadnienia. Skraca to też liczbę konfiguracji i czas.
_PARAM_GRID = {
    "Percentile": [
        {"lower_pct": lo, "upper_pct": hi}
        for lo, hi in [(0.005, 0.995), (0.01, 0.99), (0.02, 0.98), (0.05, 0.95)]
    ],
    "Mahalanobis": [{"alpha": a} for a in [0.01, 0.025, 0.05]],
    "Robust Mahalanobis": [{"alpha": a} for a in [0.01, 0.025, 0.05]],
    "HotellingT2": [{"alpha": a} for a in [0.01, 0.05, 0.10]],
    "PCA": [
        {"variance_ratio": v, "alpha": a}
        for v in [0.90, 0.95, 0.99]
        for a in [0.01, 0.025, 0.05]
    ],
    "PCA Reconstruction": [
        {"variance_ratio": v, "contamination": c}
        for v in [0.90, 0.95, 0.99]
        for c in [0.005, 0.01, 0.02, 0.05]
    ],
    "GMM": [
        {"n_components": n, "contamination": c}
        for n in [2, 3]
        for c in [0.005, 0.01, 0.02, 0.05]
    ],
    "EllipticEnvelope": [{"contamination": c} for c in [0.005, 0.01, 0.02, 0.05]],
    "IsolationForest": [{"contamination": c} for c in [0.005, 0.01, 0.02, 0.05, 0.1]],
    "LOF": [
        {"n_neighbors": n, "contamination": c}
        for n in [10, 20, 30]
        for c in [0.01, 0.02, 0.05]
    ],
    "kNN": [
        {"n_neighbors": n, "contamination": c}
        for n in [5, 10, 20]
        for c in [0.01, 0.02, 0.05]
    ],
    "OneClassSVM": [{"nu": nu} for nu in [0.005, 0.01, 0.02, 0.05]],
    "SAM": [{"contamination": c} for c in [0.005, 0.01, 0.02, 0.05]],
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

_PL_FIELDNAMES = [
    "Lp.",
    "Metoda",
    "Algorytm",
    "DA_%",
    "Kappa",
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
    "Kappa_hat",
]

_DIR_CLASSIFICATIONS = "classifications"
_DIR_ACCURACY = "accuracy"
_DIR_CSV_PER_ALGO = "csv_per_algorytm"
_DIR_CHARTS_PER_ALGO = "wykresy_per_algorytm"

_FILE_TEST_REFERENCE = "test_reference.gpkg"
_FILE_SUMMARY_CSV = "wyniki_zbiorcze.csv"
_FILE_PER_CLASS_CSV = "wyniki_per_klasa.csv"
_FILE_CSV_PER_ALGO = "wyniki_%s.csv"

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


# Repairs invalid polygon geometries in a cutline GeoPackage in place. gdalwarp
# rejects self-touching ("bowtie") cutlines with "Cutline polygon is invalid";
# MakeValid only splits the corner touches, so the clipped pixels are unchanged.
def _ensure_valid_cutline(roi_gpkg):
    ds = ogr.Open(roi_gpkg, 1)
    if ds is None:
        return
    try:
        layer = ds.GetLayer()
        for feat in layer:
            geom = feat.GetGeometryRef()
            if geom is None or geom.IsValid():
                continue
            try:
                fixed = geom.MakeValid()
            except Exception:
                fixed = None
            if fixed is None or fixed.IsEmpty():
                fixed = geom.Buffer(0)
            if fixed is not None and not fixed.IsEmpty() and fixed.IsValid():
                feat.SetGeometry(fixed)
                layer.SetFeature(feat)
        ds.FlushCache()
    finally:
        ds = None


def _warp_to_memory(raster_path, roi_gpkg, nodata=0):
    _ensure_valid_cutline(roi_gpkg)
    opts = gdal.WarpOptions(
        format="MEM", cutlineDSName=roi_gpkg, cropToCutline=True, dstNodata=nodata
    )
    return gdal.Warp("", raster_path, options=opts)


def _warp_multiband_to_memory(band_paths, roi_gpkg, nodata=0):
    unique = list(dict.fromkeys(band_paths))
    if len(unique) == 1:
        return _warp_to_memory(unique[0], roi_gpkg, nodata)
    vrt = _tmp(".vrt")
    gdal.BuildVRT(vrt, unique, separate=True)
    ds = _warp_to_memory(vrt, roi_gpkg, nodata)
    try:
        os.remove(vrt)
    except OSError:
        pass
    return ds


def _create_mask_dataset(mask_array, geotransform, projection):
    h, w = mask_array.shape
    ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)
    band = ds.GetRasterBand(1)
    band.WriteArray(mask_array)
    return ds, band


# Returns a valid version of a polygon geometry (mirrors _make_valid_geometry in
# the plugin's interface/remove_outliers.py). MakeValid only splits self-touching
# corners, so the rasterized pixel set is unchanged.
def _make_valid_geom(geom):
    if geom is None or geom.IsValid():
        return geom
    try:
        fixed = geom.MakeValid()
    except Exception:
        fixed = None
    if fixed is None or fixed.IsEmpty():
        fixed = geom.Buffer(0)
    if fixed is not None and not fixed.IsEmpty() and fixed.IsValid():
        return fixed
    return geom


def _update_roi_geometry(geometry_file, sig_id, clean_mask, geotransform, projection):
    clean_uint8 = clean_mask.astype(np.uint8)
    ds_mask, mask_band = _create_mask_dataset(clean_uint8, geotransform, projection)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)

    tmp_gpkg = _tmp(".gpkg")
    tmp_ds = ogr.GetDriverByName("GPKG").CreateDataSource(tmp_gpkg)
    tmp_layer = tmp_ds.CreateLayer("clean", srs=srs, geom_type=ogr.wkbPolygon)
    tmp_layer.CreateField(ogr.FieldDefn("val", ogr.OFTInteger))
    gdal.Polygonize(mask_band, mask_band, tmp_layer, 0)
    tmp_ds.FlushCache()
    tmp_ds = None
    ds_mask = None

    tmp_ds = ogr.Open(tmp_gpkg)
    tmp_layer = tmp_ds.GetLayer()
    # Łączenie poligonów przez Union (rozpuszcza wspólne krawędzie) + MakeValid,
    # identycznie jak wtyczka SCP (interface/remove_outliers.py). Daje ten sam
    # zbiór pikseli co AddGeometry, ale poprawną geometrię ROI dla klasyfikacji.
    union_geom = None
    for feat in tmp_layer:
        g = feat.GetGeometryRef()
        if g is not None:
            union_geom = g.Clone() if union_geom is None else union_geom.Union(g)
    tmp_ds = None
    try:
        os.remove(tmp_gpkg)
    except OSError:
        pass

    if union_geom is None:
        return
    union_geom = _make_valid_geom(union_geom)

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


def _mad(stack, valid_mask, threshold=3.5):
    # Zgodne 1:1 z mad_filter_stack we wtyczce (interface/remove_outliers.py):
    # mediana/MAD liczone per pasmo po całym stacku (nanmedian), a MAD==0 daje
    # mianownik 1 (a nie pominięcie pasma).
    median = np.nanmedian(stack, axis=(1, 2), keepdims=True)
    mad = np.nanmedian(np.abs(stack - median), axis=(1, 2), keepdims=True)
    mad = np.where(mad == 0, 1, mad)
    modified_z = 0.6745 * (stack - median) / mad
    outlier_mask = np.any(np.abs(modified_z) > threshold, axis=0)
    return valid_mask & (~outlier_mask)


def _band_zscore(stack, valid_mask, threshold=3.0):
    outlier_mask = np.zeros(valid_mask.shape, dtype=bool)
    for b in range(stack.shape[0]):
        band = stack[b]
        vals = band[valid_mask]
        if len(vals) < 3:
            continue
        mean, std = np.nanmean(vals), np.nanstd(vals)
        if std == 0:
            continue
        z = (band - mean) / std
        outlier_mask |= (np.abs(z) > threshold) & valid_mask
    return ~outlier_mask & valid_mask


def _percentile(stack, valid_mask, lower_pct=0.01, upper_pct=0.99):
    outlier_mask = np.zeros(valid_mask.shape, dtype=bool)
    for b in range(stack.shape[0]):
        band = stack[b]
        vals = band[valid_mask]
        if len(vals) < 3:
            continue
        lo = np.nanpercentile(vals, lower_pct * 100)
        hi = np.nanpercentile(vals, upper_pct * 100)
        outlier_mask |= ((band < lo) | (band > hi)) & valid_mask
    return ~outlier_mask & valid_mask


def _iqr(stack, valid_mask, factor=1.5):
    outlier_mask = np.zeros(valid_mask.shape, dtype=bool)
    for b in range(stack.shape[0]):
        band = stack[b]
        vals = band[valid_mask]
        if len(vals) < 3:
            continue
        q1, q3 = np.nanpercentile(vals, 25), np.nanpercentile(vals, 75)
        iqr = q3 - q1
        if iqr == 0:
            iqr = 1.0  # mianownik bezpieczny, zgodnie z iqr_filter we wtyczce
        outlier_mask |= (
            (band < q1 - factor * iqr) | (band > q3 + factor * iqr)
        ) & valid_mask
    return ~outlier_mask & valid_mask


def _mahalanobis(stack, valid_mask, threshold=None, alpha=None):
    pts = StandardScaler().fit_transform(stack[:, valid_mask].T)
    if pts.shape[0] < pts.shape[1] + 2:
        return valid_mask
    try:
        cov = EmpiricalCovariance().fit(pts)
        md = cov.mahalanobis(pts)
    except Exception as err:
        # Plugin (mahalanobis_filter) nie łapie wyjątku → ROI pomijane.
        raise ValueError("Mahalanobis failed: %s" % err)
    cutoff = _mahalanobis_cutoff(threshold, alpha, df=pts.shape[1])
    new_mask = np.zeros(valid_mask.shape, dtype=bool)
    new_mask[valid_mask] = md <= cutoff
    return new_mask


def _robust_mahalanobis(stack, valid_mask, threshold=None, alpha=None):
    pts = StandardScaler().fit_transform(stack[:, valid_mask].T)
    try:
        cov = MinCovDet(random_state=_RANDOM_SEED).fit(pts)
        md = cov.mahalanobis(pts)
    except Exception as err:
        # Plugin (mahalanobis_filter robust=True) nie łapie wyjątku → ROI pomijane.
        raise ValueError("Robust Mahalanobis failed: %s" % err)
    cutoff = _mahalanobis_cutoff(threshold, alpha, df=pts.shape[1])
    new_mask = np.zeros(valid_mask.shape, dtype=bool)
    new_mask[valid_mask] = md <= cutoff
    return new_mask


def _resolve_pca_components(n_components, variance_ratio, n_features):
    """variance_ratio ∈ (0,1) → ułamek wariancji (sklearn dobiera liczbę
    składowych); w przeciwnym razie stała liczba składowych przycięta do cech."""
    if variance_ratio is not None and 0.0 < float(variance_ratio) < 1.0:
        return float(variance_ratio)
    return max(1, min(int(n_components), n_features))


def _pca(
    stack, valid_mask, n_components=3, threshold=None, alpha=None,
    variance_ratio=None,
):
    pts = StandardScaler().fit_transform(stack[:, valid_mask].T)
    n_comp = _resolve_pca_components(n_components, variance_ratio, pts.shape[1])
    try:
        data_pca = PCA(n_components=n_comp, random_state=_RANDOM_SEED).fit_transform(
            pts
        )
        cov = MinCovDet(random_state=_RANDOM_SEED).fit(data_pca)
        md = cov.mahalanobis(data_pca)
    except Exception as err:
        # Plugin (pca_filter) nie łapie wyjątku → ROI pomijane.
        raise ValueError("PCA Mahalanobis failed: %s" % err)
    cutoff = _mahalanobis_cutoff(threshold, alpha, df=data_pca.shape[1])
    new_mask = np.zeros(valid_mask.shape, dtype=bool)
    new_mask[valid_mask] = md <= cutoff
    return new_mask


def _pca_reconstruction(
    stack, valid_mask, n_components=3, contamination=0.01, variance_ratio=None,
):
    pts = StandardScaler().fit_transform(stack[:, valid_mask].T)
    n_comp = _resolve_pca_components(n_components, variance_ratio, pts.shape[1])
    try:
        pca = PCA(n_components=n_comp, random_state=42)
        proj = pca.inverse_transform(pca.fit_transform(pts))
        errors = np.linalg.norm(pts - proj, axis=1)
        thresh = np.quantile(errors, 1.0 - contamination)
    except Exception as err:
        raise ValueError("PCA Reconstruction failed: %s" % err)
    new_mask = np.zeros(valid_mask.shape, dtype=bool)
    new_mask[valid_mask] = errors <= thresh
    return new_mask


def _isolation_forest(stack, valid_mask, contamination=0.01):
    pts = StandardScaler().fit_transform(stack[:, valid_mask].T)
    n = pts.shape[0]
    if n < 5:
        return valid_mask
    try:
        sample_size = 10000
        if n > sample_size:
            rng = np.random.default_rng(_RANDOM_SEED)
            sample = pts[rng.choice(n, sample_size, replace=False)]
        else:
            sample = pts
        clf = IsolationForest(
            contamination=contamination, random_state=_RANDOM_SEED, n_jobs=-1
        )
        clf.fit(sample)
        pred = clf.predict(pts)
    except Exception as err:
        raise ValueError("IsolationForest failed: %s" % err)
    new_mask = np.zeros(valid_mask.shape, dtype=bool)
    new_mask[valid_mask] = pred == 1
    return new_mask


def _lof(stack, valid_mask, n_neighbors=20, contamination=0.01):
    pts = StandardScaler().fit_transform(stack[:, valid_mask].T)
    n_nb = min(n_neighbors, pts.shape[0] - 1)
    if n_nb < 2:
        return valid_mask
    try:
        pred = LocalOutlierFactor(
            n_neighbors=n_nb, contamination=contamination, n_jobs=-1
        ).fit_predict(pts)
    except Exception as err:
        raise ValueError("LOF failed: %s" % err)
    new_mask = np.zeros(valid_mask.shape, dtype=bool)
    new_mask[valid_mask] = pred == 1
    return new_mask


def _knn(stack, valid_mask, n_neighbors=5, contamination=0.01):
    pts = StandardScaler().fit_transform(stack[:, valid_mask].T)
    n_nb = min(n_neighbors, pts.shape[0] - 1)
    if n_nb < 2:
        return valid_mask
    try:
        nn = NearestNeighbors(n_neighbors=n_nb, n_jobs=-1)
        nn.fit(pts)
        dists, _ = nn.kneighbors(pts)
        scores = dists[:, -1]
        thresh = np.quantile(scores, 1.0 - contamination)
    except Exception as err:
        raise ValueError("kNN failed: %s" % err)
    new_mask = np.zeros(valid_mask.shape, dtype=bool)
    new_mask[valid_mask] = scores <= thresh
    return new_mask


def _elliptic_envelope(stack, valid_mask, contamination=0.01):
    pts = StandardScaler().fit_transform(stack[:, valid_mask].T)
    try:
        pred = EllipticEnvelope(
            contamination=contamination, random_state=_RANDOM_SEED
        ).fit_predict(pts)
    except Exception as err:
        raise ValueError("EllipticEnvelope failed: %s" % err)
    new_mask = np.zeros(valid_mask.shape, dtype=bool)
    new_mask[valid_mask] = pred == 1
    return new_mask


def _hotelling_t2(stack, valid_mask, alpha=0.05):
    pts = StandardScaler().fit_transform(stack[:, valid_mask].T)
    n, p = pts.shape
    if n <= p + 1:
        return valid_mask
    try:
        cov = EmpiricalCovariance().fit(pts)
        t2 = cov.mahalanobis(pts)
        f_crit = f_dist.ppf(1.0 - alpha, p, n - p)
        t2_crit = p * (n - 1) / (n - p) * f_crit
    except Exception:
        return valid_mask
    new_mask = np.zeros(valid_mask.shape, dtype=bool)
    new_mask[valid_mask] = t2 <= t2_crit
    return new_mask


def _gmm(stack, valid_mask, n_components=2, contamination=0.01):
    pts = StandardScaler().fit_transform(stack[:, valid_mask].T)
    if pts.shape[0] < n_components + 1:
        return valid_mask
    try:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=_RANDOM_SEED,
        )
        gmm.fit(pts)
        log_lik = gmm.score_samples(pts)
        thr = np.quantile(log_lik, contamination)
    except Exception as err:
        log.warning("GMM failed: %s", err)
        return valid_mask
    new_mask = np.zeros(valid_mask.shape, dtype=bool)
    new_mask[valid_mask] = log_lik >= thr
    return new_mask


def _one_class_svm(stack, valid_mask, nu=0.01):
    pts = StandardScaler().fit_transform(stack[:, valid_mask].T)
    if pts.shape[0] < 5:
        return valid_mask
    try:
        pred = OneClassSVM(nu=nu, gamma="scale", kernel="rbf").fit_predict(pts)
    except Exception as err:
        raise ValueError("OneClassSVM failed: %s" % err)
    new_mask = np.zeros(valid_mask.shape, dtype=bool)
    new_mask[valid_mask] = pred == 1
    return new_mask


def _sam(stack, valid_mask, contamination=0.01):
    pts = stack[:, valid_mask].T
    if pts.shape[0] < 3:
        return valid_mask
    try:
        mean_vec = pts.mean(axis=0)
        norms_pts = np.linalg.norm(pts, axis=1, keepdims=True)
        norms_pts[norms_pts == 0] = 1e-10
        norm_mean = np.linalg.norm(mean_vec)
        if norm_mean == 0:
            return valid_mask
        cos = (pts @ mean_vec) / (norms_pts.flatten() * norm_mean)
        angles = np.arccos(np.clip(cos, -1, 1))
        thresh = np.quantile(angles, 1.0 - contamination)
    except Exception as err:
        raise ValueError("SAM failed: %s" % err)
    new_mask = np.zeros(valid_mask.shape, dtype=bool)
    new_mask[valid_mask] = angles <= thresh
    return new_mask


def _erosion(stack, valid_mask, iterations=1):
    try:
        eroded = binary_erosion(valid_mask, iterations=iterations)
    except Exception:
        return valid_mask
    if not np.any(eroded):
        return valid_mask
    return valid_mask & eroded


_METHOD_FN = {
    "MAD": _mad,
    "Band Z-score": _band_zscore,
    "Percentile": _percentile,
    "IQR": _iqr,
    "Mahalanobis": _mahalanobis,
    "Robust Mahalanobis": _robust_mahalanobis,
    "HotellingT2": _hotelling_t2,
    "PCA": _pca,
    "PCA Reconstruction": _pca_reconstruction,
    "GMM": _gmm,
    "EllipticEnvelope": _elliptic_envelope,
    "IsolationForest": _isolation_forest,
    "LOF": _lof,
    "kNN": _knn,
    "OneClassSVM": _one_class_svm,
    "SAM": _sam,
    "Erosion": _erosion,
}


def run_pipeline(ds, steps, nodata=None, use_majority_voting=False, vote_threshold=2):
    band1 = ds.GetRasterBand(1)
    if nodata is None:
        nodata = band1.GetNoDataValue()

    stack = ds.ReadAsArray().astype(np.float32)
    if stack.ndim == 2:
        stack = stack[np.newaxis]
    if nodata is not None:
        try:
            nodata = float(nodata)
            stack[np.isclose(stack, nodata, equal_nan=False)] = np.nan
        except (TypeError, ValueError):
            stack[stack == nodata] = np.nan

    total_raster_px = int(stack.shape[1] * stack.shape[2])
    original_valid = np.all(~np.isnan(stack), axis=0)
    valid_px = int(np.sum(original_valid))
    mask_final = original_valid.copy()
    step_reports = []

    if use_majority_voting:
        # Semantyka głosowania zgodna z wtyczką (ensemble_outlier_mask):
        # liczone są głosy "inlier" (piksel zachowany przez metodę), a piksel
        # zostaje, gdy zachowa go co najmniej vote_threshold metod.
        keep_votes = np.zeros(original_valid.shape, dtype=np.int32)
        for method, params in steps:
            fn = _METHOD_FN.get(method)
            if fn is None:
                continue
            if method not in ("MAD", "Erosion") and int(
                np.sum(original_valid)
            ) < max(10, stack.shape[0] + 2):
                raise ValueError("Too few valid pixels in ROI")
            m = fn(stack, original_valid, **params)
            keep_votes += (m & original_valid).astype(np.int32)
        mask_final = original_valid & (keep_votes >= vote_threshold)
        removed = int(np.sum(original_valid & ~mask_final))
        step_reports.append(
            {
                "step": 1,
                "method": "MajorityVoting",
                "removed_pixels": removed,
                "removed_percent": 100.0 * removed / valid_px if valid_px else 0.0,
            }
        )
    else:
        for idx, (method, params) in enumerate(steps, 1):
            fn = _METHOD_FN.get(method)
            if fn is None:
                continue
            # Zgodnie z wtyczką (remove_multivariate_outliers): każda metoda poza
            # MAD i Erosion wymaga min. max(10, pasma+2) ważnych pikseli; inaczej
            # ROI jest pomijane (ValueError łapany w _apply_removal_to_catalog).
            if method not in ("MAD", "Erosion") and int(
                np.sum(mask_final)
            ) < max(10, stack.shape[0] + 2):
                raise ValueError("Too few valid pixels in ROI")
            mask_final = fn(stack, mask_final, **params)
            removed = int(np.sum(original_valid & ~mask_final))
            pct = 100.0 * removed / valid_px if valid_px else 0.0
            step_reports.append(
                {
                    "step": idx,
                    "method": method,
                    "removed_pixels": removed,
                    "removed_percent": pct,
                }
            )

    return mask_final, {
        "total_raster_pixels": total_raster_px,
        "valid_pixels": valid_px,
        "steps": step_reports,
    }


def _apply_removal_to_catalog(catalog, band_paths, steps, use_voting, vote_threshold):
    geometry_file = catalog.geometry_file
    ds_read = ogr.Open(geometry_file)
    if ds_read is None:
        raise ValueError("Cannot open geometry file: %s" % geometry_file)

    layer = ds_read.GetLayer()
    srs = layer.GetSpatialRef()
    candidates = []
    tbl = catalog.table

    for feat in layer:
        sig_id = feat.GetField("roi_id")
        row_mask = tbl["signature_id"] == sig_id
        if not row_mask.any():
            continue
        if tbl[row_mask]["geometry"][0] != 1:
            continue
        geom = feat.GetGeometryRef()
        if geom is not None:
            candidates.append((sig_id, geom.Clone()))
    ds_read = None

    ogr_driver = ogr.GetDriverByName("GPKG")
    modified = []
    report = []

    for sig_id, geom in candidates:
        roi_gpkg = _tmp(".gpkg")
        roi_ds = ogr_driver.CreateDataSource(roi_gpkg)
        roi_layer = roi_ds.CreateLayer("roi", srs=srs, geom_type=ogr.wkbMultiPolygon)
        feat = ogr.Feature(roi_layer.GetLayerDefn())
        feat.SetGeometry(geom)
        roi_layer.CreateFeature(feat)
        roi_ds.FlushCache()
        roi_ds = None

        ds_raster = _warp_multiband_to_memory(band_paths, roi_gpkg)
        try:
            os.remove(roi_gpkg)
        except OSError:
            pass

        if ds_raster is None:
            log.warning("Warp failed for sig_id=%s — skipped", sig_id)
            continue

        nodata = ds_raster.GetRasterBand(1).GetNoDataValue()
        stack = ds_raster.ReadAsArray().astype(np.float32)
        if stack.ndim == 2:
            stack = stack[np.newaxis]
        if nodata is not None:
            try:
                nodata = float(nodata)
                stack[np.isclose(stack, nodata, equal_nan=False)] = np.nan
            except (TypeError, ValueError):
                stack[stack == nodata] = np.nan

        try:
            mask, pipeline_report = run_pipeline(
                ds_raster,
                steps,
                use_majority_voting=use_voting,
                vote_threshold=vote_threshold,
            )
        except ValueError as err:
            log.warning("Pipeline skipped for sig_id=%s: %s", sig_id, err)
            continue

        clean_stack = stack[:, mask]
        if clean_stack.shape[1] == 0:
            log.warning("All pixels removed for sig_id=%s — skipped", sig_id)
            continue

        means = np.nanmean(clean_stack, axis=1).tolist()
        stds = np.nanstd(clean_stack, axis=1).tolist()

        last = pipeline_report["steps"][-1] if pipeline_report["steps"] else {}
        removed = int(last.get("removed_pixels", 0))
        valid_px = pipeline_report["valid_pixels"]

        try:
            wavelengths = catalog.signatures[sig_id].wavelength.tolist()
            catalog.signatures[sig_id] = _rs_tm.create_spectral_signature_table(
                value_list=means,
                wavelength_list=wavelengths,
                standard_deviation_list=stds,
            )
            idx_arr = np.where(tbl["signature_id"] == sig_id)[0]
            if len(idx_arr) > 0:
                catalog.table["pixel_count"][idx_arr[0]] = int(clean_stack.shape[1])
            _update_roi_geometry(
                geometry_file,
                sig_id,
                mask,
                ds_raster.GetGeoTransform(),
                ds_raster.GetProjection(),
            )
        except Exception as err:
            log.error("Signature update failed for sig_id=%s: %s", sig_id, err)
            continue

        report.append(
            {
                "sig_id": sig_id,
                "removed_pixels": removed,
                "valid_pixels": valid_px,
                "total_raster_pixels": pipeline_report["total_raster_pixels"],
                "removed_percent": 100.0 * removed / valid_px if valid_px else 0.0,
            }
        )
        modified.append(sig_id)

    return modified, report


def _load_fresh_catalog(rs, scpx_path, bandset):
    catalog = rs.spectral_signatures_catalog(bandset=bandset)
    catalog.load(file_path=scpx_path)
    return catalog


def _split_sig_ids(catalog, test_fraction=0.3, random_seed=42):
    rng = random.Random(random_seed)
    tbl = catalog.table

    class_to_sigs = {}
    for i in range(len(tbl)):
        if tbl["geometry"][i] != 1:
            continue
        class_to_sigs.setdefault(tbl["class_id"][i], []).append(tbl["signature_id"][i])

    train_ids, test_ids = [], []
    for sig_ids in class_to_sigs.values():
        shuffled = sig_ids[:]
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_ids.extend(shuffled)
        else:
            n_test = max(1, round(len(shuffled) * test_fraction))
            test_ids.extend(shuffled[:n_test])
            train_ids.extend(shuffled[n_test:])

    return train_ids, test_ids


def _export_test_roi_gpkg(catalog, test_sig_ids, output_path, use_macroclass=False):
    ds_read = ogr.Open(catalog.geometry_file)
    if ds_read is None:
        raise ValueError("Cannot open geometry file: %s" % catalog.geometry_file)

    layer = ds_read.GetLayer()
    srs = layer.GetSpatialRef()
    tbl = catalog.table
    test_set = set(test_sig_ids)

    out_ds = ogr.GetDriverByName("GPKG").CreateDataSource(output_path)
    out_layer = out_ds.CreateLayer("reference", srs=srs, geom_type=ogr.wkbMultiPolygon)
    out_layer.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))

    for feat in layer:
        sig_id = feat.GetField("roi_id")
        if sig_id not in test_set:
            continue
        row_mask = tbl["signature_id"] == sig_id
        if not row_mask.any():
            continue
        row = tbl[row_mask]
        class_id = int(
            row["macroclass_id"][0] if use_macroclass else row["class_id"][0]
        )
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        out_feat = ogr.Feature(out_layer.GetLayerDefn())
        out_feat.SetGeometry(geom.Clone())
        out_feat.SetField("class_id", class_id)
        out_layer.CreateFeature(out_feat)

    out_ds.FlushCache()
    out_ds = None
    ds_read = None


def _filter_catalog_to_train(catalog, train_sig_ids):
    all_ids = list(catalog.table["signature_id"])
    train_set = set(train_sig_ids)
    for sig_id in all_ids:
        if sig_id not in train_set:
            try:
                catalog.remove_signature_by_id(sig_id)
            except Exception:
                pass


def _build_cleaned_catalog(
    rs, scpx_path, bandset, steps, use_voting, vote_threshold, train_sig_ids=None
):
    catalog = _load_fresh_catalog(rs, scpx_path, bandset)
    if train_sig_ids is not None:
        _filter_catalog_to_train(catalog, train_sig_ids)
    band_paths = bandset.get_absolute_paths()
    _, report = _apply_removal_to_catalog(
        catalog, band_paths, steps, use_voting, vote_threshold
    )
    total_removed = sum(r["removed_pixels"] for r in report)
    return catalog, total_removed


def _classify(rs, bandset_catalog, catalog, out_path, algorithm, macroclass):
    # Hiperparametry ustawione 1:1 z domyślnymi wartościami interfejsu wtyczki
    # SCP (run_classifier w interface/classification_tab.py). Bez nich
    # band_classification używa domyślnych remotior_sensus, które różnią się od
    # wtyczki: Random Forest (100 vs 10 drzew, min_samples_split None vs 2),
    # MLP (alpha 0.0001 vs 0.01, batch_size liczony vs "auto") oraz
    # cross_validation (True vs False). Pozostałe parametry pokrywają się z
    # domyślnymi wtyczki i są podane jawnie dla pełnej zgodności.
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
        # Random Forest
        rf_number_trees=10,
        rf_min_samples_split=2,
        rf_max_features=None,
        # Support Vector Machine
        svm_c=1.0,
        svm_gamma="scale",
        svm_kernel="rbf",
        # Multi-Layer Perceptron
        mlp_training_portion=0.9,
        mlp_hidden_layer_sizes=[100],
        mlp_alpha=0.01,
        mlp_learning_rate_init=0.001,
        mlp_max_iter=200,
        mlp_batch_size="auto",
        mlp_activation="relu",
        classification_confidence=False,
    )


def _assess(rs, classification_path, reference_path, out_path, ref_field):
    return rs.cross_classification(
        classification_path=classification_path,
        reference_path=reference_path,
        output_path=out_path,
        vector_field=ref_field if ref_field else None,
        error_matrix=True,
    )


def _parse_accuracy(table_path):
    """Parsuje tabelę błędów cross_classification: OA, Kappa (globalne) oraz
    PA [%], UA [%] i Kappa hat dla każdej klasy. Zwraca (oa, kappa, per_class),
    gdzie per_class to lista dictów {class, pa, ua, kappa_hat}."""

    def _to_float(token):
        try:
            v = float(token)
        except (TypeError, ValueError):
            return None
        return None if v != v else v  # odrzuć NaN

    oa = kappa = None
    classes = pa_vals = ua_vals = kappa_vals = None
    with open(table_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = re.search(r"Overall accuracy \[%\] = ([0-9.]+)", line)
            if m:
                oa = float(m.group(1))
                continue
            m = re.search(r"Kappa hat classification = ([0-9.]+)", line)
            if m:
                kappa = float(m.group(1))
                continue
            if line.startswith("V_Classified,"):
                classes = [
                    t for t in line.split(",")[1:] if t and t != "Total"
                ]
            elif line.startswith("PA [%],"):
                pa_vals = line.split(",")[1:]
            elif line.startswith("UA [%],"):
                ua_vals = line.split(",")[1:]
            elif line.startswith("Kappa hat,"):
                kappa_vals = line.split(",")[1:]

    per_class = []
    for i, cls in enumerate(classes or []):
        per_class.append(
            {
                "class": cls,
                "pa": _to_float(pa_vals[i]) if pa_vals and i < len(pa_vals) else None,
                "ua": _to_float(ua_vals[i]) if ua_vals and i < len(ua_vals) else None,
                "kappa_hat": (
                    _to_float(kappa_vals[i])
                    if kappa_vals and i < len(kappa_vals)
                    else None
                ),
            }
        )
    return oa, kappa, per_class


def _params_label(params):
    return ",".join("%s=%s" % (k, v) for k, v in params.items())


def _build_configs(max_k, methods, grid_search=False):
    for m in methods:
        param_list = (
            _PARAM_GRID.get(m, [_DEFAULT_PARAMS[m]])
            if grid_search
            else [_DEFAULT_PARAMS[m]]
        )
        for params in param_list:
            label = "%s(%s)" % (m, _params_label(params)) if grid_search else m
            yield (label, [(m, params)], False, 1)

    for k in range(2, max_k + 1):
        for combo in itertools.permutations(methods, k):
            steps = [(m, _DEFAULT_PARAMS[m]) for m in combo]
            yield (" → ".join(combo), steps, False, 1)

        vote_thr = k // 2 + 1
        for combo in itertools.combinations(methods, k):
            steps = [(m, _DEFAULT_PARAMS[m]) for m in combo]
            yield ("Ensemble[%s]" % ", ".join(combo), steps, True, vote_thr)


def _to_pl_row(rank, row):
    return {
        "Lp.": rank,
        "Metoda": "Poziom bazowy" if row["label"] == "Baseline" else row["label"],
        "Algorytm": row.get("algorithm", ""),
        "DA_%": ("%.4f" % row["oa"]) if row.get("oa") is not None else "",
        "Kappa": ("%.4f" % row["kappa"]) if row.get("kappa") is not None else "",
        "Usuniete_px": row.get("total_removed", ""),
        "Czas_s": row.get("time_s", ""),
        "Blad": row.get("error", ""),
    }


def _write_polish_csv(path, valid_rows, invalid_rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_PL_FIELDNAMES)
        writer.writeheader()
        for rank, row in enumerate(valid_rows, 1):
            writer.writerow(_to_pl_row(rank, row))
        for row in invalid_rows:
            writer.writerow(_to_pl_row("—", row))


def _fmt_metric(value):
    return ("%.4f" % value) if value is not None else ""


def _write_per_class_csv(path, valid_rows):
    """Zapisuje PA [%], UA [%] i Kappa hat dla każdej klasy, dla każdej
    konfiguracji (metoda × algorytm)."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_PL_PER_CLASS_FIELDNAMES)
        writer.writeheader()
        for rank, row in enumerate(valid_rows, 1):
            metoda = (
                "Poziom bazowy" if row["label"] == "Baseline" else row["label"]
            )
            for pc in row.get("per_class") or []:
                writer.writerow(
                    {
                        "Lp.": rank,
                        "Metoda": metoda,
                        "Algorytm": row.get("algorithm", ""),
                        "Klasa": pc.get("class", ""),
                        "PA_%": _fmt_metric(pc.get("pa")),
                        "UA_%": _fmt_metric(pc.get("ua")),
                        "Kappa_hat": _fmt_metric(pc.get("kappa_hat")),
                    }
                )


def _generate_charts(results, algorithms, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("WARNING: matplotlib not available — skipping chart generation.")
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

    if scatter_pts:
        cmap = plt.cm.get_cmap("tab10", len(algorithms))
        algo_color = {a: cmap(i) for i, a in enumerate(algorithms)}

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

    # ---- PA / UA / Kappa hat per class: best config vs baseline ----
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
            ("kappa_hat", "Kappa hat (per klasa)", (0, 1)),
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
        print("  Per-class accuracy: %s" % path)

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

        # per-class chart for this algorithm (best method vs baseline)
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

    print("  Per-algorithm charts: %s" % per_algo_dir)


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


def _worker_init(
    image_path, scpx_path, rs_n_processes, rs_ram, keep_classifications=False
):
    global _g_rs, _g_bandset_catalog, _g_bandset, _g_scpx_path
    global _g_keep_classifications
    _g_scpx_path = scpx_path
    _g_keep_classifications = keep_classifications
    _g_rs = remotior_sensus.Session(n_processes=rs_n_processes, available_ram=rs_ram)
    _g_bandset_catalog = _g_rs.bandset_catalog()
    _g_bandset_catalog.create_bandset(paths=image_path, bandset_number=1)
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
        train_sig_ids,
        macroclass,
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
                train_sig_ids=train_sig_ids,
            )
        else:
            catalog = _load_fresh_catalog(rs, scpx_path, bandset)
            if train_sig_ids is not None:
                _filter_catalog_to_train(catalog, train_sig_ids)
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

        acc_path = str(accuracy_dir / ("%s.tif" % file_key))
        if os.path.exists(acc_path):
            os.remove(acc_path)
        try:
            acc_output = _assess(
                rs, actual_class_path, reference_path, acc_path, ref_field
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

        oa = kappa = None
        per_class = []
        if acc_output.check:
            _, table_path = acc_output.paths
            if table_path and Path(table_path).exists():
                oa, kappa, per_class = _parse_accuracy(table_path)

        # Rastry były potrzebne tylko do policzenia tabeli dokładności — po
        # sparsowaniu liczb kasujemy je, żeby nie zapełnić dysku (zachowujemy
        # tylko tabele i CSV). Włącz --keep-classifications, by je zostawić.
        if not _g_keep_classifications:
            for _p in (class_path, actual_class_path, acc_path):
                try:
                    os.remove(_p)
                except OSError:
                    pass

        results.append(
            {
                "label": label,
                "algorithm": algorithm,
                "oa": oa,
                "kappa": kappa,
                "per_class": per_class,
                "total_removed": total_removed,
                "time_s": round(time.time() - t0_algo, 1),
                "error": "",
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark outlier-removal pipelines for SCP signatures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image", required=True, help="Path to multispectral .tif image"
    )
    parser.add_argument(
        "--signatures", required=True, help="Path to SCP training signatures (.scpx)"
    )

    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument(
        "--reference",
        help="Path to reference file for accuracy assessment. "
        "Accepted formats: vector .gpkg or .shp (SHP is auto-converted to a "
        "temporary GPKG before use), or raster .tif with class values.",
    )
    ref_group.add_argument(
        "--test-split",
        type=float,
        metavar="FRACTION",
        help="Automatically split ROIs from .scpx into train/test sets "
        "(e.g. 0.3 reserves 30%% of ROIs per class as test). "
        "Mutually exclusive with --reference.",
    )

    parser.add_argument(
        "--ref-field",
        default="class_id",
        help="Field name in reference vector with class values "
        "(default: class_id). Applies to both --reference and --test-split.",
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
        "WARNING: large k with many methods generates factorial numbers of configs "
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
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for train/test split reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed progress output"
    )
    parser.add_argument(
        "--keep-classifications",
        action="store_true",
        help="Keep classification and accuracy rasters on disk. By default they "
        "are deleted right after the accuracy table is computed (only the tables "
        "and CSVs are kept), to avoid filling the disk on large runs.",
    )

    args = parser.parse_args()

    if not (1 <= args.max_k <= len(ALL_METHODS)):
        sys.exit(
            "ERROR: --max-k must be between 1 and %d (number of available methods)."
            % len(ALL_METHODS)
        )

    algorithms = args.algorithm if args.algorithm else ALL_ALGORITHMS

    if args.verbose:
        log.setLevel(logging.INFO)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_dir = out_dir / _DIR_CLASSIFICATIONS
    class_dir.mkdir(exist_ok=True)
    accuracy_dir = out_dir / _DIR_ACCURACY
    accuracy_dir.mkdir(exist_ok=True)

    print("Initializing remotior_sensus ...")
    rs = remotior_sensus.Session(
        n_processes=args.n_processes,
        available_ram=args.ram,
    )

    print("Loading image: %s" % args.image)
    bandset_catalog = rs.bandset_catalog()
    bandset_catalog.create_bandset(paths=args.image, bandset_number=1)
    bandset = bandset_catalog.get(1)
    if bandset is None or bandset.get_band_count() == 0:
        sys.exit("ERROR: Cannot load bands from %s" % args.image)
    print("  => loaded %d bands" % bandset.get_band_count())

    print("Loading signatures: %s" % args.signatures)
    probe_catalog = _load_fresh_catalog(rs, args.signatures, bandset)
    sig_count = len(probe_catalog.table)
    print("  => found %d signatures" % sig_count)
    if sig_count == 0:
        sys.exit("ERROR: No signatures in %s" % args.signatures)

    train_sig_ids = None
    reference_path = args.reference
    ref_field = args.ref_field

    if reference_path and reference_path.lower().endswith(".shp"):
        print("Converting SHP reference to temporary GPKG ...")
        try:
            reference_path = _shp_to_gpkg(reference_path)
            print("  => %s" % reference_path)
        except Exception as err:
            sys.exit("ERROR: SHP to GPKG conversion failed: %s" % err)

    if args.test_split is not None:
        frac = args.test_split
        if not (0.0 < frac < 1.0):
            sys.exit("ERROR: --test-split must be in (0, 1), e.g. 0.3")

        train_sig_ids, test_sig_ids = _split_sig_ids(
            probe_catalog, test_fraction=frac, random_seed=args.random_seed
        )
        print(
            "  => Train: %d   Test: %d   (split %.0f%%/%.0f%%)"
            % (len(train_sig_ids), len(test_sig_ids), (1 - frac) * 100, frac * 100)
        )

        if not test_sig_ids:
            sys.exit(
                "ERROR: Test set is empty — too few signatures per class. "
                "Add more ROIs or reduce --test-split."
            )
        if not train_sig_ids:
            sys.exit("ERROR: Train set is empty. Reduce --test-split.")

        test_gpkg = str(out_dir / _FILE_TEST_REFERENCE)
        _export_test_roi_gpkg(
            probe_catalog, test_sig_ids, test_gpkg, use_macroclass=args.macroclass
        )
        print("  => Test set exported: %s" % test_gpkg)
        reference_path = test_gpkg
        ref_field = "class_id"

    configs = [("Baseline", [], False, 1)]
    configs += list(
        _build_configs(args.max_k, args.methods, grid_search=args.grid_search)
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
            train_sig_ids,
            args.macroclass,
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
            args.image, args.signatures, args.n_processes, args.ram,
            args.keep_classifications,
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
                args.image, args.signatures, worker_n_proc, worker_ram,
                args.keep_classifications,
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

    summary_path = out_dir / _FILE_SUMMARY_CSV
    _write_polish_csv(summary_path, valid, invalid)
    print("Saved: %s" % summary_path)

    per_class_path = out_dir / _FILE_PER_CLASS_CSV
    _write_per_class_csv(per_class_path, valid)
    print("Per-class accuracy: %s" % per_class_path)

    csv_dir = out_dir / _DIR_CSV_PER_ALGO
    csv_dir.mkdir(exist_ok=True)
    for algo in algorithms:
        algo_valid = [r for r in valid if r["algorithm"] == algo]
        algo_invalid = [r for r in invalid if r.get("algorithm") == algo]
        algo_safe = re.sub(r"[^\w\-]", "_", algo)
        algo_path = csv_dir / (_FILE_CSV_PER_ALGO % algo_safe)
        _write_polish_csv(algo_path, algo_valid, algo_invalid)
    print("Per-algorithm files: %s" % csv_dir)

    baseline_oa_by_algo = {
        r["algorithm"]: r["oa"]
        for r in results
        if r["label"] == "Baseline" and r.get("oa") is not None
    }

    print("\n" + "=" * 82)
    print("RESULTS SUMMARY  (sorted by Overall Accuracy, descending)")
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
        print("  %d configs failed (see 'Blad' column in CSV)" % len(invalid))

    # per-class accuracy breakdown for the best configuration
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
            % ("Class", "PA [%]", "UA [%]", "Kappa hat")
        )
        print("-" * 82)
        for pc in best["per_class"]:
            print(
                "%-20s %12s %12s %14s"
                % (
                    str(pc["class"])[:20],
                    _fmt_metric(pc["pa"]) or "-",
                    _fmt_metric(pc["ua"]) or "-",
                    _fmt_metric(pc["kappa_hat"]) or "-",
                )
            )
        print("  Full per-class table (all configs): %s" % per_class_path)

    print("\nGenerating charts ...")
    _generate_charts(results, algorithms, out_dir)

    print("\nFull results (summary): %s" % summary_path)
    print("Per-algorithm results:  %s" % csv_dir)
    print("Classification files:   %s" % class_dir)
    print("Accuracy tables:        %s" % accuracy_dir)


if __name__ == "__main__":
    main()
