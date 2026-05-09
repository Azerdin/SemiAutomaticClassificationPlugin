import numpy as np
from osgeo import ogr
from scipy.ndimage import binary_erosion
from scipy.stats import f as f_dist
from sklearn.covariance import MinCovDet, EllipticEnvelope, EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler

cfg = __import__(str(__name__).split(".")[0] + ".core.config", fromlist=[""])


def run_pipeline(
    ds, pipeline_steps, nodata_value=None, use_majority_voting=False, vote_threshold=2
):
    band1 = ds.GetRasterBand(1)
    if nodata_value is None:
        nodata_value = band1.GetNoDataValue()

    stack = ds.ReadAsArray().astype(np.float32)

    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]

    if nodata_value is not None:
        stack[stack == nodata_value] = np.nan

    total_raster_pixels = int(stack.shape[1] * stack.shape[2])
    original_valid_mask = np.all(~np.isnan(stack), axis=0)
    valid_pixels = int(np.sum(original_valid_mask))
    mask_final = original_valid_mask.copy()
    steps = []

    if use_majority_voting:
        stack, mask_final, votes = ensemble_outlier_mask(
            stack, pipeline_steps, vote_threshold=vote_threshold
        )

        removed = int(np.sum(original_valid_mask & ~mask_final))
        removed_pct = 100.0 * removed / valid_pixels if valid_pixels > 0 else 0.0

        steps.append(
            {
                "step": 1,
                "method": "MajorityVoting",
                "removed_pixels": removed,
                "removed_percent": float(removed_pct),
                "vote_threshold": int(vote_threshold),
            }
        )

        print(f"MajorityVoting | removed: {removed} px ({removed_pct:.2f}%)")

        return (
            stack,
            mask_final,
            {
                "total_raster_pixels": total_raster_pixels,
                "valid_pixels": valid_pixels,
                "steps": steps,
            },
        )

    for step_id, (method_name, kwargs) in enumerate(pipeline_steps, start=1):
        method_key = normalize_method_name(method_name)

        if method_key == "mad":
            stack_filtered, mask, scores = mad_filter_stack(stack, **kwargs)
        elif method_key == "erosion":
            stack_filtered, mask, scores = erosion_filter(stack, **kwargs)
        else:
            stack_filtered, mask, scores = remove_multivariate_outliers(
                stack, method=method_key, **kwargs
            )

        mask_final &= mask
        stack = np.where(mask_final[None, :, :], stack_filtered, np.nan)

        removed = int(np.sum(original_valid_mask & ~mask_final))
        removed_pct = 100.0 * removed / valid_pixels if valid_pixels > 0 else 0.0

        steps.append(
            {
                "step": step_id,
                "method": method_name,
                "removed_pixels": removed,
                "removed_percent": float(removed_pct),
            }
        )

        print(f"{method_name} | removed: {removed} px ({removed_pct:.2f}%)")

    return (
        stack,
        mask_final,
        {
            "total_raster_pixels": total_raster_pixels,
            "valid_pixels": valid_pixels,
            "steps": steps,
        },
    )


def scale_data(data, scaler="standard"):
    if scaler is None:
        return data, None

    if scaler == "standard":
        sc = StandardScaler()
    elif scaler == "robust":
        sc = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler}")

    data_scaled = sc.fit_transform(data)
    return data_scaled, sc


# Extracts non-NaN pixels from a band stack into a flat array.
def prepare_valid_pixels(stack):
    bands, ysize, xsize = stack.shape
    data = stack.transpose(1, 2, 0).reshape(-1, bands)
    valid_mask = ~np.isnan(data).any(axis=1)
    data_valid = data[valid_mask]

    return data_valid, valid_mask, ysize, xsize


# Maps a per-valid-pixel boolean mask back to the 2D grid.
def rebuild_mask(mask_valid, valid_mask_flat, ysize, xsize):
    full_mask = np.zeros(valid_mask_flat.shape[0], dtype=bool)
    full_mask[valid_mask_flat] = mask_valid
    return full_mask.reshape(ysize, xsize)


# Strips non-alphanumeric chars and lowercases a method name for comparison.
def normalize_method_name(method_name):
    cleaned = str(method_name).lower()
    return "".join(ch for ch in cleaned if ch.isalnum())


def mad_filter_stack(stack, threshold=3.5):
    valid_mask = np.all(~np.isnan(stack), axis=0)

    median = np.nanmedian(stack, axis=(1, 2), keepdims=True)
    mad = np.nanmedian(np.abs(stack - median), axis=(1, 2), keepdims=True)
    mad = np.where(mad == 0, 1, mad)

    modified_z = 0.6745 * (stack - median) / mad
    outlier_mask = np.any(np.abs(modified_z) > threshold, axis=0)

    final_mask = valid_mask & (~outlier_mask)
    filtered_stack = np.where(final_mask[None, :, :], stack, np.nan)

    scores = np.nanmax(np.abs(modified_z), axis=0)

    return filtered_stack, final_mask, scores


def erosion_filter(stack, iterations=1):
    valid_mask = np.all(~np.isnan(stack), axis=0)
    try:
        eroded = binary_erosion(valid_mask, iterations=iterations)
    except Exception:
        return stack.copy(), valid_mask, np.zeros(valid_mask.shape, dtype=np.float32)

    if not np.any(eroded):
        return stack.copy(), valid_mask, np.zeros(valid_mask.shape, dtype=np.float32)

    final_mask = valid_mask & eroded
    filtered_stack = np.where(final_mask[None, :, :], stack, np.nan)
    scores = (~eroded & valid_mask).astype(np.float32)
    return filtered_stack, final_mask, scores


def remove_multivariate_outliers(stack, method="isolationforest", **kwargs):
    bands, ysize, xsize = stack.shape
    method = normalize_method_name(method)

    data_valid, valid_mask, ysize, xsize = prepare_valid_pixels(stack)

    if data_valid.shape[0] < max(10, bands + 2):
        raise ValueError("Too few valid pixels in ROI")

    scaler_name = kwargs.get("scaler", "standard")
    data_scaled, _ = scale_data(data_valid, scaler=scaler_name)

    if method == "mahalanobis":
        threshold = kwargs.get("threshold", 3.0)
        mask_valid, scores = mahalanobis_filter(data_scaled, threshold)

    elif method == "robustmahalanobis":
        threshold = kwargs.get("threshold", 3.0)
        mask_valid, scores = mahalanobis_filter(data_scaled, threshold, robust=True)

    elif method == "bandzscore":
        threshold = kwargs.get("threshold", 3.0)
        mask_valid, scores = band_zscore_filter(data_valid, threshold)

    elif method == "percentile":
        lower_pct = kwargs.get("lower_pct", 0.01)
        upper_pct = kwargs.get("upper_pct", 0.99)
        mask_valid, scores = percentile_filter(data_valid, lower_pct, upper_pct)

    elif method == "iqr":
        factor = kwargs.get("factor", 1.5)
        mask_valid, scores = iqr_filter(data_valid, factor)

    elif method == "pca":
        n_components = kwargs.get("n_components", min(bands, data_valid.shape[0], 5))
        threshold = kwargs.get("threshold", 3.0)
        mask_valid, scores = pca_filter(data_scaled, n_components, threshold)

    elif method == "pcareconstruction":
        n_components = kwargs.get("n_components", min(bands, max(2, bands // 2)))
        contamination = kwargs.get("contamination", 0.01)
        mask_valid, scores = pca_reconstruction_filter(
            data_scaled, n_components, contamination
        )

    elif method == "isolationforest":
        contamination = kwargs.get("contamination", 0.01)
        sample_size = kwargs.get("sample_size", 10000)
        mask_valid, scores = isolation_forest_filter(
            data_scaled, contamination, sample_size
        )

    elif method == "lof":
        n_neighbors = min(kwargs.get("n_neighbors", 20), data_valid.shape[0] - 1)
        contamination = kwargs.get("contamination", 0.01)
        mask_valid, scores = lof_filter(data_scaled, n_neighbors, contamination)

    elif method == "oneclasssvm":
        nu = kwargs.get("nu", 0.01)
        gamma = kwargs.get("gamma", "scale")
        kernel = kwargs.get("kernel", "rbf")
        mask_valid, scores = one_class_svm_filter(data_scaled, nu, gamma, kernel)

    elif method == "knn":
        n_neighbors = min(kwargs.get("n_neighbors", 5), data_valid.shape[0] - 1)
        contamination = kwargs.get("contamination", 0.01)
        mask_valid, scores = knn_filter(data_scaled, n_neighbors, contamination)

    elif method in ("elliptic", "ellipticenvelope"):
        contamination = kwargs.get("contamination", 0.01)
        mask_valid, scores = elliptic_filter(data_scaled, contamination)

    elif method == "hotellingt2":
        alpha = kwargs.get("alpha", 0.05)
        mask_valid, scores = hotelling_t2_filter(data_scaled, alpha)

    elif method == "gmm":
        n_components = kwargs.get("n_components", 2)
        contamination = kwargs.get("contamination", 0.01)
        mask_valid, scores = gmm_filter(data_scaled, n_components, contamination)

    elif method == "sam":
        contamination = kwargs.get("contamination", 0.01)
        reference = kwargs.get("reference", None)
        mask_valid, scores = sam_filter(data_valid, contamination, reference)

    else:
        raise ValueError(f"Unknown method: {method}")

    full_mask = rebuild_mask(mask_valid, valid_mask, ysize, xsize)
    cleaned_stack = np.where(full_mask[None, :, :], stack, np.nan)

    return cleaned_stack, full_mask, scores


def mahalanobis_filter(data, threshold, robust=False):
    cov = (
        MinCovDet(random_state=42).fit(data)
        if robust
        else EmpiricalCovariance().fit(data)
    )
    md = cov.mahalanobis(data)
    mask = md <= threshold**2
    return mask, md


def pca_filter(data, n_components, threshold):
    pca = PCA(n_components=n_components, random_state=42)
    data_pca = pca.fit_transform(data)
    cov = MinCovDet(random_state=42).fit(data_pca)
    md = cov.mahalanobis(data_pca)
    mask = md <= threshold**2
    return mask, md


def pca_reconstruction_filter(data, n_components, contamination):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(data)
    X_rec = pca.inverse_transform(X_pca)
    rec_error = np.linalg.norm(data - X_rec, axis=1)

    thr = np.quantile(rec_error, 1 - contamination)
    mask = rec_error <= thr
    return mask, rec_error


def band_zscore_filter(data, threshold=3.0):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std_safe = np.where(std == 0, 1.0, std)

    z = np.abs((data - mean) / std_safe)
    scores = np.max(z, axis=1)
    mask = scores <= threshold
    return mask, scores


def percentile_filter(data, lower_pct=0.01, upper_pct=0.99):
    lower = np.quantile(data, lower_pct, axis=0)
    upper = np.quantile(data, upper_pct, axis=0)

    below = data < lower
    above = data > upper
    score = np.maximum(
        np.where(below, (lower - data) / (np.abs(lower) + 1e-12), 0.0),
        np.where(above, (data - upper) / (np.abs(upper) + 1e-12), 0.0),
    )
    scores = np.max(score, axis=1)
    mask = np.all(~(below | above), axis=1)
    return mask, scores


def iqr_filter(data, factor=1.5):
    q1 = np.quantile(data, 0.25, axis=0)
    q3 = np.quantile(data, 0.75, axis=0)
    iqr = q3 - q1
    iqr_safe = np.where(iqr == 0, 1.0, iqr)

    lower = q1 - factor * iqr_safe
    upper = q3 + factor * iqr_safe

    below = data < lower
    above = data > upper
    score = np.maximum(
        np.where(below, (lower - data) / (iqr_safe + 1e-12), 0.0),
        np.where(above, (data - upper) / (iqr_safe + 1e-12), 0.0),
    )
    scores = np.max(score, axis=1)
    mask = np.all(~(below | above), axis=1)
    return mask, scores


def isolation_forest_filter(data, contamination, sample_size=10000):
    n = data.shape[0]

    if n > sample_size:
        idx = np.random.choice(n, sample_size, replace=False)
        sample = data[idx]
    else:
        sample = data

    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    clf.fit(sample)

    pred = clf.predict(data)
    scores = -clf.score_samples(data)

    return pred == 1, scores


def lof_filter(data, n_neighbors, contamination):
    if data.shape[0] < 5:
        return np.ones(data.shape[0], dtype=bool), np.zeros(data.shape[0])

    clf = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1
    )

    pred = clf.fit_predict(data)
    scores = -clf.negative_outlier_factor_

    return pred == 1, scores


def one_class_svm_filter(data, nu=0.01, gamma="scale", kernel="rbf"):
    clf = OneClassSVM(nu=nu, gamma=gamma, kernel=kernel)
    pred = clf.fit_predict(data)
    scores = -clf.decision_function(data).ravel()
    return pred == 1, scores


def knn_filter(data, n_neighbors=5, contamination=0.01):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(data)
    distances, _ = nbrs.kneighbors(data)

    score = distances[:, -1]

    thr = np.quantile(score, 1 - contamination)
    mask = score <= thr

    return mask, score


def elliptic_filter(data, contamination=0.01):
    clf = EllipticEnvelope(contamination=contamination, random_state=42)
    pred = clf.fit_predict(data)
    scores = -clf.decision_function(data)
    return pred == 1, scores


def hotelling_t2_filter(data, alpha=0.05):
    n, p = data.shape
    if n <= p + 1:
        return np.ones(n, dtype=bool), np.zeros(n)
    try:
        cov = EmpiricalCovariance().fit(data)
        t2 = cov.mahalanobis(data)  # squared Mahalanobis = T^2 statistic
        f_crit = f_dist.ppf(1.0 - alpha, p, n - p)
        t2_crit = p * (n - 1) / (n - p) * f_crit
        mask = t2 <= t2_crit
    except Exception:
        return np.ones(n, dtype=bool), np.zeros(n)
    return mask, t2


def gmm_filter(data, n_components=2, contamination=0.01):
    try:
        gmm = GaussianMixture(
            n_components=n_components, covariance_type="full", random_state=42
        )
        gmm.fit(data)
        log_lik = gmm.score_samples(data)
        thr = np.quantile(log_lik, contamination)
        mask = log_lik >= thr
        scores = -log_lik
    except Exception:
        return np.ones(data.shape[0], dtype=bool), np.zeros(data.shape[0])
    return mask, scores


def sam_filter(data, contamination=0.01, reference=None):
    if reference is None:
        reference = np.nanmean(data, axis=0)

    reference = np.asarray(reference, dtype=np.float32)

    numerator = np.sum(data * reference, axis=1)
    denominator = (np.linalg.norm(data, axis=1) * np.linalg.norm(reference)) + 1e-12

    cos_theta = np.clip(numerator / denominator, -1, 1)
    angles = np.arccos(cos_theta)

    thr = np.quantile(angles, 1 - contamination)
    mask = angles <= thr

    return mask, angles


def ensemble_outlier_mask(stack, methods_config, vote_threshold=2):
    masks = []

    for method_name, kwargs in methods_config:
        method_key = normalize_method_name(method_name)

        if method_key == "mad":
            _, mask, _ = mad_filter_stack(stack, **kwargs)
        elif method_key == "erosion":
            _, mask, _ = erosion_filter(stack, **kwargs)
        else:
            _, mask, _ = remove_multivariate_outliers(
                stack, method=method_key, **kwargs
            )

        masks.append(mask.astype(np.uint8))

    masks = np.stack(masks, axis=0)
    votes = np.sum(masks, axis=0)

    final_mask = votes >= vote_threshold
    cleaned_stack = np.where(final_mask[None, :, :], stack, np.nan)

    return cleaned_stack, final_mask, votes


def run_pipeline_on_signature(
    data,
    pipeline_steps,
    use_majority_voting=False,
    vote_threshold=2,
):
    if data.ndim != 2:
        raise ValueError("Signature sample matrix must be 2D [n_pixels, bands]")

    valid_mask = ~np.isnan(data).any(axis=1)
    if np.sum(valid_mask) < 10:
        raise ValueError("Too few valid samples in signature")

    if use_majority_voting:
        cleaned, mask_final = ensemble_outlier_mask_signature(
            data[valid_mask], pipeline_steps, vote_threshold=vote_threshold
        )

        full_mask = np.zeros(len(data), dtype=bool)
        inlier_idx = np.where(valid_mask)[0]
        full_mask[inlier_idx[mask_final]] = True

        removed = int(np.sum(valid_mask & ~full_mask))
        total = int(np.sum(valid_mask))
        report = [
            {
                "step": 1,
                "method": "MajorityVoting",
                "removed_pixels": removed,
                "removed_percent": float(100.0 * removed / total if total > 0 else 0.0),
                "vote_threshold": int(vote_threshold),
            }
        ]
        return data[full_mask], full_mask, report

    mask_final = valid_mask.copy()
    current_data = data[valid_mask]
    report = []

    for step_id, (method_name, kwargs) in enumerate(pipeline_steps, start=1):
        method_key = normalize_method_name(method_name)

        if method_key == "mad":
            _, current_mask = mad_filter_signature(current_data, **kwargs)
        elif method_key == "erosion":
            # Spatial erosion has no meaning for a flat signature matrix — skip.
            current_mask = np.ones(len(current_data), dtype=bool)
        else:
            _, current_mask = remove_multivariate_outliers_signature(
                current_data, method=method_key, **kwargs
            )

        inlier_indices = np.where(mask_final)[0]
        mask_final[inlier_indices[~current_mask]] = False
        current_data = current_data[current_mask]

        removed = int(np.sum(valid_mask & ~mask_final))
        total = int(np.sum(valid_mask))
        report.append(
            {
                "step": step_id,
                "method": method_name,
                "removed_pixels": removed,
                "removed_percent": float(100.0 * removed / total if total > 0 else 0.0),
            }
        )

    return data[mask_final], mask_final, report


def mad_filter_signature(data, threshold=3.5):
    median = np.nanmedian(data, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(data - median), axis=0, keepdims=True)
    mad = np.where(mad == 0, 1.0, mad)

    modified_z = 0.6745 * (data - median) / mad
    outlier_mask = np.any(np.abs(modified_z) > threshold, axis=1)
    final_mask = ~outlier_mask
    filtered = data[final_mask]
    return filtered, final_mask


def remove_multivariate_outliers_signature(data, method="isolationforest", **kwargs):
    data_T = np.asarray(data, dtype=np.float32)
    if data_T.ndim != 2:
        raise ValueError("Signature data must be 2D [n_pixels, bands]")

    scaler_name = kwargs.get("scaler", "standard")
    data_scaled, _ = scale_data(data_T, scaler=scaler_name)
    method = normalize_method_name(method)

    if method == "mahalanobis":
        mask_valid, _ = mahalanobis_filter(data_scaled, kwargs.get("threshold", 3.0))
    elif method == "robustmahalanobis":
        mask_valid, _ = mahalanobis_filter(
            data_scaled, kwargs.get("threshold", 3.0), robust=True
        )
    elif method == "bandzscore":
        mask_valid, _ = band_zscore_filter(data_T, kwargs.get("threshold", 3.0))
    elif method == "percentile":
        mask_valid, _ = percentile_filter(
            data_T, kwargs.get("lower_pct", 0.01), kwargs.get("upper_pct", 0.99)
        )
    elif method == "iqr":
        mask_valid, _ = iqr_filter(data_T, kwargs.get("factor", 1.5))
    elif method == "pca":
        n_components = kwargs.get(
            "n_components", min(data_T.shape[1], data_T.shape[0], 5)
        )
        mask_valid, _ = pca_filter(
            data_scaled, n_components, kwargs.get("threshold", 3.0)
        )
    elif method == "pcareconstruction":
        n_components = kwargs.get(
            "n_components", min(data_T.shape[1], max(2, data_T.shape[1] // 2))
        )
        mask_valid, _ = pca_reconstruction_filter(
            data_scaled, n_components, kwargs.get("contamination", 0.01)
        )
    elif method == "isolationforest":
        mask_valid, _ = isolation_forest_filter(
            data_scaled,
            kwargs.get("contamination", 0.01),
            kwargs.get("sample_size", 10000),
        )
    elif method == "lof":
        n_neighbors = min(kwargs.get("n_neighbors", 20), data_T.shape[0] - 1)
        mask_valid, _ = lof_filter(
            data_scaled, n_neighbors, kwargs.get("contamination", 0.01)
        )
    elif method == "oneclasssvm":
        mask_valid, _ = one_class_svm_filter(
            data_scaled,
            kwargs.get("nu", 0.01),
            kwargs.get("gamma", "scale"),
            kwargs.get("kernel", "rbf"),
        )
    elif method == "knn":
        n_neighbors = min(kwargs.get("n_neighbors", 5), data_T.shape[0] - 1)
        mask_valid, _ = knn_filter(
            data_scaled, n_neighbors, kwargs.get("contamination", 0.01)
        )
    elif method in ("elliptic", "ellipticenvelope"):
        mask_valid, _ = elliptic_filter(data_scaled, kwargs.get("contamination", 0.01))
    elif method == "hotellingt2":
        mask_valid, _ = hotelling_t2_filter(data_scaled, kwargs.get("alpha", 0.05))
    elif method == "gmm":
        n_components = kwargs.get("n_components", 2)
        mask_valid, _ = gmm_filter(
            data_scaled, n_components, kwargs.get("contamination", 0.01)
        )
    elif method == "sam":
        mask_valid, _ = sam_filter(
            data_T, kwargs.get("contamination", 0.01), kwargs.get("reference", None)
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return data_T[mask_valid], mask_valid


def ensemble_outlier_mask_signature(data, methods_config, vote_threshold=2):
    masks = []
    for method_name, kwargs in methods_config:
        method_key = normalize_method_name(method_name)
        if method_key == "mad":
            _, mask = mad_filter_signature(data, **kwargs)
        elif method_key == "erosion":
            mask = np.ones(len(data), dtype=bool)
        else:
            _, mask = remove_multivariate_outliers_signature(
                data, method=method_key, **kwargs
            )
        masks.append(mask.astype(np.uint8))

    masks = np.stack(masks, axis=0)
    votes = np.sum(masks, axis=0)
    final_mask = votes >= vote_threshold
    return data[final_mask], final_mask


def _process_catalog_features(
    new_catalog, feature_predicate, pipeline_steps, use_majority_voting, vote_threshold
):
    geometry_file = new_catalog.geometry_file
    bandset = new_catalog.bandset
    if bandset is None:
        raise ValueError("No bandset attached to signature catalog")

    band_paths = bandset.get_absolute_paths()

    ds_read = ogr.Open(geometry_file)
    if ds_read is None:
        raise ValueError(f"Cannot open geometry file: {geometry_file}")

    layer = ds_read.GetLayer()
    srs = layer.GetSpatialRef()

    candidates = []
    for feat in layer:
        sig_id = feat.GetField("roi_id")
        mc_id = feat.GetField("macroclass_id")
        cl_id = feat.GetField("class_id")

        if not feature_predicate(sig_id, mc_id, cl_id):
            continue

        tbl = new_catalog.table
        row_mask = tbl["signature_id"] == sig_id
        if not row_mask.any():
            continue
        if tbl[row_mask]["geometry"][0] != 1:
            continue

        geom = feat.GetGeometryRef()
        if geom is None:
            continue

        candidates.append((sig_id, geom.Clone()))

    ds_read = None

    ogr_driver = ogr.GetDriverByName("GPKG")
    modified_sig_ids = []
    report = []
    total = len(candidates)

    for i, (sig_id, geom) in enumerate(candidates):
        step = int((i + 1) * 80 / max(total, 1))
        cfg.ui_utils.update_bar(
            step,
            "Removing outliers: ROI %d/%d" % (i + 1, total),
            percentage=step,
        )
        roi_tmp = cfg.rs.configurations.temp.temporary_file_path(name_suffix=".gpkg")
        roi_ds = ogr_driver.CreateDataSource(roi_tmp)
        roi_layer = roi_ds.CreateLayer("roi", srs=srs, geom_type=ogr.wkbMultiPolygon)
        new_feat = ogr.Feature(roi_layer.GetLayerDefn())
        new_feat.SetGeometry(geom)
        roi_layer.CreateFeature(new_feat)
        roi_ds.FlushCache()
        roi_ds = None

        ds_raster = cfg.util_gdal.warp_multiband_to_memory(band_paths, roi_tmp)
        if ds_raster is None:
            cfg.logger.log.error(
                "warp_multiband_to_memory returned None for sig_id=%s" % sig_id
            )
            continue

        try:
            stack, mask, pipeline_report = run_pipeline(
                ds_raster,
                pipeline_steps,
                use_majority_voting=use_majority_voting,
                vote_threshold=vote_threshold,
            )
            step_report = pipeline_report["steps"]
            valid_pixels = pipeline_report["valid_pixels"]
            total_raster_pixels = pipeline_report["total_raster_pixels"]
        except ValueError as err:
            cfg.logger.log.error("run_pipeline skipped sig_id=%s: %s" % (sig_id, err))
            continue

        mask_ds, mask_band = cfg.util_gdal.create_mask_dataset(
            mask.astype(np.uint8),
            ds_raster.GetGeoTransform(),
            ds_raster.GetProjection(),
        )
        out_gpkg = cfg.rs.configurations.temp.temporary_file_path(name_suffix=".gpkg")
        cfg.util_gdal.polygonize_mask(mask_band, ds_raster.GetProjection(), out_gpkg)

        out_ds = ogr.Open(out_gpkg)
        if out_ds is None:
            continue

        union_geom = None
        for out_feat in out_ds.GetLayer():
            g = out_feat.GetGeometryRef()
            if g is not None:
                union_geom = g.Clone() if union_geom is None else union_geom.Union(g)
        out_ds = None

        if union_geom is None:
            cfg.logger.log.error(
                "Polygonize produced no geometry for sig_id=%s" % sig_id
            )
            continue

        geo_ds = ogr.Open(geometry_file, 1)
        if geo_ds is not None:
            geo_layer = geo_ds.GetLayer()
            geo_layer.SetAttributeFilter("roi_id = '%s'" % sig_id)
            geo_feat = geo_layer.GetNextFeature()
            if geo_feat is not None:
                geo_feat.SetGeometry(union_geom)
                geo_layer.SetFeature(geo_feat)
            geo_layer.SetAttributeFilter(None)
            geo_ds.FlushCache()
            geo_ds = None

        last_step = step_report[-1] if step_report else {}
        removed = int(last_step.get("removed_pixels", 0))
        report.append(
            {
                "sig_id": sig_id,
                "removed_pixels": removed,
                "valid_pixels": valid_pixels,
                "total_raster_pixels": total_raster_pixels,
                "removed_percent": (
                    100.0 * removed / valid_pixels if valid_pixels > 0 else 0.0
                ),
                "steps": step_report,
            }
        )
        modified_sig_ids.append(sig_id)

        print(
            "Outliers removed | sig=%s | removed=%d px (%.2f%%)"
            % (sig_id, removed, report[-1]["removed_percent"])
        )

    return modified_sig_ids, report


def _recalculate_signatures_silent(new_catalog, modified_sig_ids):
    total = len(modified_sig_ids)
    for idx, sig_id in enumerate(modified_sig_ids):
        step = 80 + int((idx + 1) * 20 / max(total, 1))
        cfg.ui_utils.update_bar(
            step,
            "Recalculating signature %d/%d" % (idx + 1, total),
            percentage=step,
        )
        tbl = new_catalog.table
        row_mask = tbl["signature_id"] == sig_id
        if not row_mask.any():
            continue

        row = tbl[row_mask]
        macroclass_id = row["macroclass_id"][0]
        class_id = row["class_id"][0]
        class_name = row["class_name"][0]
        color = row["color"][0]
        macroclass_name = new_catalog.macroclasses.get(
            macroclass_id, str(macroclass_id)
        )

        try:
            new_catalog.merge_signatures_by_id(
                signature_id_list=[sig_id],
                calculate_signature=True,
                macroclass_id=macroclass_id,
                class_id=class_id,
                macroclass_name=macroclass_name,
                class_name=class_name,
                color_string=color,
            )
            new_catalog.remove_signature_by_id(sig_id)
        except Exception as err:
            cfg.logger.log.error(
                "Signature recalculation failed for %s: %s" % (sig_id, err)
            )


def build_cleaned_catalog_copy(
    pipeline_steps, use_majority_voting=False, vote_threshold=2
):
    if cfg.scp_training is None or cfg.scp_training.signature_catalog is None:
        raise ValueError("No signature catalog loaded")

    cfg.ui_utils.add_progress_bar()
    try:
        new_catalog = cfg.scp_training.signature_catalog_copy()

        modified_sig_ids, report = _process_catalog_features(
            new_catalog,
            lambda *_: True,
            pipeline_steps,
            use_majority_voting,
            vote_threshold,
        )

        _recalculate_signatures_silent(new_catalog, modified_sig_ids)

        return new_catalog, report
    finally:
        cfg.ui_utils.remove_progress_bar(sound=False)


def _recalculate_signatures_and_reload(new_catalog, modified_sig_ids):
    total = len(modified_sig_ids)
    for idx, sig_id in enumerate(modified_sig_ids):
        step = 80 + int((idx + 1) * 20 / max(total, 1))
        cfg.ui_utils.update_bar(
            step,
            "Recalculating signature %d/%d" % (idx + 1, total),
            percentage=step,
        )
        tbl = new_catalog.table
        row_mask = tbl["signature_id"] == sig_id
        if not row_mask.any():
            cfg.logger.log.error(
                "sig_id %s not found in table, skipping recalculation" % sig_id
            )
            continue

        row = tbl[row_mask]
        macroclass_id = row["macroclass_id"][0]
        class_id = row["class_id"][0]
        class_name = row["class_name"][0]
        color = row["color"][0]
        macroclass_name = new_catalog.macroclasses.get(
            macroclass_id, str(macroclass_id)
        )

        try:
            new_catalog.merge_signatures_by_id(
                signature_id_list=[sig_id],
                calculate_signature=True,
                macroclass_id=macroclass_id,
                class_id=class_id,
                macroclass_name=macroclass_name,
                class_name=class_name,
                color_string=color,
            )
            new_catalog.remove_signature_by_id(sig_id)
        except Exception as err:
            cfg.logger.log.error(
                "Signature recalculation failed for %s: %s" % (sig_id, err)
            )

    cfg.scp_training.set_signature_catalog(new_catalog)
    cfg.scp_training.roi_signature_table_tree()
    cfg.dock_class_dlg.ui.undo_save_Button.setEnabled(True)
    cfg.dock_class_dlg.ui.redo_save_Button.setEnabled(False)
    if cfg.project_registry[cfg.reg_save_training_input_check] == 2:
        cfg.scp_training.save_signature_catalog()


def remove_outliers_all_signatures(
    pipeline_steps,
    use_majority_voting=False,
    vote_threshold=2,
):
    if cfg.scp_training is None or cfg.scp_training.signature_catalog is None:
        raise ValueError("No signature catalog loaded")

    cfg.ui_utils.add_progress_bar()
    try:
        cfg.scp_training.save_temporary_signature_catalog()
        new_catalog = cfg.scp_training.signature_catalog_copy()

        modified_sig_ids, report = _process_catalog_features(
            new_catalog,
            lambda *_: True,
            pipeline_steps,
            use_majority_voting,
            vote_threshold,
        )

        _recalculate_signatures_and_reload(new_catalog, modified_sig_ids)
        return report
    finally:
        cfg.ui_utils.remove_progress_bar(sound=False)


def remove_outliers_selected_signatures(
    signature_ids,
    pipeline_steps,
    use_majority_voting=False,
    vote_threshold=2,
):
    if cfg.scp_training is None or cfg.scp_training.signature_catalog is None:
        raise ValueError("No signature catalog loaded")

    sig_id_set = {str(s) for s in signature_ids}

    cfg.ui_utils.add_progress_bar()
    try:
        cfg.scp_training.save_temporary_signature_catalog()
        new_catalog = cfg.scp_training.signature_catalog_copy()

        modified_sig_ids, report = _process_catalog_features(
            new_catalog,
            lambda sig_id, *_: str(sig_id) in sig_id_set,
            pipeline_steps,
            use_majority_voting,
            vote_threshold,
        )

        _recalculate_signatures_and_reload(new_catalog, modified_sig_ids)
        return report
    finally:
        cfg.ui_utils.remove_progress_bar(sound=False)


def remove_outliers_by_class(
    macroclass,
    classId,
    pipeline_steps,
    use_majority_voting=False,
    vote_threshold=2,
):
    if cfg.scp_training is None or cfg.scp_training.signature_catalog is None:
        raise ValueError("No signature catalog loaded")

    cfg.ui_utils.add_progress_bar()
    try:
        cfg.scp_training.save_temporary_signature_catalog()
        new_catalog = cfg.scp_training.signature_catalog_copy()

        modified_sig_ids, report = _process_catalog_features(
            new_catalog,
            lambda _, mc_id, cl_id: (
                int(mc_id) == int(macroclass) and int(cl_id) == int(classId)
            ),
            pipeline_steps,
            use_majority_voting,
            vote_threshold,
        )

        _recalculate_signatures_and_reload(new_catalog, modified_sig_ids)
        return report
    finally:
        cfg.ui_utils.remove_progress_bar(sound=False)


def remove_outliers_drawing_roi(
    pipeline_steps, use_majority_voting=False, vote_threshold=2
):
    cfg.ui_utils.add_progress_bar()
    try:
        util_gdal = cfg.util_gdal
        output_gpkg = cfg.rs.configurations.temp.temporary_file_path(
            name_suffix=".gpkg"
        )

        bandset_number = cfg.dialog.ui.band_set_comb_spinBox_10.value()
        bandset = cfg.bandset_catalog.get(bandset_number)
        band_paths = bandset.get_absolute_paths()

        roi_tmp = cfg.rs.configurations.temp.temporary_file_path(name_suffix=".gpkg")
        cfg.util_qgis.save_memory_layer_to_geopackage(cfg.temporary_roi, roi_tmp)

        cfg.ui_utils.update_bar(20, "Clipping raster to ROI...", percentage=20)
        ds = util_gdal.warp_multiband_to_memory(band_paths, roi_tmp)

        cfg.ui_utils.update_bar(50, "Running outlier pipeline...", percentage=50)
        stack, mask, report = run_pipeline(
            ds,
            pipeline_steps,
            use_majority_voting=use_majority_voting,
            vote_threshold=vote_threshold,
        )

        cfg.ui_utils.update_bar(80, "Polygonizing result...", percentage=80)
        _mask_ds, mask_band = util_gdal.create_mask_dataset(
            mask.astype(np.uint8), ds.GetGeoTransform(), ds.GetProjection()
        )
        util_gdal.polygonize_mask(mask_band, ds.GetProjection(), output_gpkg)

        out_ds = ogr.Open(output_gpkg)
        union_geom = None
        srs = None
        if out_ds is not None:
            out_layer = out_ds.GetLayer()
            srs = out_layer.GetSpatialRef()
            for out_feat in out_layer:
                g = out_feat.GetGeometryRef()
                if g is not None:
                    union_geom = (
                        g.Clone() if union_geom is None else union_geom.Union(g)
                    )
            out_ds = None

        if union_geom is None:
            cfg.logger.log.error("Polygonize produced no geometry for drawn ROI")
            return stack, mask, report

        ogr_driver = ogr.GetDriverByName("GPKG")
        merged_gpkg = cfg.rs.configurations.temp.temporary_file_path(
            name_suffix=".gpkg"
        )
        merged_ds = ogr_driver.CreateDataSource(merged_gpkg)
        merged_layer = merged_ds.CreateLayer(
            "roi", srs=srs, geom_type=ogr.wkbMultiPolygon
        )
        new_feat = ogr.Feature(merged_layer.GetLayerDefn())
        new_feat.SetGeometry(union_geom)
        merged_layer.CreateFeature(new_feat)
        merged_ds.FlushCache()
        merged_ds = None

        cfg.temporary_roi = cfg.util_qgis.load_geopackage_to_memory_layer(merged_gpkg)
        cfg.scp_dock.clear_scp_dock_rubber()
        cfg.scp_dock.add_roi_polygon_to_map(cfg.temporary_roi, 1)

        return stack, mask, report
    finally:
        cfg.ui_utils.remove_progress_bar(sound=False)
