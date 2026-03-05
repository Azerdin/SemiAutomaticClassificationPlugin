import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

cfg = __import__(str(__name__).split(".")[0] + ".core.config", fromlist=[""])


def run_pipeline(ds, pipeline_steps):
    stack = ds.ReadAsArray().astype(np.float32)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    stack[stack == 0] = np.nan
    mask_final = None
    for method_name, kwargs in pipeline_steps:
        if method_name.upper() == "MAD":
            stack_filtered, mask = mad_filter_stack(stack, **kwargs)
        else:
            stack_filtered, mask = remove_multivariate_outliers(
                stack, method=method_name.lower(), **kwargs
            )
        mask_final = mask if mask_final is None else (mask_final & mask)
        stack[:, ~mask_final] = np.nan
        print("✔", method_name)
    return stack, mask_final


def mad_filter_stack(stack, threshold=3.5):
    valid_mask = np.all(~np.isnan(stack), axis=0)
    median = np.nanmedian(stack, axis=(1, 2), keepdims=True)
    mad = np.nanmedian(np.abs(stack - median), axis=(1, 2), keepdims=True)
    mad = np.where(mad == 0, 1, mad)
    modified_z = 0.6745 * (stack - median) / mad
    outlier_mask = np.any(np.abs(modified_z) > threshold, axis=0)
    final_mask = valid_mask & (~outlier_mask)
    filtered_stack = np.where(final_mask[None, :, :], stack, np.nan)
    return filtered_stack, final_mask


def remove_multivariate_outliers(stack, method="isolationforest", **kwargs):
    bands, ysize, xsize = stack.shape
    data = stack.transpose(1, 2, 0).reshape(-1, bands)
    valid_mask = ~np.isnan(data).any(axis=1)
    data_valid = data[valid_mask]
    if data_valid.shape[0] < 10:
        raise ValueError("Too few valid pixels in ROI")
    if method == "mahalanobis":
        threshold = kwargs.get("threshold", 3.0)
        mask = mahalanobis_filter(data_valid, threshold)
    elif method == "pca":
        n_components = kwargs.get("n_components", min(bands, data_valid.shape[0]))
        threshold = kwargs.get("threshold", 3.0)
        mask = pca_filter(data_valid, n_components, threshold)
    elif method == "isolationforest":
        contamination = kwargs.get("contamination", 0.01)
        sample_size = kwargs.get("sample_size", 10000)
        mask = isolation_forest_filter(data_valid, contamination, sample_size)
    elif method == "lof":
        n_neighbors = min(kwargs.get("n_neighbors", 20), data_valid.shape[0] - 1)
        contamination = kwargs.get("contamination", 0.01)
        mask = lof_filter(data_valid, n_neighbors, contamination)
    else:
        raise ValueError(f"Unknown method: {method}")
    full_mask = np.zeros(data.shape[0], dtype=bool)
    full_mask[valid_mask] = mask
    full_mask = full_mask.reshape(ysize, xsize)
    cleaned_stack = np.where(full_mask[None, :, :], stack, np.nan)
    return cleaned_stack, full_mask


def mahalanobis_filter(data, threshold):
    cov = MinCovDet().fit(data)
    md = cov.mahalanobis(data)
    return md <= threshold**2


def pca_filter(data, n_components, threshold):
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    cov = MinCovDet().fit(data_pca)
    md = cov.mahalanobis(data_pca)
    return md <= threshold**2


def isolation_forest_filter(data, contamination, sample_size=10000):
    n = data.shape[0]
    if n > sample_size:
        idx = np.random.choice(n, sample_size, replace=False)
        sample = data[idx]
    else:
        sample = data
    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    clf.fit(sample)
    return clf.predict(data) == 1


def lof_filter(data, n_neighbors, contamination):
    if data.shape[0] < 5:
        return np.ones(data.shape[0], dtype=bool)
    clf = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1
    )
    return clf.fit_predict(data) == 1


def remove_outliers_drawing_roi(pipeline_steps):
    util_gdal = cfg.util_gdal
    output_gpkg = cfg.rs.configurations.temp.temporary_file_path(name_suffix=".gpkg")
    bandset_number = cfg.dialog.ui.band_set_comb_spinBox_10.value()
    bandset = cfg.bandset_catalog.get(bandset_number)
    raster_path = bandset.bands[0].path
    roi_tmp = cfg.rs.configurations.temp.temporary_file_path(name_suffix=".gpkg")
    cfg.util_qgis.save_memory_layer_to_geopackage(cfg.temporary_roi, roi_tmp)
    ds = util_gdal.warp_to_memory(raster_path, roi_tmp)
    stack, mask = run_pipeline(ds, pipeline_steps)
    mask_ds, mask_band = util_gdal.create_mask_dataset(
        mask.astype(np.uint8), ds.GetGeoTransform(), ds.GetProjection()
    )
    util_gdal.polygonize_mask(mask_band, ds.GetProjection(), output_gpkg)
    cfg.temporary_roi = cfg.util_qgis.load_geopackage_to_memory_layer(output_gpkg)
    cfg.scp_dock.clear_scp_dock_rubber()
    cfg.scp_dock.add_roi_polygon_to_map(cfg.temporary_roi, 1)
