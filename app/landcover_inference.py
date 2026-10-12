from __future__ import annotations

import os
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import joblib
import numpy as np
import rasterio
from rasterio.windows import Window

from .config import settings


ALLOWED_GEOTIFF_EXTENSIONS = {".tif", ".tiff"}
DEFAULT_BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI"]
DEFAULT_BAND_INDEXES = [1, 2, 3, 4, 5, 6, 7]
WINDOW_SIZE = 1024

CLASS_COLORS = {
    0: (0, 0, 0, 0),
    1: (0, 100, 0, 255),
    2: (0, 102, 255, 255),
    3: (210, 180, 80, 255),
    4: (220, 40, 40, 255),
    5: (120, 200, 80, 255),
}

_model_lock = threading.Lock()


def _safe_relative_path(value: str) -> Path:
    raw = str(value or "").strip().replace("\\", "/")
    relative = Path(raw)

    if not raw or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("relative_path phải là đường dẫn tương đối an toàn")

    if relative.suffix.lower() not in ALLOWED_GEOTIFF_EXTENSIONS:
        raise ValueError("Chỉ hỗ trợ file .tif hoặc .tiff")

    return relative


def ensure_geotiff_directories() -> None:
    for relative in ("input", "results"):
        directory = Path(settings.FILE_SERVER_GEOTIFF_DIR) / relative
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o755)


def resolve_geotiff_source(relative_path: str, must_exist: bool = True) -> Path:
    """Ánh xạ relative_path vào đúng root, chống path traversal."""
    ensure_geotiff_directories()
    relative = _safe_relative_path(relative_path)
    root = Path(settings.FILE_SERVER_GEOTIFF_DIR).resolve()
    resolved = (root / relative).resolve()

    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("GeoTIFF nằm ngoài FILE_SERVER_GEOTIFF_DIR") from error

    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"Không tìm thấy GeoTIFF: {relative.as_posix()}")

    return resolved


def build_public_geotiff_url(relative_path: str) -> str:
    relative = _safe_relative_path(relative_path)
    encoded = "/".join(quote(part) for part in relative.parts)
    return f"{settings.FILE_SERVER_PUBLIC_URL.rstrip('/')}/geotiff/{encoded}"


def create_output_path(job_id: str) -> tuple[Path, str]:
    safe_job_id = "".join(
        char for char in str(job_id) if char.isalnum() or char in {"-", "_"}
    )
    if not safe_job_id:
        raise ValueError("job_id không hợp lệ")

    relative = Path("results") / safe_job_id / "landcover_prediction.tif"
    output_path = resolve_geotiff_source(relative.as_posix(), must_exist=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(output_path.parent, 0o755)
    return output_path, relative.as_posix()


@lru_cache(maxsize=8)
def load_landcover_artifact(model_path: str) -> dict:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    with _model_lock:
        loaded = joblib.load(path)

    if isinstance(loaded, dict) and "model" in loaded:
        artifact = loaded
    elif hasattr(loaded, "predict"):
        artifact = {
            "model": loaded,
            "class_names": {},
            "band_order": DEFAULT_BAND_ORDER,
            "fallback_band_indexes": DEFAULT_BAND_INDEXES,
        }
    else:
        raise ValueError("File joblib không chứa model có hàm predict")

    if not hasattr(artifact["model"], "predict"):
        raise ValueError("Model trong artifact không có hàm predict")

    return artifact


def _normalize_band_name(value: str | None) -> str:
    return str(value or "").strip().upper().replace(" ", "").replace("-", "_")


def resolve_band_indexes(src, artifact: dict) -> tuple[list[int], list[str], str]:
    band_order = list(artifact.get("band_order") or DEFAULT_BAND_ORDER)
    wanted = [_normalize_band_name(item) for item in band_order]
    descriptions = [_normalize_band_name(item) for item in src.descriptions]

    if all(item in descriptions for item in wanted):
        indexes = [descriptions.index(item) + 1 for item in wanted]
        method = "band_descriptions"
    else:
        indexes = list(
            artifact.get("fallback_band_indexes") or DEFAULT_BAND_INDEXES
        )
        method = "fallback_band_indexes"

    if len(indexes) != len(band_order):
        raise ValueError("Số vị trí band không khớp band_order của model")
    if not indexes or min(indexes) < 1 or max(indexes) > src.count:
        raise ValueError(
            f"GeoTIFF có {src.count} band nhưng model yêu cầu vị trí {indexes}"
        )

    expected_features = getattr(artifact["model"], "n_features_in_", len(indexes))
    if int(expected_features) != len(indexes):
        raise ValueError(
            f"Model yêu cầu {expected_features} đặc trưng nhưng cấu hình có {len(indexes)}"
        )

    return indexes, band_order, method


def iter_windows(width: int, height: int, size: int = WINDOW_SIZE):
    for row_off in range(0, height, size):
        window_height = min(size, height - row_off)
        for col_off in range(0, width, size):
            window_width = min(size, width - col_off)
            yield Window(col_off, row_off, window_width, window_height)


def _pixel_area_hectares(src) -> tuple[float | None, str | None]:
    if not src.crs or not src.crs.is_projected:
        return None, "CRS không phải hệ tọa độ phẳng; không tính diện tích ha"

    linear_unit = str(getattr(src.crs, "linear_units", "") or "").lower()
    if linear_unit not in {"metre", "meter", "metres", "meters", "m"}:
        return None, f"Đơn vị CRS không phải mét: {linear_unit or 'unknown'}"

    transform = src.transform
    area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
    return area_m2 / 10000.0, None


def predict_geotiff(
    source_path: Path,
    output_path: Path,
    model_path: str,
    progress_callback=None,
) -> dict:
    artifact = load_landcover_artifact(model_path)
    model = artifact["model"]
    class_names = {
        int(key): str(value)
        for key, value in (artifact.get("class_names") or {}).items()
    }
    class_counts: Counter[int] = Counter()

    with rasterio.open(source_path) as src:
        indexes, band_order, band_method = resolve_band_indexes(src, artifact)
        windows = list(iter_windows(src.width, src.height))
        total_windows = len(windows)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate",
            predictor=2,
            BIGTIFF="IF_SAFER",
        )

        pixel_area_ha, area_warning = _pixel_area_hectares(src)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.set_band_description(1, "landcover_class_id")
            dst.update_tags(
                model_path=model_path,
                band_order=",".join(band_order),
                band_indexes=",".join(str(item) for item in indexes),
            )

            try:
                dst.write_colormap(1, CLASS_COLORS)
            except Exception:
                pass

            for index, window in enumerate(windows, start=1):
                source = src.read(indexes, window=window, masked=True).astype(np.float32)
                source_mask = np.ma.getmaskarray(source).any(axis=0)
                source_data = np.ma.filled(source, np.nan)
                features = np.moveaxis(source_data, 0, -1)

                valid = (~source_mask) & np.all(np.isfinite(features), axis=2)
                valid &= src.dataset_mask(window=window) > 0
                valid &= np.any(np.abs(features) > 1e-12, axis=2)

                predicted = np.zeros(valid.shape, dtype=np.uint8)
                if np.any(valid):
                    values = model.predict(features[valid])
                    values = np.asarray(values, dtype=np.uint8)
                    predicted[valid] = values
                    class_counts.update(int(item) for item in values)

                dst.write(predicted, 1, window=window)

                if progress_callback:
                    progress_callback(index, total_windows, sum(class_counts.values()))

    os.chmod(output_path, 0o644)
    valid_pixel_count = sum(class_counts.values())
    statistics = []

    for class_id in sorted(class_counts):
        pixels = int(class_counts[class_id])
        percent = (pixels * 100.0 / valid_pixel_count) if valid_pixel_count else 0.0
        statistics.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "pixels": pixels,
                "percent": round(percent, 4),
                "area_hectares": (
                    round(pixels * pixel_area_ha, 4)
                    if pixel_area_ha is not None
                    else None
                ),
            }
        )

    return {
        "width": src.width,
        "height": src.height,
        "crs": str(src.crs) if src.crs else None,
        "source_band_count": src.count,
        "band_order": band_order,
        "band_indexes": indexes,
        "band_resolution_method": band_method,
        "valid_pixel_count": valid_pixel_count,
        "statistics": statistics,
        "area_warning": area_warning,
        "output_size_bytes": output_path.stat().st_size,
        "window_count": total_windows,
    }
