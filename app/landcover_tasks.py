from __future__ import annotations

import traceback

from .celery_app import celery_app
from .config import settings
from .database import mark_completed, mark_failed, mark_running, update_progress
from .landcover_inference import (
    build_public_geotiff_url,
    create_output_path,
    predict_geotiff,
    resolve_geotiff_source,
)
from .mqtt_client import publish
from .storage import upload_json


@celery_app.task(
    bind=True,
    name="process_landcover_geotiff",
    queue="landcover",
)
def process_landcover_geotiff(
    self,
    job_id: str,
    mission_id: str,
    model_name: str,
    model_version: str,
    model_path: str,
    relative_path: str,
):
    try:
        source_path = resolve_geotiff_source(relative_path, must_exist=True)
        output_path, output_relative_path = create_output_path(job_id)

        mark_running(job_id)
        publish(
            f"missions/{mission_id}/inference/started",
            {
                "event": "inference.started",
                "job_id": job_id,
                "mission_id": mission_id,
                "status": "running",
                "input_type": "LANDCOVER_GEOTIFF_PATH",
                "model_name": model_name,
                "model_version": model_version,
            },
        )

        def update_window_progress(processed: int, total: int, valid_pixels: int):
            progress = int(processed * 99 / total) if total else 0
            update_progress(
                job_id,
                processed,
                total,
                valid_pixels,
                progress,
            )
            self.update_state(
                state="PROGRESS",
                meta={
                    "processed_windows": processed,
                    "total_windows": total,
                    "progress": progress,
                },
            )

        summary = predict_geotiff(
            source_path=source_path,
            output_path=output_path,
            model_path=model_path,
            progress_callback=update_window_progress,
        )

        source_url = build_public_geotiff_url(relative_path)
        output_url = build_public_geotiff_url(output_relative_path)

        result_data = {
            "job_id": job_id,
            "status": "completed",
            "input_type": "LANDCOVER_GEOTIFF_PATH",
            "mission_id": mission_id,
            "model": {
                "name": model_name,
                "version": model_version,
                "path": model_path,
            },
            "input": {
                "relative_path": relative_path,
                "url": source_url,
                "width": summary["width"],
                "height": summary["height"],
                "crs": summary["crs"],
                "source_band_count": summary["source_band_count"],
                "band_order": summary["band_order"],
                "band_indexes": summary["band_indexes"],
                "band_resolution_method": summary["band_resolution_method"],
            },
            "output": {
                "relative_path": output_relative_path,
                "url": output_url,
                "size_bytes": summary["output_size_bytes"],
            },
            "summary": {
                "valid_pixel_count": summary["valid_pixel_count"],
                "window_count": summary["window_count"],
                "statistics": summary["statistics"],
                "area_warning": summary["area_warning"],
            },
        }

        result_url = upload_json(job_id, result_data)
        mark_completed(
            job_id,
            result_url,
            summary["window_count"],
            summary["window_count"],
            summary["valid_pixel_count"],
        )

        public_result_url = result_url.replace(
            settings.MINIO_ENDPOINT,
            settings.MINIO_PUBLIC_ENDPOINT,
        )
        publish(
            f"missions/{mission_id}/inference/completed",
            {
                "event": "inference.completed",
                "job_id": job_id,
                "mission_id": mission_id,
                "status": "completed",
                "input_type": "LANDCOVER_GEOTIFF_PATH",
                "model_name": model_name,
                "model_version": model_version,
                "result_url": public_result_url,
                "geotiff_url": output_url,
                "statistics": summary["statistics"],
            },
        )
        return result_data

    except Exception as error:
        traceback.print_exc()
        try:
            mark_failed(job_id, error)
        except Exception:
            traceback.print_exc()

        try:
            publish(
                f"missions/{mission_id}/inference/failed",
                {
                    "event": "inference.failed",
                    "job_id": job_id,
                    "mission_id": mission_id,
                    "status": "failed",
                    "input_type": "LANDCOVER_GEOTIFF_PATH",
                    "error": str(error),
                },
            )
        except Exception:
            traceback.print_exc()

        raise
