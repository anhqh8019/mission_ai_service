import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings


class ApiTest(unittest.TestCase):
    def test_all_sources_and_enqueue_failure(self):
        mission = {"mission_uuid": "uuid", "model_name": "rf", "model_version": "v1", "model_path": "model.joblib"}
        with tempfile.TemporaryDirectory() as folder, patch.object(settings, "FILE_SERVER_GEOTIFF_DIR", folder), patch("app.main.resolve_mission", return_value=mission), patch("app.main.create_job") as create, patch("app.main.process_landcover_geotiff.apply_async") as enqueue:
            client = TestClient(app)
            source = Path(folder)/"input/source.tif"
            source.parent.mkdir()
            with rasterio.open(source, "w", driver="GTiff", width=2, height=2, count=7, dtype="uint16", crs="EPSG:32648", transform=from_origin(500000, 1500000, 10, 10)) as dataset:
                dataset.write(np.ones((7, 2, 2), dtype=np.uint16))
            for data, kind in (({"relative_path": "input/source.tif"}, "PATH"), ({"input_url": "https://example.com/a.tif"}, "URL")):
                response = client.post("/api/inference/landcover/geotiff", json={"mission_id": "landcover", **data})
                self.assertEqual(response.status_code, 202, response.text)
                self.assertEqual(enqueue.call_args.kwargs["kwargs"]["input_type"], "LANDCOVER_GEOTIFF_"+kind)
            response = client.post("/api/inference/landcover/geotiff/upload", data={"mission_id": "landcover"}, files={"file": ("source.tif", source.read_bytes(), "image/tiff")})
            self.assertEqual(response.status_code, 202, response.text)
            self.assertTrue((Path(folder)/response.json()["source"]["relative_path"]).is_file())
            self.assertEqual(enqueue.call_args.kwargs["kwargs"]["input_type"], "LANDCOVER_GEOTIFF_UPLOAD")
            for name, content, status in (("bad.tif", b"invalid", 400), ("bad.png", b"invalid", 400)):
                response = client.post("/api/inference/landcover/geotiff/upload", data={"mission_id": "landcover"}, files={"file": (name, content)})
                self.assertEqual(response.status_code, status)
            self.assertEqual(create.call_count, 3)
            enqueue.side_effect = RuntimeError("queue unavailable")
            before = set((Path(folder)/"input/uploads").glob("*.tif"))
            with patch("app.database.mark_failed") as failed:
                response = client.post("/api/inference/landcover/geotiff/upload", data={"mission_id": "landcover"}, files={"file": ("source.tif", source.read_bytes())})
                self.assertEqual(response.status_code, 503)
                failed.assert_called_once()
            self.assertEqual(before, set((Path(folder)/"input/uploads").glob("*.tif")))
