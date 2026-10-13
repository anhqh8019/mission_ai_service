import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import rasterio
from rasterio.transform import from_origin
from pydantic import ValidationError

from app.landcover_inputs import InputTooLarge, save_chunks, download_geotiff, validate_public_url
from app.schemas import LandcoverGeotiffRequest


class InputsTest(unittest.TestCase):
    def test_schema_exactly_one_source(self):
        for data in ({}, {"relative_path": "input/a.tif", "input_url": "https://example.com/a.tif"}):
            with self.assertRaises(ValidationError):
                LandcoverGeotiffRequest(mission_id="landcover", **data)
        self.assertEqual(LandcoverGeotiffRequest(mission_id="landcover", relative_path="input/a.tif").relative_path, "input/a.tif")
        self.assertIsNotNone(LandcoverGeotiffRequest(mission_id="landcover", input_url="https://example.com/a.tif").input_url)

    def test_upload_valid_invalid_and_oversize(self):
        with tempfile.TemporaryDirectory() as folder:
            src, dst = Path(folder)/"source.tif", Path(folder)/"saved.tif"
            with rasterio.open(src, "w", driver="GTiff", width=2, height=2, count=7,
                               dtype="uint16", crs="EPSG:32648", transform=from_origin(500000, 1500000, 10, 10)) as dataset:
                dataset.write(np.ones((7, 2, 2), dtype=np.uint16))
            content = src.read_bytes()
            save_chunks([content[:100], content[100:]], dst)
            self.assertEqual(dst.read_bytes(), content)
            dst.unlink()
            for chunks, limit, error in (([b"not tiff"], 100, ValueError), ([content], 10, InputTooLarge)):
                with patch("app.landcover_inputs.GEOTIFF_MAX_BYTES", limit), self.assertRaises(error):
                    save_chunks(chunks, dst)
                self.assertFalse(dst.exists())
                self.assertFalse(dst.with_suffix(".part").exists())

    def test_public_url_blocks_internal_addresses(self):
        for ip in ("127.0.0.1", "10.1.53.88", "169.254.169.254", "::1"):
            with patch("app.landcover_inputs.socket.getaddrinfo", return_value=[(0, 0, 0, "", (ip, 80))]):
                with self.assertRaises(ValueError):
                    validate_public_url("https://example.com/a.tif")

    def test_redirect_to_internal_address_rejected(self):
        session = MagicMock()
        response = session.get.return_value.__enter__.return_value
        response.status_code = 302
        response.headers = {"Location": "http://127.0.0.1/a.tif"}
        def addresses(host, *args, **kwargs):
            return [(0, 0, 0, "", ("8.8.8.8" if host == "example.com" else "127.0.0.1", 80))]
        with patch("app.landcover_inputs.requests.Session") as factory, patch("app.landcover_inputs.socket.getaddrinfo", side_effect=addresses):
            factory.return_value.__enter__.return_value = session
            with self.assertRaises(ValueError):
                download_geotiff("https://example.com/a.tif", Path("unused.tif"))
        self.assertEqual(session.get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
