"""Bounded GeoTIFF ingestion into the shared file-server directory."""
import os
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import rasterio
import requests

GEOTIFF_MAX_BYTES = int(os.getenv("GEOTIFF_MAX_BYTES", str(2 * 1024**3)))
GEOTIFF_DOWNLOAD_TIMEOUT = int(os.getenv("GEOTIFF_DOWNLOAD_TIMEOUT", "120"))


class InputTooLarge(ValueError):
    pass


def validate_public_url(url):
    parsed = urlsplit(str(url))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("URL phải là HTTP/HTTPS public, không chứa thông tin đăng nhập")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except (OSError, ValueError) as error:
        raise ValueError("Không phân giải được địa chỉ URL") from error
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("URL phải trỏ tới địa chỉ public; không hỗ trợ địa chỉ nội bộ")


def validate_geotiff(path):
    try:
        with rasterio.open(path) as src:
            if src.driver != "GTiff" or src.crs is None:
                raise ValueError("File phải là GeoTIFF có hệ tọa độ CRS")
    except rasterio.errors.RasterioError as error:
        raise ValueError("File không phải GeoTIFF hợp lệ") from error


def save_chunks(chunks, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    try:
        size = 0
        with temporary.open("wb") as output:
            for chunk in chunks:
                size += len(chunk)
                if size > GEOTIFF_MAX_BYTES:
                    raise InputTooLarge("GeoTIFF vượt giới hạn dung lượng")
                output.write(chunk)
        validate_geotiff(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def download_geotiff(url, destination):
    with requests.Session() as session:
        session.trust_env = False
        for _ in range(6):
            validate_public_url(url)
            with session.get(url, stream=True, allow_redirects=False,
                             timeout=(10, GEOTIFF_DOWNLOAD_TIMEOUT)) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location:
                        raise ValueError("Redirect thiếu Location")
                    url = urljoin(url, location)
                    continue
                response.raise_for_status()
                if int(response.headers.get("Content-Length", "0")) > GEOTIFF_MAX_BYTES:
                    raise InputTooLarge("GeoTIFF vượt giới hạn dung lượng")
                save_chunks(response.iter_content(chunk_size=1024 * 1024), destination)
                return
        raise ValueError("URL chuyển hướng quá nhiều lần")
