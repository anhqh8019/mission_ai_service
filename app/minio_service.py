import io
import json
import os

from minio import Minio


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "inference-results")


# Minio() không nhận http://
endpoint = MINIO_ENDPOINT.replace("http://", "").replace("https://", "")

minio_client = Minio(
    endpoint,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_ENDPOINT.startswith("https://"),
)


def get_json_from_minio(object_name: str) -> dict:
    response = None

    try:
        response = minio_client.get_object(
            MINIO_BUCKET,
            object_name
        )

        data = response.read()

        return json.loads(data.decode("utf-8"))

    finally:
        if response is not None:
            response.close()
            response.release_conn()