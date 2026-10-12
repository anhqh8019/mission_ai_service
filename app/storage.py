import json
import os
import tempfile

import boto3

from botocore.config import Config
from botocore.exceptions import ClientError

from .config import settings


# ============================================================
# S3 / MINIO CLIENT
# ============================================================

s3 = boto3.client(
    "s3",

    endpoint_url=settings.MINIO_ENDPOINT,

    aws_access_key_id=
        settings.MINIO_ACCESS_KEY,

    aws_secret_access_key=
        settings.MINIO_SECRET_KEY,

    region_name="us-east-1",

    config=Config(
        signature_version="s3v4",

        s3={
            "addressing_style": "path"
        }
    )
)


# ============================================================
# ENSURE BUCKET
# ============================================================

def ensure_bucket():

    bucket = settings.MINIO_BUCKET

    try:
        s3.head_bucket(
            Bucket=bucket
        )

    except ClientError:
        s3.create_bucket(
            Bucket=bucket
        )


# ============================================================
# OBJECT KEYS
# ============================================================

def get_result_key(
    job_id: str
):
    return (
        f"inference/"
        f"{job_id}/"
        f"result.json"
    )


def get_annotated_video_key(
    job_id: str
):
    return (
        f"inference/"
        f"{job_id}/"
        f"annotated.mp4"
    )


# ============================================================
# INTERNAL URL
#
# URL này dùng trong Docker network.
# Client public không nên gọi trực tiếp URL này.
# ============================================================

def make_internal_url(
    key: str
):
    endpoint = (
        settings.MINIO_ENDPOINT
        .rstrip("/")
    )

    return (
        f"{endpoint}/"
        f"{settings.MINIO_BUCKET}/"
        f"{key}"
    )


# ============================================================
# UPLOAD RESULT JSON
# ============================================================

def upload_json(
    job_id: str,
    data: dict
):

    ensure_bucket()

    key = get_result_key(
        job_id
    )

    body = json.dumps(
        data,
        ensure_ascii=False
    ).encode("utf-8")

    s3.put_object(
        Bucket=
            settings.MINIO_BUCKET,

        Key=
            key,

        Body=
            body,

        ContentType=
            "application/json"
    )

    return make_internal_url(
        key
    )


# ============================================================
# UPLOAD ANNOTATED VIDEO
# ============================================================

def upload_annotated_video(
    job_id: str,
    local_path: str
):

    ensure_bucket()

    if not os.path.isfile(
        local_path
    ):
        raise FileNotFoundError(
            f"Annotated video not found: "
            f"{local_path}"
        )

    key = get_annotated_video_key(
        job_id
    )

    s3.upload_file(
        local_path,

        settings.MINIO_BUCKET,

        key,

        ExtraArgs={
            "ContentType":
                "video/mp4"
        }
    )

    return make_internal_url(
        key
    )


# ============================================================
# GET RESULT.JSON
# ============================================================

def get_result_json(
    job_id: str
):

    key = get_result_key(
        job_id
    )

    response = s3.get_object(
        Bucket=
            settings.MINIO_BUCKET,

        Key=
            key
    )

    raw = response[
        "Body"
    ].read()

    return json.loads(
        raw.decode("utf-8")
    )

def upload_annotated_image(
    job_id: str,
    model_name: str,
    filename: str,
    image_bytes: bytes,
):
    ensure_bucket()

    safe_filename = os.path.basename(
        filename
    )

    extension = (
        os.path.splitext(
            safe_filename
        )[1]
        .lower()
    )

    # Nếu file gốc không phải format web phổ biến,
    # annotated output sẽ đổi thành jpg
    if extension not in (
        ".jpg",
        ".jpeg",
        ".png"
    ):
        extension = ".jpg"

        safe_filename = (
            os.path.splitext(
                safe_filename
            )[0]
            + ".jpg"
        )

    if extension in (
        ".jpg",
        ".jpeg"
    ):
        content_type = (
            "image/jpeg"
        )

    else:
        content_type = (
            "image/png"
        )

    key = (
        f"inference/"
        f"{job_id}/"
        f"{model_name}/"
        f"annotated/"
        f"{safe_filename}"
    )

    s3.put_object(
        Bucket=
            settings.MINIO_BUCKET,

        Key=
            key,

        Body=
            image_bytes,

        ContentType=
            content_type,
    )

    endpoint = (
        settings.MINIO_ENDPOINT
        .rstrip("/")
    )

    return (
        f"{endpoint}/"
        f"{settings.MINIO_BUCKET}/"
        f"{key}"
    )


def upload_multi_annotated_video(
    job_id: str,
    model_name: str,
    video_index: int,
    local_path: str,
):
    ensure_bucket()

    if not os.path.isfile(local_path):
        raise FileNotFoundError(
            f"Annotated video not found: {local_path}"
        )

    key = (
        f"inference/"
        f"{job_id}/"
        f"{model_name}/"
        f"videos/"
        f"{video_index:04d}_annotated.mp4"
    )

    s3.upload_file(
        local_path,
        settings.MINIO_BUCKET,
        key,
        ExtraArgs={
            "ContentType": "video/mp4"
        }
    )

    endpoint = (
        settings.MINIO_ENDPOINT
        .rstrip("/")
    )

    return (
        f"{endpoint}/"
        f"{settings.MINIO_BUCKET}/"
        f"{key}"
    )


def download_minio_object(
    bucket: str,
    object_key: str,
) -> str:

    suffix = os.path.splitext(
        object_key
    )[1]

    if not suffix:
        suffix = ".mp4"

    fd, temp_path = (
        tempfile.mkstemp(
            suffix=suffix
        )
    )

    os.close(fd)

    try:

        # kiểm tra object tồn tại trước
        s3.head_object(
            Bucket=bucket,
            Key=object_key,
        )

        s3.download_file(
            bucket,
            object_key,
            temp_path,
        )

        if not os.path.isfile(
            temp_path
        ):
            raise RuntimeError(
                "MinIO download did not create file"
            )

        file_size = os.path.getsize(
            temp_path
        )

        if file_size <= 0:
            raise RuntimeError(
                f"Downloaded MinIO object is empty: "
                f"{bucket}/{object_key}"
            )

        print(
            "[MINIO DOWNLOAD] "
            f"{bucket}/{object_key} "
            f"=> {temp_path} "
            f"size={file_size}",
            flush=True,
        )

        return temp_path

    except Exception:

        if os.path.isfile(
            temp_path
        ):
            try:
                os.remove(
                    temp_path
                )
            except Exception:
                pass

        raise