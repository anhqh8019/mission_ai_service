import os
import re
import shutil
import uuid

from pathlib import Path
from urllib.parse import quote

from .config import settings


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_IMAGE_EXTENSION = ".jpg"
DEFAULT_VIDEO_EXTENSION = ".mp4"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}


# ============================================================
# SAFE NAME
# ============================================================


def safe_filename_part(
    value: str,
) -> str:
    """
    Convert a string to a safe filesystem filename component.
    """

    value = str(
        value or ""
    ).strip()

    value = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "_",
        value,
    )

    value = value.strip(
        "._"
    )

    return (
        value
        or "unknown"
    )


# ============================================================
# DIRECTORY HELPERS
# ============================================================


def ensure_directory(
    directory: str,
) -> str:
    """
    Create directory and force mode 755.

    Important:
    os.makedirs(mode=...) is still affected by umask,
    therefore chmod is called afterwards.
    """

    Path(
        directory
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    os.chmod(
        directory,
        0o755,
    )

    return directory


def ensure_file_server_dirs() -> None:

    ensure_directory(
        settings.FILE_SERVER_IMAGE_DIR
    )

    ensure_directory(
        settings.FILE_SERVER_VIDEO_DIR
    )


def ensure_job_image_dir(
    job_id: str,
) -> str:

    safe_job_id = (
        safe_filename_part(
            job_id
        )
    )

    directory = os.path.join(
        settings.FILE_SERVER_IMAGE_DIR,
        safe_job_id,
    )

    return ensure_directory(
        directory
    )


def ensure_job_video_dir(
    job_id: str,
) -> str:

    safe_job_id = (
        safe_filename_part(
            job_id
        )
    )

    directory = os.path.join(
        settings.FILE_SERVER_VIDEO_DIR,
        safe_job_id,
    )

    return ensure_directory(
        directory
    )


# ============================================================
# PUBLIC URL HELPERS
# ============================================================


def get_public_base_url() -> str:

    return (
        settings.FILE_SERVER_PUBLIC_URL
        .rstrip("/")
    )


def build_public_image_url(
    job_id: str,
    filename: str,
) -> str:

    base_url = (
        get_public_base_url()
    )

    safe_job_id = (
        safe_filename_part(
            job_id
        )
    )

    return (
        f"{base_url}/images/"
        f"{quote(safe_job_id)}/"
        f"{quote(filename)}"
    )


def build_public_video_url(
    job_id: str,
    filename: str,
) -> str:

    base_url = (
        get_public_base_url()
    )

    safe_job_id = (
        safe_filename_part(
            job_id
        )
    )

    return (
        f"{base_url}/videos/"
        f"{quote(safe_job_id)}/"
        f"{quote(filename)}"
    )


# ============================================================
# EXTENSION HELPERS
# ============================================================


def normalize_image_extension(
    extension: str,
) -> str:

    extension = str(
        extension or ""
    ).strip().lower()

    if not extension.startswith("."):
        extension = (
            f".{extension}"
            if extension
            else ""
        )

    if extension == ".jpeg":
        return ".jpg"

    if extension not in IMAGE_EXTENSIONS:
        return DEFAULT_IMAGE_EXTENSION

    return extension


def normalize_video_extension(
    extension: str,
) -> str:

    extension = str(
        extension or ""
    ).strip().lower()

    if not extension.startswith("."):
        extension = (
            f".{extension}"
            if extension
            else ""
        )

    if extension not in VIDEO_EXTENSIONS:
        return DEFAULT_VIDEO_EXTENSION

    return extension


# ============================================================
# IMAGE - BYTES
# ============================================================


def save_annotated_image_bytes(
    job_id: str,
    image_index: int,
    model_name: str,
    image_bytes: bytes,
    extension: str = ".jpg",
) -> dict:

    if not image_bytes:
        raise ValueError(
            "Annotated image bytes are empty"
        )

    ensure_file_server_dirs()

    job_dir = (
        ensure_job_image_dir(
            job_id
        )
    )

    extension = (
        normalize_image_extension(
            extension
        )
    )

    safe_model = (
        safe_filename_part(
            model_name
        )
    )

    filename = (
        f"image_"
        f"{int(image_index):04d}_"
        f"{safe_model}_"
        f"{uuid.uuid4().hex[:8]}"
        f"{extension}"
    )

    output_path = os.path.join(
        job_dir,
        filename,
    )

    with open(
        output_path,
        "wb",
    ) as output_file:

        output_file.write(
            image_bytes
        )

        output_file.flush()

        os.fsync(
            output_file.fileno()
        )

    # IMPORTANT:
    # Nginx must be able to read the file.
    os.chmod(
        output_path,
        0o644,
    )

    # make sure job dir is traversable
    os.chmod(
        job_dir,
        0o755,
    )

    file_size = (
        os.path.getsize(
            output_path
        )
    )

    if file_size <= 0:
        raise RuntimeError(
            f"Annotated image is empty: "
            f"{output_path}"
        )

    file_mode = (
        os.stat(
            output_path
        ).st_mode
        & 0o777
    )

    print(
        f"[FILE SERVER] "
        f"image saved "
        f"path={output_path} "
        f"size={file_size} "
        f"mode={oct(file_mode)}",
        flush=True,
    )

    return {
        "filename":
            filename,

        "path":
            output_path,

        "size":
            file_size,

        "mode":
            oct(
                file_mode
            ),

        "url":
            build_public_image_url(
                job_id,
                filename,
            ),
    }


# ============================================================
# IMAGE - LOCAL FILE
# ============================================================


def save_annotated_image_file(
    job_id: str,
    image_index: int,
    model_name: str,
    source_path: str,
) -> dict:

    if not os.path.isfile(
        source_path
    ):
        raise FileNotFoundError(
            f"Annotated image not found: "
            f"{source_path}"
        )

    ensure_file_server_dirs()

    job_dir = (
        ensure_job_image_dir(
            job_id
        )
    )

    extension = (
        os.path.splitext(
            source_path
        )[1]
    )

    extension = (
        normalize_image_extension(
            extension
        )
    )

    safe_model = (
        safe_filename_part(
            model_name
        )
    )

    filename = (
        f"image_"
        f"{int(image_index):04d}_"
        f"{safe_model}_"
        f"{uuid.uuid4().hex[:8]}"
        f"{extension}"
    )

    output_path = os.path.join(
        job_dir,
        filename,
    )

    # copy2 may preserve source permission,
    # so chmod MUST happen afterwards.
    shutil.copy2(
        source_path,
        output_path,
    )

    os.chmod(
        output_path,
        0o644,
    )

    os.chmod(
        job_dir,
        0o755,
    )

    file_size = (
        os.path.getsize(
            output_path
        )
    )

    if file_size <= 0:
        raise RuntimeError(
            f"Annotated image is empty: "
            f"{output_path}"
        )

    file_mode = (
        os.stat(
            output_path
        ).st_mode
        & 0o777
    )

    print(
        f"[FILE SERVER] "
        f"image copied "
        f"path={output_path} "
        f"size={file_size} "
        f"mode={oct(file_mode)}",
        flush=True,
    )

    return {
        "filename":
            filename,

        "path":
            output_path,

        "size":
            file_size,

        "mode":
            oct(
                file_mode
            ),

        "url":
            build_public_image_url(
                job_id,
                filename,
            ),
    }


# ============================================================
# VIDEO - LOCAL FILE
# ============================================================


def save_annotated_video_file(
    job_id: str,
    video_index: int,
    model_name: str,
    source_path: str,
) -> dict:

    if not os.path.isfile(
        source_path
    ):
        raise FileNotFoundError(
            f"Annotated video not found: "
            f"{source_path}"
        )

    source_size = (
        os.path.getsize(
            source_path
        )
    )

    if source_size <= 0:
        raise RuntimeError(
            f"Annotated video is empty: "
            f"{source_path}"
        )

    ensure_file_server_dirs()

    job_dir = (
        ensure_job_video_dir(
            job_id
        )
    )

    source_extension = (
        os.path.splitext(
            source_path
        )[1]
    )

    extension = (
        normalize_video_extension(
            source_extension
        )
    )

    safe_model = (
        safe_filename_part(
            model_name
        )
    )

    filename = (
        f"video_"
        f"{int(video_index):04d}_"
        f"{safe_model}_"
        f"{uuid.uuid4().hex[:8]}"
        f"{extension}"
    )

    output_path = os.path.join(
        job_dir,
        filename,
    )

    # IMPORTANT:
    #
    # tempfile.mkstemp() normally creates source with mode 600.
    # shutil.copy2() preserves metadata including mode.
    #
    # Therefore chmod MUST happen AFTER copy2().
    shutil.copy2(
        source_path,
        output_path,
    )

    # Nginx reader:
    # rw-r--r--
    os.chmod(
        output_path,
        0o644,
    )

    # Nginx must traverse job directory
    os.chmod(
        job_dir,
        0o755,
    )

    output_size = (
        os.path.getsize(
            output_path
        )
    )

    if output_size <= 0:
        raise RuntimeError(
            f"Copied annotated video is empty: "
            f"{output_path}"
        )

    file_mode = (
        os.stat(
            output_path
        ).st_mode
        & 0o777
    )

    directory_mode = (
        os.stat(
            job_dir
        ).st_mode
        & 0o777
    )

    print(
        f"[FILE SERVER] "
        f"video saved "
        f"path={output_path} "
        f"size={output_size} "
        f"file_mode={oct(file_mode)} "
        f"dir_mode={oct(directory_mode)}",
        flush=True,
    )

    return {
        "filename":
            filename,

        "path":
            output_path,

        "size":
            output_size,

        "mode":
            oct(
                file_mode
            ),

        "url":
            build_public_video_url(
                job_id,
                filename,
            ),
    }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================


def save_annotated_image(
    job_id: str,
    local_path: str,
    image_index: int = 0,
    model_name: str = "model",
) -> str:
    """
    Compatibility wrapper for older tasks.py code.
    Returns URL only.
    """

    result = (
        save_annotated_image_file(
            job_id=
                job_id,

            image_index=
                image_index,

            model_name=
                model_name,

            source_path=
                local_path,
        )
    )

    return result[
        "url"
    ]


def save_annotated_video(
    job_id: str,
    local_path: str,
    video_index: int = 0,
    model_name: str = "model",
) -> str:
    """
    Compatibility wrapper for older tasks.py code.
    Returns URL only.
    """

    result = (
        save_annotated_video_file(
            job_id=
                job_id,

            video_index=
                video_index,

            model_name=
                model_name,

            source_path=
                local_path,
        )
    )

    return result[
        "url"
    ]


# ============================================================
# DELETE JOB FILES
# ============================================================


def delete_job_files(
    job_id: str,
) -> None:

    safe_job_id = (
        safe_filename_part(
            job_id
        )
    )

    image_dir = os.path.join(
        settings.FILE_SERVER_IMAGE_DIR,
        safe_job_id,
    )

    video_dir = os.path.join(
        settings.FILE_SERVER_VIDEO_DIR,
        safe_job_id,
    )

    for directory in (
        image_dir,
        video_dir,
    ):

        if not os.path.isdir(
            directory
        ):
            continue

        try:

            shutil.rmtree(
                directory
            )

            print(
                f"[FILE SERVER] "
                f"deleted directory: "
                f"{directory}",
                flush=True,
            )

        except Exception as e:

            print(
                f"[FILE SERVER] "
                f"cannot delete "
                f"{directory}: {e}",
                flush=True,
            )


# ============================================================
# HEALTH / DEBUG INFO
# ============================================================


def get_file_server_info() -> dict:

    ensure_file_server_dirs()

    image_dir = (
        settings.FILE_SERVER_IMAGE_DIR
    )

    video_dir = (
        settings.FILE_SERVER_VIDEO_DIR
    )

    return {
        "public_url":
            settings.FILE_SERVER_PUBLIC_URL,

        "image_dir":
            image_dir,

        "video_dir":
            video_dir,

        "image_dir_exists":
            os.path.isdir(
                image_dir
            ),

        "video_dir_exists":
            os.path.isdir(
                video_dir
            ),

        "image_dir_mode":
            oct(
                os.stat(
                    image_dir
                ).st_mode
                & 0o777
            ),

        "video_dir_mode":
            oct(
                os.stat(
                    video_dir
                ).st_mode
                & 0o777
            ),
    }