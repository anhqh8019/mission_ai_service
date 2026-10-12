import os
import cv2
import tempfile
import requests
import traceback
import uuid
import numpy as np
import time

from .celery_app import celery_app
from .model_manager import model_manager

from .database import (
    mark_running,
    update_progress,
    mark_completed,
    mark_failed,
)

from .mqtt_client import publish

from .storage import (
    upload_json,
    download_minio_object,
)

from .file_server import (
    save_annotated_image_bytes,
    save_annotated_video_file,
)

from .config import settings

# ============================================================
# Helper
# ============================================================
IMAGE_DOWNLOAD_TIMEOUT = 30

MAX_IMAGE_DOWNLOAD_SIZE = (
    20 * 1024 * 1024
)  # 20 MB / image


# ============================================================
# CONFIG
# ============================================================

DOWNLOAD_TIMEOUT = 120
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
IMAGE_BATCH_SIZE = 8

# ============================================================
# DOWNLOAD VIDEO
# ============================================================

def download_video(video_url: str) -> str:
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    )

    temp_path = temp_file.name

    try:
        with requests.get(
            video_url,
            stream=True,
            timeout=DOWNLOAD_TIMEOUT,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()

            for chunk in response.iter_content(
                chunk_size=DOWNLOAD_CHUNK_SIZE
            ):
                if chunk:
                    temp_file.write(chunk)

        temp_file.flush()
        temp_file.close()

        return temp_path

    except Exception:
        temp_file.close()

        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise


# ============================================================
# RESOLVE VIDEO SOURCE
# ============================================================

def resolve_video_source(video_source: str):
    # --------------------------------------------------------
    # Uploaded video file
    # --------------------------------------------------------

    if video_source.startswith(
        "/shared/mission_uploads/"
    ):
        if not os.path.isfile(video_source):
            raise FileNotFoundError(
                f"Uploaded video not found: {video_source}"
            )

        return video_source, True

    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    if (
        video_source.startswith("http://")
        or video_source.startswith("https://")
    ):
        video_path = download_video(
            video_source
        )

        return video_path, True

    raise ValueError(
        f"Unsupported video source: {video_source}"
    )


# ============================================================
# PARSE YOLO RESULT
# ============================================================

def parse_yolo_result(
    result,
    model,
):
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        bbox = (
            box.xyxy[0]
            .detach()
            .cpu()
            .tolist()
        )

        try:
            class_name = model.names[
                class_id
            ]

        except Exception:
            class_name = str(
                class_id
            )

        detections.append(
            {
                "class_id": class_id,

                "class_name": str(
                    class_name
                ),

                "confidence": round(
                    confidence,
                    6,
                ),

                "bbox": [
                    round(
                        float(v),
                        2,
                    )
                    for v in bbox
                ],
            }
        )

    return detections


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    frame,
    detections,
):
    annotated = frame.copy()

    for detection in detections:
        bbox = detection.get(
            "bbox",
            [],
        )

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [
            int(v)
            for v in bbox
        ]

        class_name = detection.get(
            "class_name",
            "object",
        )

        confidence = float(
            detection.get(
                "confidence",
                0,
            )
        )

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        # bbox
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # text size
        (text_width, text_height), _ = (
            cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )
        )

        text_y = max(
            y1,
            text_height + 10,
        )

        # label background
        cv2.rectangle(
            annotated,
            (
                x1,
                text_y
                - text_height
                - 8,
            ),
            (
                x1
                + text_width
                + 8,
                text_y + 4,
            ),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            annotated,
            label,
            (
                x1 + 4,
                text_y - 3,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


# ============================================================
# CREATE VIDEO WRITER
# ============================================================

def create_video_writer(
    fps: float,
    width: int,
    height: int,
):
    fd, output_path = (
        tempfile.mkstemp(
            suffix=".mp4"
        )
    )

    os.close(fd)

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        if os.path.exists(
            output_path
        ):
            os.remove(
                output_path
            )

        raise RuntimeError(
            "Cannot create annotated video"
        )

    return writer, output_path


# ============================================================
# CELERY TASK
# ============================================================

@celery_app.task(
    bind=True,
    name="process_video",
)
def process_video(
    self,

    job_id: str,
    mission_id: str,

    model_name: str,
    model_version: str,
    model_path: str,

    video_source: str,

    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,

    frame_skip: int = 5,
    batch_size: int = 8,

    generate_annotated_video: bool = False,
):
    video_path = None
    delete_after_process = False

    cap = None
    writer = None

    annotated_path = None

    annotated_video_url = None

    try:
        # ====================================================
        # PARAMS
        # ====================================================

        frame_skip = max(
            1,
            int(frame_skip),
        )

        batch_size = max(
            1,
            int(batch_size),
        )

        confidence_threshold = float(
            confidence_threshold
        )

        iou_threshold = float(
            iou_threshold
        )

        generate_annotated_video = bool(
            generate_annotated_video
        )

        print(
            f"[JOB {job_id}] "
            f"generate_annotated_video="
            f"{generate_annotated_video}",
            flush=True,
        )

        # ====================================================
        # RUNNING
        # ====================================================

        mark_running(
            job_id
        )

        # ====================================================
        # MQTT STARTED
        # ====================================================

        publish(
            f"missions/"
            f"{mission_id}/"
            f"inference/started",

            {
                "event":
                    "inference.started",

                "job_id":
                    job_id,

                "mission_id":
                    mission_id,

                "status":
                    "running",

                "model_name":
                    model_name,

                "model_version":
                    model_version,

                "generate_annotated_video":
                    generate_annotated_video,
            }
        )

        print(
            f"[JOB {job_id}] "
            f"Starting video inference",
            flush=True,
        )

        print(
            f"[JOB {job_id}] "
            f"model="
            f"{model_name}:"
            f"{model_version}",
            flush=True,
        )

        print(
            f"[JOB {job_id}] "
            f"source="
            f"{video_source}",
            flush=True,
        )

        # ====================================================
        # VIDEO SOURCE
        # ====================================================

        (
            video_path,
            delete_after_process,
        ) = resolve_video_source(
            video_source
        )

        print(
            f"[JOB {job_id}] "
            f"video_path="
            f"{video_path}",
            flush=True,
        )

        # ====================================================
        # LOAD MODEL
        # ====================================================

        model = (
            model_manager
            .get_model(
                model_name,
                model_version,
                model_path,
            )
        )

        print(
            f"[JOB {job_id}] "
            f"model loaded",
            flush=True,
        )

        # ====================================================
        # OPEN VIDEO
        # ====================================================

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: "
                f"{video_path}"
            )

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        if fps <= 0:
            fps = 30.0

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        duration_seconds = (
            total_frames / fps
            if fps > 0
            else 0
        )

        print(
            f"[JOB {job_id}] "
            f"video="
            f"{width}x{height}, "
            f"fps={fps:.3f}, "
            f"frames={total_frames}",
            flush=True,
        )

        # ====================================================
        # RESULT VARIABLES
        # ====================================================

        frames_result = []

        processed_frames = 0

        detection_count = 0

        # ====================================================
        # MODE A:
        # GENERATE ANNOTATED VIDEO
        #
        # PASS 1 = batch inference
        # PASS 2 = draw/write all frames
        # ====================================================

        if generate_annotated_video:

            # ------------------------------------------------
            # VIDEO WRITER
            # ------------------------------------------------

            (
                writer,
                annotated_path,
            ) = create_video_writer(
                fps,
                width,
                height,
            )

            print(
                f"[JOB {job_id}] "
                f"annotated output="
                f"{annotated_path}",
                flush=True,
            )

            # =================================================
            # PASS 1 - BATCH INFERENCE
            # =================================================

            detections_by_frame = {}

            batch_images = []
            batch_metadata = []

            current_frame = 0

            def process_annotation_batch():
                nonlocal processed_frames
                nonlocal detection_count

                if not batch_images:
                    return

                predictions = (
                    model.predict(
                        source=batch_images,

                        conf=
                            confidence_threshold,

                        iou=
                            iou_threshold,

                        verbose=False,
                    )
                )

                for (
                    prediction,
                    metadata,
                ) in zip(
                    predictions,
                    batch_metadata,
                ):
                    detections = (
                        parse_yolo_result(
                            prediction,
                            model,
                        )
                    )

                    frame_index = (
                        metadata[
                            "frame"
                        ]
                    )

                    detections_by_frame[
                        frame_index
                    ] = detections

                    processed_frames += 1

                    detection_count += (
                        len(
                            detections
                        )
                    )

                    frames_result.append(
                        {
                            "frame":
                                frame_index,

                            "time_seconds":
                                metadata[
                                    "time_seconds"
                                ],

                            "detections":
                                detections,
                        }
                    )

                batch_images.clear()
                batch_metadata.clear()

            # ------------------------------------------------
            # READ PASS 1
            # ------------------------------------------------

            while True:
                ret, frame = (
                    cap.read()
                )

                if not ret:
                    break

                if (
                    current_frame
                    % frame_skip
                    == 0
                ):
                    batch_images.append(
                        frame
                    )

                    batch_metadata.append(
                        {
                            "frame":
                                current_frame,

                            "time_seconds":
                                round(
                                    current_frame
                                    / fps,
                                    3,
                                ),
                        }
                    )

                    if (
                        len(batch_images)
                        >= batch_size
                    ):
                        process_annotation_batch()

                        if (
                            total_frames
                            > 0
                        ):
                            progress = min(
                                90.0,

                                (
                                    (
                                        current_frame
                                        + 1
                                    )
                                    /
                                    total_frames
                                )
                                * 90.0,
                            )

                        else:
                            progress = 0.0

                        update_progress(
                            job_id,
                            progress,
                            processed_frames,
                            total_frames,
                            detection_count,
                        )

                        print(
                            f"[JOB {job_id}] "
                            f"inference progress="
                            f"{progress:.2f}% "
                            f"processed="
                            f"{processed_frames} "
                            f"detections="
                            f"{detection_count}",
                            flush=True,
                        )

                current_frame += 1

            # last partial batch
            process_annotation_batch()

            # close pass 1 reader
            cap.release()
            cap = None

            print(
                f"[JOB {job_id}] "
                f"PASS 1 completed: "
                f"processed="
                f"{processed_frames}, "
                f"detections="
                f"{detection_count}",
                flush=True,
            )

            # =================================================
            # PASS 2 - WRITE ANNOTATED VIDEO
            # =================================================

            cap = cv2.VideoCapture(
                video_path
            )

            if not cap.isOpened():
                raise RuntimeError(
                    f"Cannot reopen video: "
                    f"{video_path}"
                )

            current_frame = 0

            last_detections = []

            while True:
                ret, frame = (
                    cap.read()
                )

                if not ret:
                    break

                # --------------------------------------------
                # detection mới cho sampled frame
                # --------------------------------------------

                if (
                    current_frame
                    in detections_by_frame
                ):
                    last_detections = (
                        detections_by_frame[
                            current_frame
                        ]
                    )

                # --------------------------------------------
                # draw
                # --------------------------------------------

                annotated_frame = (
                    draw_detections(
                        frame,
                        last_detections,
                    )
                )

                writer.write(
                    annotated_frame
                )

                # --------------------------------------------
                # annotation progress 90 -> 99
                # --------------------------------------------

                if (
                    current_frame
                    % 50
                    == 0
                ):
                    if (
                        total_frames
                        > 0
                    ):
                        progress = (
                            90.0
                            +
                            (
                                (
                                    current_frame
                                    + 1
                                )
                                /
                                total_frames
                            )
                            * 9.0
                        )

                        progress = min(
                            99.0,
                            progress,
                        )

                    else:
                        progress = 90.0

                    update_progress(
                        job_id,
                        progress,
                        processed_frames,
                        total_frames,
                        detection_count,
                    )

                    print(
                        f"[JOB {job_id}] "
                        f"annotation progress="
                        f"{progress:.2f}%",
                        flush=True,
                    )

                current_frame += 1

            # -----------------------------------------------
            # CLOSE PASS 2
            # -----------------------------------------------

            writer.release()
            writer = None

            cap.release()
            cap = None

            print(
                f"[JOB {job_id}] "
                f"PASS 2 completed",
                flush=True,
            )

            # =================================================
            # SAVE ANNOTATED VIDEO -> PUBLIC FILE SERVER
            # =================================================

            annotated_info = (
                save_annotated_video_file(
                    job_id=job_id,
                    video_index=0,
                    model_name=model_name,
                    source_path=annotated_path,
                )
            )

            annotated_video_url = (
                annotated_info["url"]
            )

            print(
                f"[JOB {job_id}] "
                f"annotated video saved: "
                f"path={annotated_info['path']} "
                f"url={annotated_video_url}",
                flush=True,
            )

        # ====================================================
        # MODE B:
        # NO ANNOTATED VIDEO
        #
        # batch inference only
        # ====================================================

        else:

            batch_images = []
            batch_metadata = []

            current_frame = 0

            def process_batch():
                nonlocal processed_frames
                nonlocal detection_count

                if not batch_images:
                    return

                predictions = (
                    model.predict(
                        source=
                            batch_images,

                        conf=
                            confidence_threshold,

                        iou=
                            iou_threshold,

                        verbose=False,
                    )
                )

                for (
                    prediction,
                    metadata,
                ) in zip(
                    predictions,
                    batch_metadata,
                ):
                    detections = (
                        parse_yolo_result(
                            prediction,
                            model,
                        )
                    )

                    processed_frames += 1

                    detection_count += (
                        len(
                            detections
                        )
                    )

                    frames_result.append(
                        {
                            "frame":
                                metadata[
                                    "frame"
                                ],

                            "time_seconds":
                                metadata[
                                    "time_seconds"
                                ],

                            "detections":
                                detections,
                        }
                    )

                batch_images.clear()
                batch_metadata.clear()

            # ------------------------------------------------
            # VIDEO LOOP
            # ------------------------------------------------

            while True:
                ret, frame = (
                    cap.read()
                )

                if not ret:
                    break

                if (
                    current_frame
                    % frame_skip
                    != 0
                ):
                    current_frame += 1
                    continue

                batch_images.append(
                    frame
                )

                batch_metadata.append(
                    {
                        "frame":
                            current_frame,

                        "time_seconds":
                            round(
                                current_frame
                                / fps,
                                3,
                            ),
                    }
                )

                if (
                    len(batch_images)
                    >= batch_size
                ):
                    process_batch()

                    if (
                        total_frames
                        > 0
                    ):
                        progress = min(
                            99.0,

                            (
                                (
                                    current_frame
                                    + 1
                                )
                                /
                                total_frames
                            )
                            * 100.0,
                        )

                    else:
                        progress = 0.0

                    update_progress(
                        job_id,
                        progress,
                        processed_frames,
                        total_frames,
                        detection_count,
                    )

                    print(
                        f"[JOB {job_id}] "
                        f"progress="
                        f"{progress:.2f}% "
                        f"processed="
                        f"{processed_frames} "
                        f"detections="
                        f"{detection_count}",
                        flush=True,
                    )

                current_frame += 1

            # partial batch
            process_batch()

            cap.release()
            cap = None

        # ====================================================
        # RESULT DATA
        # ====================================================

        result_data = {
            "job_id":
                job_id,

            "mission_id":
                mission_id,

            "status":
                "completed",

            "model": {
                "name":
                    model_name,

                "version":
                    model_version,

                "confidence_threshold":
                    confidence_threshold,

                "iou_threshold":
                    iou_threshold,
            },

            "video": {
                "fps":
                    round(
                        fps,
                        3,
                    ),

                "width":
                    width,

                "height":
                    height,

                "total_frames":
                    total_frames,

                "duration_seconds":
                    round(
                        duration_seconds,
                        3,
                    ),

                "frame_skip":
                    frame_skip,

                "batch_size":
                    batch_size,
            },

            "processed_frames":
                processed_frames,

            "detection_count":
                detection_count,

            "annotated_video": {
                "generated":
                    generate_annotated_video,

                "url":
                    annotated_video_url,

            },

            "frames":
                frames_result,
        }

        # ====================================================
        # RESULT JSON -> MINIO
        # ====================================================

        result_url = (
            upload_json(
                job_id,
                result_data,
            )
        )

        print(
            f"[JOB {job_id}] "
            f"result uploaded: "
            f"{result_url}",
            flush=True,
        )

        # ====================================================
        # DB COMPLETED
        # ====================================================

        mark_completed(
            job_id,
            result_url,
            processed_frames,
            total_frames,
            detection_count,
        )

        # ====================================================
        # MQTT COMPLETED
        # ====================================================
        replaced_result_url = result_url.replace(settings.MINIO_ENDPOINT, settings.MINIO_PUBLIC_ENDPOINT)
        publish(
            f"missions/"
            f"{mission_id}/"
            f"inference/completed",

            {
                "event":
                    "inference.completed",

                "job_id":
                    job_id,

                "mission_id":
                    mission_id,

                "status":
                    "completed",

                "model_name":
                    model_name,

                "model_version":
                    model_version,

                "result_url":replaced_result_url,
                    #result_url,

                "annotated_video_url":
                    annotated_video_url,

                "processed_frames":
                    processed_frames,

                "detection_count":
                    detection_count,
            }
        )

        print(
            f"[JOB {job_id}] "
            f"COMPLETED",
            flush=True,
        )

        return {
            "job_id":
                job_id,

            "status":
                "completed",

            "result_url":
                result_url,

            "annotated_video_url":
                annotated_video_url,

            "processed_frames":
                processed_frames,

            "detection_count":
                detection_count,
        }

    # ========================================================
    # FAILED
    # ========================================================

    except Exception as e:
        error_message = str(
            e
        )

        print(
            f"[JOB {job_id}] "
            f"FAILED: "
            f"{error_message}",
            flush=True,
        )

        traceback.print_exc()

        try:
            mark_failed(
                job_id,
                error_message,
            )

        except Exception as db_error:
            print(
                f"[JOB {job_id}] "
                f"mark_failed DB error: "
                f"{db_error}",
                flush=True,
            )

        try:
            publish(
                f"missions/"
                f"{mission_id}/"
                f"inference/failed",

                {
                    "event":
                        "inference.failed",

                    "job_id":
                        job_id,

                    "mission_id":
                        mission_id,

                    "status":
                        "failed",

                    "model_name":
                        model_name,

                    "model_version":
                        model_version,

                    "error":
                        error_message,
                }
            )

        except Exception as mqtt_error:
            print(
                f"[JOB {job_id}] "
                f"MQTT error: "
                f"{mqtt_error}",
                flush=True,
            )

        raise

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        # ----------------------------------------------------
        # Reader
        # ----------------------------------------------------

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        # ----------------------------------------------------
        # Writer
        # ----------------------------------------------------

        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass

        # ----------------------------------------------------
        # Annotated temporary file
        # ----------------------------------------------------

        if (
            annotated_path
            and
            os.path.isfile(
                annotated_path
            )
        ):
            try:
                os.remove(
                    annotated_path
                )

                print(
                    f"[JOB {job_id}] "
                    f"deleted annotated temp: "
                    f"{annotated_path}",
                    flush=True,
                )

            except Exception as e:
                print(
                    f"[JOB {job_id}] "
                    f"cannot delete "
                    f"annotated temp: "
                    f"{e}",
                    flush=True,
                )

        # ----------------------------------------------------
        # Input temporary/upload video
        # ----------------------------------------------------

        if (
            delete_after_process
            and
            video_path
            and
            os.path.isfile(
                video_path
            )
        ):
            try:
                os.remove(
                    video_path
                )

                print(
                    f"[JOB {job_id}] "
                    f"deleted temporary video: "
                    f"{video_path}",
                    flush=True,
                )

            except Exception as e:
                print(
                    f"[JOB {job_id}] "
                    f"cannot delete "
                    f"video: {e}",
                    flush=True,
                )

def encode_annotated_image(
    image,
    detections,
    filename: str,
):
    """
    Vẽ detections lên ảnh và encode về bytes
    để upload lên MinIO.
    """

    annotated = draw_detections(
        image,
        detections
    )

    extension = (
        os.path.splitext(
            filename
        )[1]
        .lower()
    )

    # Chuẩn hóa extension
    if extension == ".jpeg":
        extension = ".jpg"

    if extension not in (
        ".jpg",
        ".png"
    ):
        extension = ".jpg"

    # Encode image
    ok, encoded = cv2.imencode(
        extension,
        annotated
    )

    if not ok:
        raise RuntimeError(
            f"Cannot encode annotated image: {filename}"
        )

    return encoded.tobytes()

def draw_merged_image_detections(image, model_results: list):
    """Draw detections from all models on one image."""
    annotated = image.copy()
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (0, 165, 255), (128, 0, 255),
    ]
    for model_index, model_result in enumerate(model_results):
        color = colors[model_index % len(colors)]
        model_name = str(model_result.get("model_name", "model"))
        mission_code = str(model_result.get("mission_code", ""))
        prefix = mission_code or model_name
        for detection in model_result.get("detections", []):
            bbox = detection.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            class_name = str(detection.get("class_name", "object"))
            confidence = float(detection.get("confidence", 0))
            label = f"{prefix}:{class_name} {confidence:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            text_y = max(y1, th + 10)
            cv2.rectangle(annotated, (x1, text_y - th - 8), (x1 + tw + 8, text_y + 4), color, -1)
            cv2.putText(annotated, label, (x1 + 4, text_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return annotated


def generate_merged_annotated_image(job_id: str, image_index: int, image, model_results: list):
    """Create and save exactly one merged annotated image for one input image."""
    annotated = draw_merged_image_detections(image, model_results)
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError(f"Cannot encode merged annotated image index={image_index}")
    return save_annotated_image_bytes(
        job_id=job_id,
        image_index=image_index,
        model_name="merged",
        image_bytes=encoded.tobytes(),
        extension=".jpg",
    )


@celery_app.task(
    name="process_images_multi"
)
def process_images_multi(
        job_id: str,
        models: list,
        images: list,
        generate_annotated_images: bool = False,
        image_batch_size: int = IMAGE_BATCH_SIZE,
):
    generate_annotated_images = bool(
        generate_annotated_images
    )

    image_batch_size = max(
        1,
        int(image_batch_size)
    )

    try:

        # ====================================================
        # JOB RUNNING
        # ====================================================

        mark_running(
            job_id
        )

        print(
            f"[JOB {job_id}] "
            f"multi-image inference started",
            flush=True
        )

        print(
            f"[JOB {job_id}] "
            f"models={len(models)}, "
            f"images={len(images)}, "
            f"annotated="
            f"{generate_annotated_images}",
            flush=True
        )

        # ====================================================
        # MQTT STARTED
        #
        # 'multi' vẫn match:
        # missions/+/inference/#
        # ====================================================

        publish(
            "missions/"
            "multi/"
            "inference/started",

            {
                "event":
                    "inference.started",

                "job_id":
                    job_id,

                "status":
                    "running",

                "input_type":
                    "IMAGE_FILES_MULTI",

                "image_count":
                    len(images),

                "model_count":
                    len(models),

                "generate_annotated_images":
                    generate_annotated_images,

                "models": [
                    {
                        "mission_code":
                            m[
                                "mission_code"
                            ],

                        "model_name":
                            m[
                                "model_name"
                            ],

                        "model_version":
                            m[
                                "model_version"
                            ],
                    }
                    for m in models
                ],
            }
        )

        # ====================================================
        # LOAD IMAGES ONCE
        # ====================================================

        loaded_images = []

        for index, item in enumerate(
                images
        ):

            path = item[
                "path"
            ]

            image = cv2.imread(
                path
            )

            if image is None:
                raise RuntimeError(
                    f"Cannot read image: "
                    f"{path}"
                )

            height, width = (
                image.shape[:2]
            )

            loaded_images.append(
                {
                    "index":
                        index,

                    "filename":
                        item[
                            "filename"
                        ],

                    "path":
                        path,

                    "image":
                        image,

                    "width":
                        width,

                    "height":
                        height,
                }
            )

        # ====================================================
        # COUNTERS
        # ====================================================

        image_count = len(
            loaded_images
        )

        model_count = len(
            models
        )

        total_units = (
                image_count
                * model_count
        )

        completed_units = 0

        total_detection_count = 0

        model_outputs = []

        # ====================================================
        # MODEL LOOP
        #
        # model được load 1 lần
        # rồi infer tất cả ảnh theo batch
        # ====================================================

        for model_config in models:

            mission_id = str(
                model_config[
                    "mission_id"
                ]
            )

            mission_code = (
                model_config[
                    "mission_code"
                ]
            )

            model_name = (
                model_config[
                    "model_name"
                ]
            )

            model_version = (
                model_config[
                    "model_version"
                ]
            )

            model_path = (
                model_config[
                    "model_path"
                ]
            )

            confidence_threshold = float(
                model_config[
                    "confidence_threshold"
                ]
            )

            iou_threshold = float(
                model_config[
                    "iou_threshold"
                ]
            )

            print(
                f"[JOB {job_id}] "
                f"loading "
                f"{model_name}:"
                f"{model_version}",
                flush=True
            )

            model = (
                model_manager
                .get_model(
                    model_name,
                    model_version,
                    model_path,
                )
            )

            model_detection_count = 0

            image_outputs = []

            # =================================================
            # IMAGE BATCH LOOP
            # =================================================

            for start in range(
                    0,
                    image_count,
                    image_batch_size
            ):

                batch = (
                    loaded_images[
                    start:
                    start
                    + image_batch_size
                    ]
                )

                batch_sources = [
                    item[
                        "image"
                    ]
                    for item in batch
                ]

                predictions = (
                    model.predict(
                        source=
                        batch_sources,

                        conf=
                        confidence_threshold,

                        iou=
                        iou_threshold,

                        verbose=False,
                    )
                )

                # =============================================
                # RESULT PER IMAGE
                # =============================================

                for (
                        prediction,
                        image_info
                ) in zip(
                    predictions,
                    batch
                ):

                    detections = (
                        parse_yolo_result(
                            prediction,
                            model
                        )
                    )

                    detection_count = len(
                        detections
                    )

                    model_detection_count += (
                        detection_count
                    )

                    total_detection_count += (
                        detection_count
                    )

                    completed_units += 1

                    # =========================================
                    # IMAGE RESULT
                    # =========================================

                    image_outputs.append(
                        {
                            "index":
                                image_info[
                                    "index"
                                ],

                            "filename":
                                image_info[
                                    "filename"
                                ],

                            "width":
                                image_info[
                                    "width"
                                ],

                            "height":
                                image_info[
                                    "height"
                                ],

                            "detection_count":
                                detection_count,

                            "detections":
                                detections,
                        }
                    )

                    # =========================================
                    # PROGRESS
                    # =========================================

                    progress = (
                            completed_units
                            / total_units
                            * 100.0
                    )

                    update_progress(
                        job_id,
                        progress,
                        completed_units,
                        total_units,
                        total_detection_count,
                    )

                print(
                    f"[JOB {job_id}] "
                    f"{model_name}: "
                    f"batch "
                    f"{start // image_batch_size + 1} "
                    f"done",
                    flush=True
                )

            # =================================================
            # MODEL RESULT
            # =================================================

            model_outputs.append(
                {
                    "mission_id":
                        mission_id,

                    "mission_code":
                        mission_code,

                    "model_name":
                        model_name,

                    "model_version":
                        model_version,

                    "confidence_threshold":
                        confidence_threshold,

                    "iou_threshold":
                        iou_threshold,

                    "detection_count":
                        model_detection_count,

                    "images":
                        image_outputs,
                }
            )

        # ====================================================
        # MERGED ANNOTATED IMAGES - ONE OUTPUT PER INPUT IMAGE
        # ====================================================
        merged_image_outputs = []

        for image_info in loaded_images:
            image_index = image_info["index"]
            per_model_results = []

            for model_output in model_outputs:
                image_result = next(
                    (
                        item
                        for item in model_output.get("images", [])
                        if item.get("index") == image_index
                    ),
                    None,
                )
                if image_result is None:
                    continue

                per_model_results.append(
                    {
                        "mission_id": model_output["mission_id"],
                        "mission_code": model_output["mission_code"],
                        "model_name": model_output["model_name"],
                        "model_version": model_output["model_version"],
                        "detection_count": image_result.get("detection_count", 0),
                        "detections": image_result.get("detections", []),
                    }
                )

            merged_annotated_image_url = None
            if generate_annotated_images:
                annotated_info = generate_merged_annotated_image(
                    job_id=job_id,
                    image_index=image_index,
                    image=image_info["image"],
                    model_results=per_model_results,
                )
                merged_annotated_image_url = annotated_info["url"]

            merged_item = {
                "index": image_index,
                "filename": image_info["filename"],
                "width": image_info["width"],
                "height": image_info["height"],
                "detection_count": sum(
                    item.get("detection_count", 0)
                    for item in per_model_results
                ),
                "merged_annotated_image_url": merged_annotated_image_url,
                "models": per_model_results,
            }
            merged_image_outputs.append(merged_item)

        # ====================================================
        # FINAL RESULT JSON
        # ====================================================

        result_data = {

            "job_id":
                job_id,

            "status":
                "completed",

            "input_type":
                "IMAGE_FILES_MULTI",

            "image_count":
                image_count,

            "model_count":
                model_count,

            "total_inference_units":
                total_units,

            "detection_count":
                total_detection_count,

            "generate_annotated_images":
                generate_annotated_images,

            "images":
                merged_image_outputs,

            "models":
                model_outputs,
        }

        # ====================================================
        # MINIO RESULT.JSON
        # ====================================================

        result_url = upload_json(
            job_id,
            result_data
        )

        # ====================================================
        # DB COMPLETED
        # ====================================================

        mark_completed(
            job_id,
            result_url,
            completed_units,
            total_units,
            total_detection_count,
        )

        # ====================================================
        # MQTT COMPLETED
        # ====================================================
        replaced_result_url= result_url.replace(settings.MINIO_ENDPOINT,settings.MINIO_PUBLIC_ENDPOINT)
        publish(
            "missions/"
            "multi/"
            "inference/completed",

            {
                "event":
                    "inference.completed",

                "job_id":
                    job_id,

                "status":
                    "completed",

                "input_type":
                    "IMAGE_FILES_MULTI",

                "image_count":
                    image_count,

                "model_count":
                    model_count,

                "total_inference_units":
                    total_units,

                "detection_count":
                    total_detection_count,

                "generate_annotated_images":
                    generate_annotated_images,

                "result_url":replaced_result_url,
                    #result_url,
            }
        )

        print(
            f"[JOB {job_id}] "
            f"multi-image completed",
            flush=True
        )

        return {
            "job_id":
                job_id,

            "status":
                "completed",

            "result_url":
                result_url,

            "image_count":
                image_count,

            "model_count":
                model_count,

            "detection_count":
                total_detection_count,
        }

    # ========================================================
    # FAILED
    # ========================================================

    except Exception as e:

        error_message = str(
            e
        )

        print(
            f"[JOB {job_id}] "
            f"FAILED: "
            f"{error_message}",
            flush=True
        )

        traceback.print_exc()

        try:
            mark_failed(
                job_id,
                error_message
            )

        except Exception:
            pass

        try:
            publish(
                "missions/"
                "multi/"
                "inference/failed",

                {
                    "event":
                        "inference.failed",

                    "job_id":
                        job_id,

                    "status":
                        "failed",

                    "input_type":
                        "IMAGE_FILES_MULTI",

                    "error":
                        error_message,
                }
            )

        except Exception:
            pass

        raise

    # ========================================================
    # CLEANUP ORIGINAL UPLOADS
    # ========================================================

    finally:

        for item in images:

            try:

                path = item.get(
                    "path"
                )

                if (
                        path
                        and
                        os.path.isfile(
                            path
                        )
                ):
                    os.remove(
                        path
                    )

            except Exception:
                pass

        if images:

            try:

                upload_dir = (
                    os.path.dirname(
                        images[0][
                            "path"
                        ]
                    )
                )

                if os.path.isdir(
                        upload_dir
                ):
                    os.rmdir(
                        upload_dir
                    )

            except Exception:
                pass


def download_image_from_url(
    image_url: str
):
    """
    Download image URL trực tiếp vào memory.

    Return:
        OpenCV BGR image
    """

    response = requests.get(
        image_url,
        timeout=IMAGE_DOWNLOAD_TIMEOUT,
        allow_redirects=True,
        stream=True,
    )

    response.raise_for_status()

    # ========================================================
    # CONTENT TYPE
    # ========================================================

    content_type = (
        response.headers.get(
            "content-type",
            ""
        )
        .lower()
    )

    if (
        content_type
        and
        not content_type.startswith(
            "image/"
        )
    ):
        raise RuntimeError(
            f"URL is not image: "
            f"{image_url}, "
            f"content-type={content_type}"
        )

    # ========================================================
    # LIMIT DOWNLOAD SIZE
    # ========================================================

    chunks = []

    total_size = 0

    for chunk in response.iter_content(
        chunk_size=1024 * 1024
    ):
        if not chunk:
            continue

        total_size += len(
            chunk
        )

        if (
            total_size
            > MAX_IMAGE_DOWNLOAD_SIZE
        ):
            raise RuntimeError(
                f"Image too large "
                f"(max 20MB): "
                f"{image_url}"
            )

        chunks.append(
            chunk
        )

    raw = b"".join(
        chunks
    )

    # ========================================================
    # DECODE
    # ========================================================

    np_data = np.frombuffer(
        raw,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        np_data,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            f"Cannot decode image URL: "
            f"{image_url}"
        )

    return image


@celery_app.task(
    name="process_image_urls_multi"
)
def process_image_urls_multi(
    job_id: str,
    models: list,
    image_urls: list,
    generate_annotated_images: bool = False,
    image_batch_size: int = 8,
):
    generate_annotated_images = bool(
        generate_annotated_images
    )

    image_batch_size = max(
        1,
        int(image_batch_size)
    )

    # Safety
    if len(image_urls) > 100:
        raise ValueError(
            "Maximum 100 image URLs per job"
        )

    try:
        # ====================================================
        # RUNNING
        # ====================================================

        mark_running(
            job_id
        )

        print(
            f"[JOB {job_id}] "
            f"multi image URL started "
            f"images={len(image_urls)} "
            f"models={len(models)} "
            f"annotated="
            f"{generate_annotated_images}",
            flush=True
        )

        # ====================================================
        # MQTT STARTED
        # ====================================================

        publish(
            "missions/"
            "multi/"
            "inference/started",

            {
                "event":
                    "inference.started",

                "job_id":
                    job_id,

                "status":
                    "running",

                "input_type":
                    "IMAGE_URLS_MULTI",

                "image_count":
                    len(image_urls),

                "model_count":
                    len(models),

                "generate_annotated_images":
                    generate_annotated_images,

                "models": [
                    {
                        "mission_code":
                            m[
                                "mission_code"
                            ],

                        "model_name":
                            m[
                                "model_name"
                            ],

                        "model_version":
                            m[
                                "model_version"
                            ],
                    }

                    for m in models
                ],
            }
        )

        # ====================================================
        # DOWNLOAD IMAGES
        # ====================================================

        loaded_images = []

        download_errors = []

        for index, image_url in enumerate(
            image_urls
        ):
            try:
                print(
                    f"[JOB {job_id}] "
                    f"download "
                    f"{index + 1}/"
                    f"{len(image_urls)}: "
                    f"{image_url}",
                    flush=True
                )

                image = (
                    download_image_from_url(
                        image_url
                    )
                )

                height, width = (
                    image.shape[:2]
                )

                # filename dùng cho annotated MinIO
                filename = (
                    f"{index:04d}.jpg"
                )

                loaded_images.append(
                    {
                        "index":
                            index,

                        "url":
                            image_url,

                        "filename":
                            filename,

                        "image":
                            image,

                        "width":
                            width,

                        "height":
                            height,
                    }
                )

            except Exception as e:

                download_errors.append(
                    {
                        "index":
                            index,

                        "url":
                            image_url,

                        "error":
                            str(e),
                    }
                )

                print(
                    f"[JOB {job_id}] "
                    f"download FAILED: "
                    f"{image_url}: {e}",
                    flush=True
                )

        # Không có ảnh nào dùng được
        if not loaded_images:
            raise RuntimeError(
                "Cannot download/decode any image URL"
            )

        # ====================================================
        # COUNTERS
        # ====================================================

        image_count = len(
            loaded_images
        )

        model_count = len(
            models
        )

        total_units = (
            image_count
            * model_count
        )

        completed_units = 0

        total_detection_count = 0

        model_outputs = []

        # ====================================================
        # MODEL LOOP
        # ====================================================

        for model_config in models:

            mission_id = str(
                model_config[
                    "mission_id"
                ]
            )

            mission_code = (
                model_config[
                    "mission_code"
                ]
            )

            model_name = (
                model_config[
                    "model_name"
                ]
            )

            model_version = (
                model_config[
                    "model_version"
                ]
            )

            model_path = (
                model_config[
                    "model_path"
                ]
            )

            confidence_threshold = float(
                model_config[
                    "confidence_threshold"
                ]
            )

            iou_threshold = float(
                model_config[
                    "iou_threshold"
                ]
            )

            print(
                f"[JOB {job_id}] "
                f"loading "
                f"{model_name}:"
                f"{model_version}",
                flush=True
            )

            # =================================================
            # LOAD MODEL ONCE
            # =================================================

            model = (
                model_manager
                .get_model(
                    model_name,
                    model_version,
                    model_path,
                )
            )

            model_detection_count = 0

            image_outputs = []

            # =================================================
            # BATCH LOOP
            # =================================================

            for start in range(
                0,
                image_count,
                image_batch_size
            ):
                batch = (
                    loaded_images[
                        start:
                        start
                        + image_batch_size
                    ]
                )

                batch_sources = [
                    item[
                        "image"
                    ]
                    for item in batch
                ]

                predictions = (
                    model.predict(
                        source=
                            batch_sources,

                        conf=
                            confidence_threshold,

                        iou=
                            iou_threshold,

                        verbose=False,
                    )
                )

                # =============================================
                # RESULT PER IMAGE
                # =============================================

                for (
                    prediction,
                    image_info
                ) in zip(
                    predictions,
                    batch
                ):
                    detections = (
                        parse_yolo_result(
                            prediction,
                            model
                        )
                    )

                    detection_count = len(
                        detections
                    )

                    model_detection_count += (
                        detection_count
                    )

                    total_detection_count += (
                        detection_count
                    )

                    completed_units += 1

                    # =========================================
                    # IMAGE OUTPUT
                    # =========================================

                    image_outputs.append(
                        {
                            "index":
                                image_info[
                                    "index"
                                ],

                            "url":
                                image_info[
                                    "url"
                                ],

                            "filename":
                                image_info[
                                    "filename"
                                ],

                            "width":
                                image_info[
                                    "width"
                                ],

                            "height":
                                image_info[
                                    "height"
                                ],

                            "detection_count":
                                detection_count,

                            "detections":
                                detections,
                        }
                    )

                    # =========================================
                    # PROGRESS
                    # =========================================

                    progress = (
                        completed_units
                        / total_units
                        * 100.0
                    )

                    update_progress(
                        job_id,
                        progress,
                        completed_units,
                        total_units,
                        total_detection_count,
                    )

                print(
                    f"[JOB {job_id}] "
                    f"{model_name}: "
                    f"batch "
                    f"{start // image_batch_size + 1} "
                    f"done",
                    flush=True
                )

            # =================================================
            # MODEL OUTPUT
            # =================================================

            model_outputs.append(
                {
                    "mission_id":
                        mission_id,

                    "mission_code":
                        mission_code,

                    "model_name":
                        model_name,

                    "model_version":
                        model_version,

                    "confidence_threshold":
                        confidence_threshold,

                    "iou_threshold":
                        iou_threshold,

                    "detection_count":
                        model_detection_count,

                    "images":
                        image_outputs,
                }
            )

        # ====================================================
        # MERGED ANNOTATED IMAGES - ONE OUTPUT PER INPUT IMAGE
        # ====================================================
        merged_image_outputs = []

        for image_info in loaded_images:
            image_index = image_info["index"]
            per_model_results = []

            for model_output in model_outputs:
                image_result = next(
                    (
                        item
                        for item in model_output.get("images", [])
                        if item.get("index") == image_index
                    ),
                    None,
                )
                if image_result is None:
                    continue

                per_model_results.append(
                    {
                        "mission_id": model_output["mission_id"],
                        "mission_code": model_output["mission_code"],
                        "model_name": model_output["model_name"],
                        "model_version": model_output["model_version"],
                        "detection_count": image_result.get("detection_count", 0),
                        "detections": image_result.get("detections", []),
                    }
                )

            merged_annotated_image_url = None
            if generate_annotated_images:
                annotated_info = generate_merged_annotated_image(
                    job_id=job_id,
                    image_index=image_index,
                    image=image_info["image"],
                    model_results=per_model_results,
                )
                merged_annotated_image_url = annotated_info["url"]

            merged_item = {
                "index": image_index,
                "filename": image_info["filename"],
                "width": image_info["width"],
                "height": image_info["height"],
                "detection_count": sum(
                    item.get("detection_count", 0)
                    for item in per_model_results
                ),
                "merged_annotated_image_url": merged_annotated_image_url,
                "models": per_model_results,
            }
            merged_item["url"] = image_info["url"]
            merged_image_outputs.append(merged_item)

        # ====================================================
        # RESULT
        # ====================================================

        result_data = {
            "job_id":
                job_id,

            "status":
                "completed",

            "input_type":
                "IMAGE_URLS_MULTI",

            # số URL user gửi
            "requested_image_count":
                len(image_urls),

            # số ảnh download thành công
            "image_count":
                image_count,

            "failed_image_count":
                len(download_errors),

            "download_errors":
                download_errors,

            "model_count":
                model_count,

            "total_inference_units":
                total_units,

            "detection_count":
                total_detection_count,

            "generate_annotated_images":
                generate_annotated_images,

            "images":
                merged_image_outputs,

            "models":
                model_outputs,
        }

        # ====================================================
        # MINIO
        # ====================================================

        result_url = upload_json(
            job_id,
            result_data
        )

        # ====================================================
        # DB
        # ====================================================

        mark_completed(
            job_id,
            result_url,
            completed_units,
            total_units,
            total_detection_count,
        )

        # ====================================================
        # MQTT COMPLETED
        # ====================================================
        replaced_result_url = result_url.replace(settings.MINIO_ENDPOINT,settings.MINIO_PUBLIC_ENDPOINT)
        publish(
            "missions/"
            "multi/"
            "inference/completed",

            {
                "event":
                    "inference.completed",

                "job_id":
                    job_id,

                "status":
                    "completed",

                "input_type":
                    "IMAGE_URLS_MULTI",

                "requested_image_count":
                    len(image_urls),

                "image_count":
                    image_count,

                "failed_image_count":
                    len(download_errors),

                "model_count":
                    model_count,

                "detection_count":
                    total_detection_count,

                "generate_annotated_images":
                    generate_annotated_images,

                "result_url":replaced_result_url,
                    #result_url,
            }
        )

        print(
            f"[JOB {job_id}] "
            f"multi image URL completed",
            flush=True
        )

        return {
            "job_id":
                job_id,

            "status":
                "completed",

            "result_url":
                result_url,

            "requested_image_count":
                len(image_urls),

            "image_count":
                image_count,

            "failed_image_count":
                len(download_errors),

            "model_count":
                model_count,

            "detection_count":
                total_detection_count,
        }

    # ========================================================
    # FAILED
    # ========================================================

    except Exception as e:

        error_message = str(
            e
        )

        print(
            f"[JOB {job_id}] "
            f"FAILED: "
            f"{error_message}",
            flush=True
        )

        traceback.print_exc()

        try:
            mark_failed(
                job_id,
                error_message
            )

        except Exception:
            pass

        try:
            publish(
                "missions/"
                "multi/"
                "inference/failed",

                {
                    "event":
                        "inference.failed",

                    "job_id":
                        job_id,

                    "status":
                        "failed",

                    "input_type":
                        "IMAGE_URLS_MULTI",

                    "error":
                        error_message,
                }
            )

        except Exception:
            pass

        raise

def infer_video_for_model(
    job_id: str,
    video_index: int,
    video_path: str,
    model_config: dict,
    frame_skip: int = 5,
    batch_size: int = 8,
    generate_annotated_video: bool = False,
):
    """
    Run inference for exactly one model on one video.

    IMPORTANT:
    This helper is inference-only for multi-model video jobs.
    Annotated video generation is performed once, after all models
    have completed, by generate_merged_annotated_video().
    """
    model_name = model_config["model_name"]
    model_version = model_config["model_version"]
    model_path = model_config["model_path"]

    confidence_threshold = float(
        model_config["confidence_threshold"]
    )
    iou_threshold = float(
        model_config["iou_threshold"]
    )

    model = model_manager.get_model(
        model_name,
        model_version,
        model_path,
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )
    if fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration_seconds = (
        total_frames / fps
        if fps > 0
        else 0
    )

    frames_result = []
    processed_frames = 0
    detection_count = 0
    current_frame = 0
    batch_images = []
    batch_metadata = []

    def process_batch():
        nonlocal processed_frames
        nonlocal detection_count

        if not batch_images:
            return

        predictions = model.predict(
            source=batch_images,
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        for prediction, metadata in zip(
            predictions,
            batch_metadata,
        ):
            detections = parse_yolo_result(
                prediction,
                model,
            )

            processed_frames += 1
            detection_count += len(detections)

            frames_result.append(
                {
                    "frame":
                        metadata["frame"],

                    "time_seconds":
                        metadata["time_seconds"],

                    "detections":
                        detections,
                }
            )

        batch_images.clear()
        batch_metadata.clear()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if current_frame % frame_skip == 0:
                batch_images.append(frame)

                batch_metadata.append(
                    {
                        "frame":
                            current_frame,

                        "time_seconds":
                            round(
                                current_frame / fps,
                                3,
                            ),
                    }
                )

                if len(batch_images) >= batch_size:
                    process_batch()

            current_frame += 1

        process_batch()

    finally:
        cap.release()

    return {
        "mission_id":
            str(
                model_config[
                    "mission_id"
                ]
            ),

        "mission_code":
            model_config[
                "mission_code"
            ],

        "model_name":
            model_name,

        "model_version":
            model_version,

        "video": {
            "fps":
                round(fps, 3),

            "width":
                width,

            "height":
                height,

            "total_frames":
                total_frames,

            "duration_seconds":
                round(duration_seconds, 3),

            "frame_skip":
                frame_skip,

            "batch_size":
                batch_size,
        },

        "processed_frames":
            processed_frames,

        "detection_count":
            detection_count,

        # Multi-model jobs generate one merged annotated video only.
        "annotated_video_url":
            None,

        "frames":
            frames_result,
    }


def draw_merged_video_detections(
    frame,
    model_detections: list,
):
    """Draw detections from all models on one video frame."""
    annotated = frame.copy()

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 165, 255),
        (128, 0, 255),
    ]

    for model_index, model_item in enumerate(
        model_detections
    ):
        color = colors[
            model_index % len(colors)
        ]

        mission_code = str(
            model_item.get(
                "mission_code",
                ""
            )
        )
        model_name = str(
            model_item.get(
                "model_name",
                "model"
            )
        )
        prefix = mission_code or model_name

        for detection in model_item.get(
            "detections",
            []
        ):
            bbox = detection.get(
                "bbox",
                []
            )

            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [
                int(v)
                for v in bbox
            ]

            class_name = str(
                detection.get(
                    "class_name",
                    "object"
                )
            )
            confidence = float(
                detection.get(
                    "confidence",
                    0
                )
            )

            label = (
                f"{prefix}:"
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )

            text_y = max(
                y1,
                th + 10,
            )

            cv2.rectangle(
                annotated,
                (
                    x1,
                    text_y - th - 8,
                ),
                (
                    x1 + tw + 8,
                    text_y + 4,
                ),
                color,
                -1,
            )

            cv2.putText(
                annotated,
                label,
                (
                    x1 + 4,
                    text_y - 3,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    return annotated


def generate_merged_annotated_video(
    job_id: str,
    video_index: int,
    video_path: str,
    model_results: list,
):
    """
    Generate exactly one annotated video for one input video,
    containing detections from all models.
    """
    if not model_results:
        raise RuntimeError(
            "Cannot generate merged annotated video "
            "without model results"
        )

    video_meta = model_results[0].get(
        "video",
        {}
    )

    fps = float(
        video_meta.get(
            "fps",
            30.0
        )
    )
    if fps <= 0:
        fps = 30.0

    width = int(
        video_meta.get(
            "width",
            0
        )
    )
    height = int(
        video_meta.get(
            "height",
            0
        )
    )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid video size for merged annotation: "
            f"{width}x{height}"
        )

    detections_by_model = []

    for model_result in model_results:
        frame_map = {
            int(frame_item["frame"]):
                frame_item.get(
                    "detections",
                    []
                )
            for frame_item in model_result.get(
                "frames",
                []
            )
            if frame_item.get("frame") is not None
        }

        detections_by_model.append(
            {
                "mission_code":
                    model_result.get(
                        "mission_code"
                    ),

                "model_name":
                    model_result.get(
                        "model_name"
                    ),

                "frames":
                    frame_map,
            }
        )

    writer = None
    annotated_path = None
    cap = None

    try:
        (
            writer,
            annotated_path,
        ) = create_video_writer(
            fps,
            width,
            height,
        )

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot reopen video for merged annotation: "
                f"{video_path}"
            )

        current_frame = 0

        last_detections = [
            []
            for _ in detections_by_model
        ]

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            draw_items = []

            for model_index, model_item in enumerate(
                detections_by_model
            ):
                frame_map = model_item[
                    "frames"
                ]

                if current_frame in frame_map:
                    last_detections[
                        model_index
                    ] = frame_map[
                        current_frame
                    ]

                draw_items.append(
                    {
                        "mission_code":
                            model_item[
                                "mission_code"
                            ],

                        "model_name":
                            model_item[
                                "model_name"
                            ],

                        "detections":
                            last_detections[
                                model_index
                            ],
                    }
                )

            annotated_frame = (
                draw_merged_video_detections(
                    frame,
                    draw_items,
                )
            )

            writer.write(
                annotated_frame
            )

            current_frame += 1

        cap.release()
        cap = None

        # MP4 must be closed before copying to file server.
        writer.release()
        writer = None

        annotated_info = (
            save_annotated_video_file(
                job_id=
                    job_id,

                video_index=
                    video_index,

                model_name=
                    "merged",

                source_path=
                    annotated_path,
            )
        )

        print(
            f"[JOB {job_id}] "
            f"merged annotated video="
            f"{annotated_info['url']}",
            flush=True,
        )

        return annotated_info

    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass

        if (
            annotated_path
            and
            os.path.isfile(
                annotated_path
            )
        ):
            try:
                os.remove(
                    annotated_path
                )
            except Exception:
                pass

@celery_app.task(
    name="process_videos_url_multi"
)
def process_videos_url_multi(
    job_id: str,
    models: list,
    video_urls: list,

    generate_annotated_videos: bool = False,

    frame_skip: int = 5,
    batch_size: int = 8,
):

    generate_annotated_videos = bool(
        generate_annotated_videos
    )

    frame_skip = max(
        1,
        int(
            frame_skip
        )
    )

    batch_size = max(
        1,
        int(
            batch_size
        )
    )

    if len(video_urls) > 20:

        raise ValueError(
            "Maximum 20 videos per job"
        )

    try:

        # ====================================================
        # RUNNING
        # ====================================================

        mark_running(
            job_id
        )

        print(
            f"[JOB {job_id}] "
            f"multi-video URL started "
            f"videos={len(video_urls)} "
            f"models={len(models)} "
            f"annotated="
            f"{generate_annotated_videos}",
            flush=True
        )

        # ====================================================
        # MQTT START
        # ====================================================

        publish(
            "missions/"
            "multi/"
            "inference/started",

            {
                "event":
                    "inference.started",

                "job_id":
                    job_id,

                "status":
                    "running",

                "input_type":
                    "VIDEO_URLS_MULTI",

                "video_count":
                    len(
                        video_urls
                    ),

                "model_count":
                    len(
                        models
                    ),

                "generate_annotated_videos":
                    generate_annotated_videos,
            }
        )

        # ====================================================
        # COUNTERS
        # ====================================================

        total_units = (
            len(video_urls)
            * len(models)
        )

        completed_units = 0

        total_detection_count = 0

        total_processed_frames = 0

        video_outputs = []

        download_errors = []

        # ====================================================
        # VIDEO LOOP
        # ====================================================

        for video_index, video_url in enumerate(
            video_urls
        ):

            video_path = None

            try:

                print(
                    f"[JOB {job_id}] "
                    f"downloading video "
                    f"{video_index + 1}/"
                    f"{len(video_urls)}",
                    flush=True
                )

                video_path = (
                    download_video(
                        video_url
                    )
                )

                model_outputs = []

                # =============================================
                # MODEL LOOP
                # =============================================

                for model_config in models:

                    print(
                        f"[JOB {job_id}] "
                        f"video={video_index} "
                        f"model="
                        f"{model_config['model_name']}",
                        flush=True
                    )

                    model_result = (
                        infer_video_for_model(
                            job_id=
                                job_id,

                            video_index=
                                video_index,

                            video_path=
                                video_path,

                            model_config=
                                model_config,

                            frame_skip=
                                frame_skip,

                            batch_size=
                                batch_size,

                            # Annotation is generated once after all
                            # model inference has completed.
                            generate_annotated_video=
                                False,
                        )
                    )

                    model_outputs.append(
                        model_result
                    )

                    total_detection_count += (
                        model_result[
                            "detection_count"
                        ]
                    )

                    total_processed_frames += (
                        model_result[
                            "processed_frames"
                        ]
                    )

                    completed_units += 1

                    progress = (
                        completed_units
                        / total_units
                        * 100.0
                    )

                    update_progress(
                        job_id,
                        progress,
                        completed_units,
                        total_units,
                        total_detection_count,
                    )

                # =============================================
                # ONE MERGED ANNOTATED VIDEO PER INPUT VIDEO
                # =============================================

                merged_annotated_video_url = None

                if generate_annotated_videos:
                    annotated_info = (
                        generate_merged_annotated_video(
                            job_id=
                                job_id,

                            video_index=
                                video_index,

                            video_path=
                                video_path,

                            model_results=
                                model_outputs,
                        )
                    )

                    merged_annotated_video_url = (
                        annotated_info[
                            "url"
                        ]
                    )

                    # Backward compatibility:
                    # API/client cũ đọc annotated_video_url
                    # từ từng model. Tất cả cùng trỏ tới ONE
                    # merged annotated video.
                    for model_result in model_outputs:
                        model_result[
                            "annotated_video_url"
                        ] = merged_annotated_video_url

                # =============================================
                # VIDEO RESULT
                # =============================================

                video_outputs.append(
                    {
                        "index":
                            video_index,

                        "url":
                            video_url,

                        "status":
                            "completed",

                        "merged_annotated_video_url":
                            merged_annotated_video_url,

                        "models":
                            model_outputs,
                    }
                )

            except Exception as e:

                download_errors.append(
                    {
                        "index":
                            video_index,

                        "url":
                            video_url,

                        "error":
                            str(e),
                    }
                )

                video_outputs.append(
                    {
                        "index":
                            video_index,

                        "url":
                            video_url,

                        "status":
                            "failed",

                        "error":
                            str(e),

                        "models":
                            [],
                    }
                )

                print(
                    f"[JOB {job_id}] "
                    f"video {video_index} "
                    f"FAILED: {e}",
                    flush=True
                )

            finally:

                if (
                    video_path
                    and
                    os.path.isfile(
                        video_path
                    )
                ):
                    try:
                        os.remove(
                            video_path
                        )
                    except Exception:
                        pass

        # ====================================================
        # RESULT JSON
        # ====================================================

        successful_video_count = sum(
            1
            for video
            in video_outputs
            if (
                video[
                    "status"
                ]
                == "completed"
            )
        )

        if (
            successful_video_count
            == 0
        ):
            raise RuntimeError(
                "All video URLs failed"
            )

        result_data = {

            "job_id":
                job_id,

            "status":
                "completed",

            "input_type":
                "VIDEO_URLS_MULTI",

            "requested_video_count":
                len(
                    video_urls
                ),

            "video_count":
                successful_video_count,

            "failed_video_count":
                (
                    len(video_urls)
                    -
                    successful_video_count
                ),

            "model_count":
                len(
                    models
                ),

            "total_inference_units":
                total_units,

            "completed_inference_units":
                completed_units,

            "processed_frames":
                total_processed_frames,

            "detection_count":
                total_detection_count,

            "generate_annotated_videos":
                generate_annotated_videos,

            "frame_skip":
                frame_skip,

            "batch_size":
                batch_size,

            "errors":
                download_errors,

            "videos":
                video_outputs,
        }

        # ====================================================
        # MINIO
        # ====================================================

        result_url = (
            upload_json(
                job_id,
                result_data
            )
        )

        # ====================================================
        # DB
        # ====================================================

        mark_completed(
            job_id,
            result_url,
            completed_units,
            total_units,
            total_detection_count,
        )

        # ====================================================
        # MQTT COMPLETE
        # ====================================================
        replaced_result_url = result_url.replace(settings.MINIO_ENDPOINT,settings.MINIO_PUBLIC_ENDPOINT)
        publish(
            "missions/"
            "multi/"
            "inference/completed",

            {
                "event":
                    "inference.completed",

                "job_id":
                    job_id,

                "status":
                    "completed",

                "input_type":
                    "VIDEO_URLS_MULTI",

                "requested_video_count":
                    len(
                        video_urls
                    ),

                "video_count":
                    successful_video_count,

                "failed_video_count":
                    (
                        len(video_urls)
                        -
                        successful_video_count
                    ),

                "model_count":
                    len(
                        models
                    ),

                "detection_count":
                    total_detection_count,

                "generate_annotated_videos":
                    generate_annotated_videos,

                "result_url":replaced_result_url,
                    #result_url,
            }
        )

        return {
            "job_id":
                job_id,

            "status":
                "completed",

            "result_url":
                result_url,

            "video_count":
                successful_video_count,

            "model_count":
                len(
                    models
                ),

            "detection_count":
                total_detection_count,
        }

    except Exception as e:

        error_message = str(
            e
        )

        print(
            f"[JOB {job_id}] "
            f"FAILED: {error_message}",
            flush=True
        )

        traceback.print_exc()

        try:

            mark_failed(
                job_id,
                error_message
            )

        except Exception:
            pass

        try:

            publish(
                "missions/"
                "multi/"
                "inference/failed",

                {
                    "event":
                        "inference.failed",

                    "job_id":
                        job_id,

                    "status":
                        "failed",

                    "input_type":
                        "VIDEO_URLS_MULTI",

                    "error":
                        error_message,
                }
            )

        except Exception:
            pass

        raise

@celery_app.task(
    name="process_videos_minio_multi"
)
def process_videos_minio_multi(
    job_id: str,
    models: list,
    videos: list,

    generate_annotated_videos: bool = False,

    frame_skip: int = 5,
    batch_size: int = 8,
):

    generate_annotated_videos = bool(
        generate_annotated_videos
    )

    frame_skip = max(
        1,
        int(frame_skip)
    )

    batch_size = max(
        1,
        int(batch_size)
    )

    if len(videos) > 20:
        raise ValueError(
            "Maximum 20 MinIO videos per job"
        )

    try:

        # ====================================================
        # JOB RUNNING
        # ====================================================

        mark_running(
            job_id
        )

        print(
            f"[JOB {job_id}] "
            f"multi MinIO video started "
            f"videos={len(videos)} "
            f"models={len(models)} "
            f"annotated="
            f"{generate_annotated_videos}",
            flush=True,
        )

        # ====================================================
        # MQTT START
        # ====================================================

        publish(
            "missions/"
            "multi/"
            "inference/started",

            {
                "event":
                    "inference.started",

                "job_id":
                    job_id,

                "status":
                    "running",

                "input_type":
                    "VIDEO_MINIO_MULTI",

                "video_count":
                    len(videos),

                "model_count":
                    len(models),

                "generate_annotated_videos":
                    generate_annotated_videos,
            }
        )

        # ====================================================
        # COUNTERS
        # ====================================================

        total_units = (
            len(videos)
            * len(models)
        )

        completed_units = 0

        total_detection_count = 0

        total_processed_frames = 0

        video_outputs = []

        video_errors = []

        # ====================================================
        # VIDEO LOOP
        # ====================================================

        for (
            video_index,
            video_item
        ) in enumerate(
            videos
        ):

            video_path = None

            bucket = video_item[
                "bucket"
            ]

            object_key = video_item[
                "object_key"
            ]

            try:

                print(
                    f"[JOB {job_id}] "
                    f"download MinIO "
                    f"{video_index + 1}/"
                    f"{len(videos)} "
                    f"{bucket}/{object_key}",
                    flush=True,
                )

                # =============================================
                # DOWNLOAD DIRECT FROM MINIO
                # =============================================

                video_path = (
                    download_minio_object(
                        bucket=bucket,
                        object_key=object_key,
                    )
                )

                # =============================================
                # VALIDATE VIDEO
                # =============================================

                cap = cv2.VideoCapture(
                    video_path
                )

                try:

                    if not cap.isOpened():

                        raise RuntimeError(
                            f"MinIO object is not "
                            f"a readable video: "
                            f"{bucket}/{object_key}"
                        )

                    total_frames = int(
                        cap.get(
                            cv2.CAP_PROP_FRAME_COUNT
                        )
                    )

                    fps = float(
                        cap.get(
                            cv2.CAP_PROP_FPS
                        )
                    )

                    print(
                        f"[JOB {job_id}] "
                        f"MinIO video valid "
                        f"frames={total_frames} "
                        f"fps={fps:.3f}",
                        flush=True,
                    )

                finally:

                    cap.release()

                # =============================================
                # MODEL LOOP
                # =============================================

                model_outputs = []

                for model_config in models:

                    print(
                        f"[JOB {job_id}] "
                        f"video={video_index} "
                        f"model="
                        f"{model_config['model_name']}",
                        flush=True,
                    )

                    model_result = (
                        infer_video_for_model(
                            job_id=
                                job_id,

                            video_index=
                                video_index,

                            video_path=
                                video_path,

                            model_config=
                                model_config,

                            frame_skip=
                                frame_skip,

                            batch_size=
                                batch_size,

                            # Annotation is generated once after all
                            # model inference has completed.
                            generate_annotated_video=
                                False,
                        )
                    )

                    model_outputs.append(
                        model_result
                    )

                    total_detection_count += (
                        model_result[
                            "detection_count"
                        ]
                    )

                    total_processed_frames += (
                        model_result[
                            "processed_frames"
                        ]
                    )

                    completed_units += 1

                    progress = (
                        completed_units
                        / total_units
                        * 100.0
                    )

                    update_progress(
                        job_id,
                        progress,
                        completed_units,
                        total_units,
                        total_detection_count,
                    )

                # =============================================
                # ONE MERGED ANNOTATED VIDEO PER INPUT VIDEO
                # =============================================

                merged_annotated_video_url = None

                if generate_annotated_videos:
                    annotated_info = (
                        generate_merged_annotated_video(
                            job_id=
                                job_id,

                            video_index=
                                video_index,

                            video_path=
                                video_path,

                            model_results=
                                model_outputs,
                        )
                    )

                    merged_annotated_video_url = (
                        annotated_info[
                            "url"
                        ]
                    )

                    # Backward compatibility:
                    # all model entries point to the same
                    # merged annotated video URL.
                    for model_result in model_outputs:
                        model_result[
                            "annotated_video_url"
                        ] = merged_annotated_video_url

                # =============================================
                # VIDEO COMPLETED
                # =============================================

                video_outputs.append(
                    {
                        "index":
                            video_index,

                        "bucket":
                            bucket,

                        "object_key":
                            object_key,

                        "status":
                            "completed",

                        "merged_annotated_video_url":
                            merged_annotated_video_url,

                        "models":
                            model_outputs,
                    }
                )

            # =================================================
            # PER VIDEO FAILED
            # =================================================

            except Exception as e:

                error_item = {
                    "index":
                        video_index,

                    "bucket":
                        bucket,

                    "object_key":
                        object_key,

                    "error_type":
                        type(e).__name__,

                    "error":
                        str(e),
                }

                video_errors.append(
                    error_item
                )

                video_outputs.append(
                    {
                        "index":
                            video_index,

                        "bucket":
                            bucket,

                        "object_key":
                            object_key,

                        "status":
                            "failed",

                        "error":
                            str(e),

                        "models":
                            [],
                    }
                )

                print(
                    f"[JOB {job_id}] "
                    f"video[{video_index}] FAILED: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

                traceback.print_exc()

            # =================================================
            # TEMP CLEANUP
            # =================================================

            finally:

                if (
                    video_path
                    and
                    os.path.isfile(
                        video_path
                    )
                ):

                    try:

                        os.remove(
                            video_path
                        )

                    except Exception:
                        pass

        # ====================================================
        # SUMMARY
        # ====================================================

        successful_video_count = sum(
            1
            for video in video_outputs
            if (
                video.get(
                    "status"
                )
                == "completed"
            )
        )

        if successful_video_count == 0:

            details = "; ".join(
                [
                    (
                        f"video[{item['index']}] "
                        f"{item['error_type']}: "
                        f"{item['error']}"
                    )
                    for item
                    in video_errors
                ]
            )

            raise RuntimeError(
                "All MinIO videos failed"
                + (
                    f". {details}"
                    if details
                    else ""
                )
            )

        # ====================================================
        # RESULT
        # ====================================================

        result_data = {

            "job_id":
                job_id,

            "status":
                "completed",

            "input_type":
                "VIDEO_MINIO_MULTI",

            "requested_video_count":
                len(videos),

            "video_count":
                successful_video_count,

            "failed_video_count":
                (
                    len(videos)
                    -
                    successful_video_count
                ),

            "model_count":
                len(models),

            "total_inference_units":
                total_units,

            "completed_inference_units":
                completed_units,

            "processed_frames":
                total_processed_frames,

            "detection_count":
                total_detection_count,

            "generate_annotated_videos":
                generate_annotated_videos,

            "frame_skip":
                frame_skip,

            "batch_size":
                batch_size,

            "errors":
                video_errors,

            "videos":
                video_outputs,
        }

        # ====================================================
        # MINIO RESULT
        # ====================================================

        result_url = upload_json(
            job_id,
            result_data
        )

        # ====================================================
        # DB COMPLETE
        # ====================================================

        mark_completed(
            job_id,
            result_url,
            completed_units,
            total_units,
            total_detection_count,
        )

        # ====================================================
        # MQTT COMPLETE
        # ====================================================
        replaced_result_url = result_url.replace(settings.MINIO_ENDPOINT,settings.MINIO_PUBLIC_ENDPOINT)
        publish(
            "missions/"
            "multi/"
            "inference/completed",

            {
                "event":
                    "inference.completed",

                "job_id":
                    job_id,

                "status":
                    "completed",

                "input_type":
                    "VIDEO_MINIO_MULTI",

                "video_count":
                    successful_video_count,

                "failed_video_count":
                    (
                        len(videos)
                        -
                        successful_video_count
                    ),

                "model_count":
                    len(models),

                "detection_count":
                    total_detection_count,

                "generate_annotated_videos":
                    generate_annotated_videos,

                "result_url":replaced_result_url,
                    #result_url,
            }
        )

        return {
            "job_id":
                job_id,

            "status":
                "completed",

            "result_url":
                result_url,

            "video_count":
                successful_video_count,

            "model_count":
                len(models),

            "detection_count":
                total_detection_count,
        }

    # ========================================================
    # WHOLE JOB FAILED
    # ========================================================

    except Exception as e:

        error_message = str(
            e
        )

        print(
            f"[JOB {job_id}] "
            f"FAILED: {error_message}",
            flush=True,
        )

        traceback.print_exc()

        try:

            mark_failed(
                job_id,
                error_message
            )

        except Exception:
            pass

        try:

            publish(
                "missions/"
                "multi/"
                "inference/failed",

                {
                    "event":
                        "inference.failed",

                    "job_id":
                        job_id,

                    "status":
                        "failed",

                    "input_type":
                        "VIDEO_MINIO_MULTI",

                    "error":
                        error_message,
                }
            )

        except Exception:
            pass

        raise


@celery_app.task(
    name="process_rtsp_stream"
)
def process_rtsp_stream(
    job_id: str,
    models: list,
    rtsp_url: str,

    duration_seconds: int = 60,
    frame_skip: int = 5,
    batch_size: int = 8,

    generate_annotated_video: bool = False,
):
    cap = None
    writer = None
    annotated_path = None
    merged_annotated_video_url = None

    try:
        duration_seconds = max(
            5,
            int(duration_seconds)
        )

        frame_skip = max(
            1,
            int(frame_skip)
        )

        batch_size = max(
            1,
            int(batch_size)
        )

        generate_annotated_video = bool(
            generate_annotated_video
        )

        mark_running(
            job_id
        )

        publish(
            "missions/"
            "multi/"
            "inference/started",

            {
                "event":
                    "inference.started",

                "job_id":
                    job_id,

                "status":
                    "running",

                "input_type":
                    "RTSP_STREAM",

                "model_count":
                    len(models),

                "duration_seconds":
                    duration_seconds,

                "generate_annotated_video":
                    generate_annotated_video,
            }
        )

        print(
            f"[JOB {job_id}] "
            f"RTSP started "
            f"url={rtsp_url}",
            flush=True,
        )

        # ====================================================
        # LOAD ALL MODELS
        # ====================================================

        loaded_models = []

        for config in models:

            model = model_manager.get_model(
                config["model_name"],
                config["model_version"],
                config["model_path"],
            )

            loaded_models.append(
                {
                    "config":
                        config,

                    "model":
                        model,

                    "frames":
                        [],

                    "detection_count":
                        0,

                    "processed_frames":
                        0,
                }
            )

        # ====================================================
        # OPEN RTSP
        # ====================================================

        cap = cv2.VideoCapture(
            rtsp_url,
            cv2.CAP_FFMPEG
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open RTSP stream: "
                f"{rtsp_url}"
            )

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        if fps <= 0:
            fps = 25.0

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        if width <= 0 or height <= 0:
            raise RuntimeError(
                "Invalid RTSP stream resolution"
            )

        # ====================================================
        # ONE MERGED ANNOTATED VIDEO WRITER
        # ====================================================

        if generate_annotated_video:
            (
                writer,
                annotated_path,
            ) = create_video_writer(
                fps,
                width,
                height,
            )

        # ====================================================
        # STREAM LOOP
        # ====================================================

        started_at = time.monotonic()

        frame_index = 0

        last_detections = {
            item["config"]["model_name"]: []
            for item in loaded_models
        }

        while True:

            if (
                time.monotonic()
                - started_at
                >= duration_seconds
            ):
                break

            ret, frame = cap.read()

            if not ret:

                print(
                    f"[JOB {job_id}] "
                    f"RTSP read failed",
                    flush=True,
                )

                time.sleep(
                    0.2
                )

                continue

            sampled = (
                frame_index
                % frame_skip
                == 0
            )

            if sampled:

                for item in loaded_models:

                    config = item[
                        "config"
                    ]

                    model = item[
                        "model"
                    ]

                    prediction = (
                        model.predict(
                            source=frame,
                            conf=float(
                                config[
                                    "confidence_threshold"
                                ]
                            ),
                            iou=float(
                                config[
                                    "iou_threshold"
                                ]
                            ),
                            verbose=False,
                        )[0]
                    )

                    detections = (
                        parse_yolo_result(
                            prediction,
                            model
                        )
                    )

                    model_name = (
                        config[
                            "model_name"
                        ]
                    )

                    last_detections[
                        model_name
                    ] = detections

                    item[
                        "processed_frames"
                    ] += 1

                    item[
                        "detection_count"
                    ] += len(
                        detections
                    )

                    item[
                        "frames"
                    ].append(
                        {
                            "frame":
                                frame_index,

                            "time_seconds":
                                round(
                                    time.monotonic()
                                    - started_at,
                                    3,
                                ),

                            "detections":
                                detections,
                        }
                    )

            # =================================================
            # ONE MERGED ANNOTATED FRAME
            # =================================================

            if (
                generate_annotated_video
                and writer is not None
            ):
                draw_items = []

                for item in loaded_models:
                    config = item[
                        "config"
                    ]

                    model_name = config[
                        "model_name"
                    ]

                    draw_items.append(
                        {
                            "mission_code":
                                config.get(
                                    "mission_code"
                                ),

                            "model_name":
                                model_name,

                            "detections":
                                last_detections.get(
                                    model_name,
                                    []
                                ),
                        }
                    )

                annotated = (
                    draw_merged_video_detections(
                        frame,
                        draw_items,
                    )
                )

                writer.write(
                    annotated
                )

            frame_index += 1

            # progress theo thời gian
            progress = min(
                99.0,
                (
                    (
                        time.monotonic()
                        - started_at
                    )
                    /
                    duration_seconds
                )
                * 100.0
            )

            if frame_index % 30 == 0:

                total_detection_count = sum(
                    item[
                        "detection_count"
                    ]
                    for item
                    in loaded_models
                )

                total_processed = sum(
                    item[
                        "processed_frames"
                    ]
                    for item
                    in loaded_models
                )

                update_progress(
                    job_id,
                    progress,
                    total_processed,
                    frame_index,
                    total_detection_count,
                )

        # ====================================================
        # CLOSE STREAM
        # ====================================================

        cap.release()
        cap = None

        # ====================================================
        # CLOSE + UPLOAD ONE MERGED ANNOTATED VIDEO
        # ====================================================

        if generate_annotated_video:

            if writer is not None:
                writer.release()
                writer = None

            if annotated_path:

                annotated_info = (
                    save_annotated_video_file(
                        job_id=
                            job_id,

                        # RTSP has one input stream.
                        video_index=
                            0,

                        model_name=
                            "merged",

                        source_path=
                            annotated_path,
                    )
                )

                merged_annotated_video_url = (
                    annotated_info[
                        "url"
                    ]
                )

                print(
                    f"[JOB {job_id}] "
                    f"RTSP merged annotated video="
                    f"{merged_annotated_video_url}",
                    flush=True,
                )

        # ====================================================
        # RESULT
        # ====================================================

        model_results = []

        total_detection_count = 0

        total_processed_frames = 0

        for item in loaded_models:

            config = item[
                "config"
            ]

            model_name = (
                config[
                    "model_name"
                ]
            )

            total_detection_count += (
                item[
                    "detection_count"
                ]
            )

            total_processed_frames += (
                item[
                    "processed_frames"
                ]
            )

            model_results.append(
                {
                    "mission_id":
                        config[
                            "mission_id"
                        ],

                    "mission_code":
                        config[
                            "mission_code"
                        ],

                    "model_name":
                        model_name,

                    "model_version":
                        config[
                            "model_version"
                        ],

                    "processed_frames":
                        item[
                            "processed_frames"
                        ],

                    "detection_count":
                        item[
                            "detection_count"
                        ],

                    # Backward compatibility:
                    # every model points to the SAME merged video.
                    "annotated_video_url":
                        merged_annotated_video_url,

                    "frames":
                        item[
                            "frames"
                        ],
                }
            )

        result_data = {
            "job_id":
                job_id,

            "status":
                "completed",

            "input_type":
                "RTSP_STREAM",

            "rtsp_url":
                rtsp_url,

            "duration_seconds":
                duration_seconds,

            "frame_skip":
                frame_skip,

            "model_count":
                len(
                    loaded_models
                ),

            "processed_frames":
                total_processed_frames,

            "detection_count":
                total_detection_count,

            "generate_annotated_video":
                generate_annotated_video,

            "merged_annotated_video_url":
                merged_annotated_video_url,

            "models":
                model_results,
        }

        result_url = upload_json(
            job_id,
            result_data
        )

        mark_completed(
            job_id,
            result_url,
            total_processed_frames,
            frame_index,
            total_detection_count,
        )

        replaced_result_url = result_url.replace(
            settings.MINIO_ENDPOINT,
            settings.MINIO_PUBLIC_ENDPOINT
        )

        publish(
            "missions/"
            "multi/"
            "inference/completed",

            {
                "event":
                    "inference.completed",

                "job_id":
                    job_id,

                "status":
                    "completed",

                "input_type":
                    "RTSP_STREAM",

                "duration_seconds":
                    duration_seconds,

                "model_count":
                    len(
                        loaded_models
                    ),

                "processed_frames":
                    total_processed_frames,

                "detection_count":
                    total_detection_count,

                "merged_annotated_video_url":
                    merged_annotated_video_url,

                "result_url":
                    replaced_result_url,
            }
        )

        return {
            "job_id":
                job_id,

            "status":
                "completed",

            "result_url":
                result_url,

            "merged_annotated_video_url":
                merged_annotated_video_url,

            "processed_frames":
                total_processed_frames,

            "detection_count":
                total_detection_count,
        }

    except Exception as e:

        error_message = str(
            e
        )

        print(
            f"[JOB {job_id}] "
            f"RTSP FAILED: "
            f"{error_message}",
            flush=True,
        )

        traceback.print_exc()

        try:

            mark_failed(
                job_id,
                error_message
            )

        except Exception:
            pass

        try:

            publish(
                "missions/"
                "multi/"
                "inference/failed",

                {
                    "event":
                        "inference.failed",

                    "job_id":
                        job_id,

                    "status":
                        "failed",

                    "input_type":
                        "RTSP_STREAM",

                    "error":
                        error_message,
                }
            )

        except Exception:
            pass

        raise

    finally:

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass

        if (
            annotated_path
            and
            os.path.isfile(
                annotated_path
            )
        ):
            try:
                os.remove(
                    annotated_path
                )
            except Exception:
                pass

