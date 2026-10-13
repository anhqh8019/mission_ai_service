import io
import json
import os
import shutil
import uuid
from pathlib import Path

import requests
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from PIL import Image
from pydantic import BaseModel, HttpUrl

from .database import (
    create_job,
    get_active_model_by_mission,
    get_active_models_by_missions,
    get_job,
    get_supported_models
)
from .minio_service import get_json_from_minio
from .model_manager import model_manager
from .schemas import (
    InferenceRequest,
    LandcoverGeotiffRequest,
    MultiImageUrlRequest,
    MultiVideoUrlRequest,
    MultiMinioVideoRequest,
)
from .tasks import (
    process_image_urls_multi,
    process_images_multi,
    process_video,
    process_videos_url_multi,
    process_videos_minio_multi,
    process_rtsp_stream,
)
from .landcover_inputs import InputTooLarge, save_chunks
from .landcover_tasks import process_landcover_geotiff
from .landcover_inference import (
    build_public_geotiff_url,
    resolve_geotiff_source,
)

from .schemas import RtspInferenceRequest



# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Mission AI API",
    version="1.2.0"
)


# ============================================================
# IMAGE URL SCHEMA
# ============================================================

class ImageUrlRequest(
    BaseModel
):
    mission_id: str

    url: HttpUrl


# ============================================================
# RESOLVE MISSION
# ============================================================

def resolve_mission(
    mission_code: str
):

    mission = (
        get_active_model_by_mission(
            mission_code
        )
    )

    if not mission:

        raise HTTPException(
            status_code=404,

            detail=(
                f"No active model "
                f"found for mission: "
                f"{mission_code}"
            )
        )

    return {

        "mission_code":
            mission_code,

        "mission_uuid":
            str(
                mission[
                    "mission_id"
                ]
            ),

        "model_name":
            mission[
                "model_name"
            ],

        "model_version":
            mission[
                "model_version"
            ],

        "model_path":
            mission[
                "model_path"
            ],

        "conf":
            float(
                mission[
                    "confidence_threshold"
                ]
            ),

        "iou":
            float(
                mission[
                    "iou_threshold"
                ]
            ),
    }


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_image(
    mission,
    image
):

    model = (
        model_manager
        .get_model(
            mission[
                "model_name"
            ],

            mission[
                "model_version"
            ],

            mission[
                "model_path"
            ]
        )
    )

    prediction = (
        model.predict(
            image,

            conf=
                mission[
                    "conf"
                ],

            iou=
                mission[
                    "iou"
                ],

            verbose=False
        )[0]
    )

    detections = []

    for box in (
        prediction.boxes
    ):

        class_id = int(
            box.cls[0]
        )

        detections.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    model.names[
                        class_id
                    ],

                "confidence":
                    float(
                        box.conf[0]
                    ),

                "bbox":
                    box.xyxy[
                        0
                    ].tolist(),
            }
        )

    return detections


# ============================================================
# HEALTH
# ============================================================

@app.get("/")
def health():
    # supported_models ="mission-fire, mission-rust, mission-break_cabble_conductor, mission-cabinet,mission-crack,mission-damage_cabble,mission-hotspot,mission-nest,mission-rebar,mission-rru,mission-screw,mission-thermal,mission-tilt,mission-wire,mission-miss_bolt,mission-anten"
    # supported_models =get_supported_models()
    # return {
    #     "status": "ok",
    #     "service":"mission-ai-service",
    #     "version":"1.1.0",
    #     "supported_models":supported_models
    # }

    models = get_supported_models()

    return {
        "status": "ok",
        "service": "mission-ai-service",
        "models_supported": len(models),
        "models": [
            {
                "mission_id": str(model["mission_id"]),
                "mission_code": model["mission_code"],
                "mission_name": model["mission_name"],
                "model_name": model["model_name"],
                "model_version": model["model_version"],
                "confidence_threshold": float(
                    model["confidence_threshold"]
                ),
                "iou_threshold": float(
                    model["iou_threshold"]
                ),
            }
            for model in models
        ],
    }

# ============================================================
# VIDEO URL
# ============================================================

@app.post(
    "/api/inference"
)
def create_inference(
    request: InferenceRequest
):

    mission_code = (
        request.mission_id
    )

    mission = (
        resolve_mission(
            mission_code
        )
    )

    input_type = (
        request.input.type
        .strip()
        .upper()
    )

    if (
        input_type
        != "VIDEO_URL"
    ):

        raise HTTPException(
            status_code=400,

            detail=(
                "Only VIDEO_URL "
                "is supported "
                "on this endpoint"
            )
        )

    source_url = str(
        request.input.url
    )

    job_id = str(
        uuid.uuid4()
    )

    # --------------------------------------------------------
    # DB
    # --------------------------------------------------------

    create_job(
        job_id=
            job_id,

        mission_id=
            mission[
                "mission_uuid"
            ],

        model_name=
            mission[
                "model_name"
            ],

        model_version=
            mission[
                "model_version"
            ],

        input_type=
            "VIDEO_URL",

        input_url=
            source_url
    )

    # --------------------------------------------------------
    # CELERY
    # --------------------------------------------------------

    process_video.delay(

        job_id,

        mission[
            "mission_uuid"
        ],

        mission[
            "model_name"
        ],

        mission[
            "model_version"
        ],

        mission[
            "model_path"
        ],

        source_url,

        mission[
            "conf"
        ],

        mission[
            "iou"
        ],

        5,

        8,

        request.
        generate_annotated_video,
    )

    return {

        "job_id":
            job_id,

        "mission": {

            "code":
                mission_code,

            "id":
                mission[
                    "mission_uuid"
                ],
        },

        "model": {

            "name":
                mission[
                    "model_name"
                ],

            "version":
                mission[
                    "model_version"
                ],
        },

        "input": {

            "type":
                "VIDEO_URL",

            "url":
                source_url,
        },

        "generate_annotated_video":
            request.
            generate_annotated_video,

        "status":
            "queued",
    }


# ============================================================
# VIDEO FILE
# ============================================================

@app.post(
    "/api/inference/video/file"
)
async def inference_video_file(

    mission_id:
        str = Form(...),

    generate_annotated_video:
        bool = Form(False),

    file:
        UploadFile = File(...),
):
    print(
        f"[API] generate_annotated_video="
        f"{generate_annotated_video!r}, "
        f"type={type(generate_annotated_video)}"
    )

    mission = (
        resolve_mission(
            mission_id
        )
    )

    suffix = (
        os.path.splitext(
            file.filename
            or ""
        )[1]
    )

    if not suffix:
        suffix = ".mp4"

    # Shared volume giữa
    # mission-api và worker
    upload_dir = (
        "/shared/"
        "mission_uploads"
    )

    os.makedirs(
        upload_dir,
        exist_ok=True
    )

    job_id = str(
        uuid.uuid4()
    )

    video_path = (
        os.path.join(
            upload_dir,

            f"{job_id}"
            f"{suffix}"
        )
    )

    try:

        with open(
            video_path,
            "wb"
        ) as output:

            shutil.copyfileobj(
                file.file,
                output
            )

    except Exception as e:

        raise HTTPException(
            status_code=500,

            detail=(
                f"Failed to save "
                f"video: {e}"
            )
        )

    # --------------------------------------------------------
    # DB
    # --------------------------------------------------------

    try:

        create_job(

            job_id=
                job_id,

            mission_id=
                mission[
                    "mission_uuid"
                ],

            model_name=
                mission[
                    "model_name"
                ],

            model_version=
                mission[
                    "model_version"
                ],

            input_type=
                "VIDEO_FILE",

            input_url=
                video_path,
        )

    except Exception:

        if os.path.exists(
            video_path
        ):
            os.remove(
                video_path
            )

        raise

    # --------------------------------------------------------
    # CELERY
    # --------------------------------------------------------

    process_video.delay(

        job_id,

        mission[
            "mission_uuid"
        ],

        mission[
            "model_name"
        ],

        mission[
            "model_version"
        ],

        mission[
            "model_path"
        ],

        video_path,

        mission[
            "conf"
        ],

        mission[
            "iou"
        ],

        5,

        8,

        generate_annotated_video,
    )

    return {

        "job_id":
            job_id,

        "mission": {

            "code":
                mission[
                    "mission_code"
                ],

            "id":
                mission[
                    "mission_uuid"
                ],
        },

        "model": {

            "name":
                mission[
                    "model_name"
                ],

            "version":
                mission[
                    "model_version"
                ],
        },

        "filename":
            file.filename,

        "generate_annotated_video":
            generate_annotated_video,

        "status":
            "queued",
    }


# ============================================================
# IMAGE FILE
# ============================================================

@app.post(
    "/api/inference/image/file"
)
async def inference_image_file(

    mission_id:
        str = Form(...),

    file:
        UploadFile = File(...),
):

    mission = (
        resolve_mission(
            mission_id
        )
    )

    try:

        content = (
            await file.read()
        )

        image = (
            Image.open(
                io.BytesIO(
                    content
                )
            )
            .convert("RGB")
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid image: "
                f"{e}"
            )
        )

    detections = (
        predict_image(
            mission,
            image
        )
    )

    return {

        "mission": {

            "code":
                mission[
                    "mission_code"
                ],

            "id":
                mission[
                    "mission_uuid"
                ],
        },

        "model": {

            "name":
                mission[
                    "model_name"
                ],

            "version":
                mission[
                    "model_version"
                ],
        },

        "filename":
            file.filename,

        "detection_count":
            len(
                detections
            ),

        "detections":
            detections,
    }


# ============================================================
# IMAGE URL
# ============================================================

@app.post(
    "/api/inference/image/url"
)
def inference_image_url(
    request:
        ImageUrlRequest
):

    mission = (
        resolve_mission(
            request.mission_id
        )
    )

    try:

        response = (
            requests.get(
                str(
                    request.url
                ),

                timeout=30
            )
        )

        response.raise_for_status()

        image = (
            Image.open(
                io.BytesIO(
                    response.content
                )
            )
            .convert("RGB")
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,

            detail=(
                f"Cannot load "
                f"image URL: {e}"
            )
        )

    detections = (
        predict_image(
            mission,
            image
        )
    )

    return {

        "mission": {

            "code":
                mission[
                    "mission_code"
                ],

            "id":
                mission[
                    "mission_uuid"
                ],
        },

        "model": {

            "name":
                mission[
                    "model_name"
                ],

            "version":
                mission[
                    "model_version"
                ],
        },

        "source":
            str(
                request.url
            ),

        "detection_count":
            len(
                detections
            ),

        "detections":
            detections,
    }


# ============================================================
# LAND COVER RANDOM FOREST - PATH, PUBLIC URL OR UPLOAD
# ============================================================

def queue_landcover(mission, mission_id, relative_path, input_type, input_url=None):
    job_id = str(uuid.uuid4())
    source_url = input_url or build_public_geotiff_url(relative_path)
    create_job(
        job_id=job_id, mission_id=mission["mission_uuid"],
        model_name=mission["model_name"], model_version=mission["model_version"],
        input_type=input_type, input_url=source_url,
    )
    try:
        process_landcover_geotiff.apply_async(kwargs={
            "job_id": job_id, "mission_id": mission["mission_uuid"],
            "model_name": mission["model_name"], "model_version": mission["model_version"],
            "model_path": mission["model_path"], "relative_path": relative_path,
            "input_url": input_url, "input_type": input_type,
        }, queue="landcover")
    except Exception as error:
        from .database import mark_failed
        mark_failed(job_id, error)
        raise HTTPException(status_code=503, detail="Không thể đưa job vào hàng đợi") from error
    return {
        "job_id": job_id, "status": "queued", "input_type": input_type,
        "mission_id": mission_id,
        "source": {"relative_path": relative_path, "url": source_url},
        "model": {"name": mission["model_name"], "version": mission["model_version"]},
        "status_url": f"/api/inference/jobs/{job_id}",
        "result_url": f"/api/inference/jobs/{job_id}/result",
    }


@app.post("/api/inference/landcover/geotiff", status_code=202)
def inference_landcover_geotiff(request: LandcoverGeotiffRequest):
    mission = resolve_mission(request.mission_id)
    if request.input_url is not None:
        relative_path = f"input/downloads/{uuid.uuid4()}.tif"
        return queue_landcover(mission, request.mission_id, relative_path,
                              "LANDCOVER_GEOTIFF_URL", str(request.input_url))
    try:
        resolve_geotiff_source(request.relative_path, must_exist=True)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return queue_landcover(mission, request.mission_id, request.relative_path,
                          "LANDCOVER_GEOTIFF_PATH")


@app.post("/api/inference/landcover/geotiff/upload", status_code=202)
def inference_landcover_geotiff_upload(
    mission_id: str = Form(..., min_length=1),
    file: UploadFile = File(...),
):
    destination = None
    try:
        mission = resolve_mission(mission_id)
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".tif", ".tiff"}:
            raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .tif hoặc .tiff")
        relative_path = f"input/uploads/{uuid.uuid4()}{suffix}"
        destination = resolve_geotiff_source(relative_path, must_exist=False)
        save_chunks(iter(lambda: file.file.read(1024 * 1024), b""), destination)
        return queue_landcover(mission, mission_id, relative_path, "LANDCOVER_GEOTIFF_UPLOAD")
    except InputTooLarge as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise
    finally:
        file.file.close()


# ============================================================
# JOB STATUS
# ============================================================

@app.get(
    "/api/inference/jobs/{job_id}"
)
def job_status(
    job_id: str
):

    job = get_job(
        job_id
    )

    if not job:

        raise HTTPException(
            status_code=404,

            detail=(
                "Inference job "
                "not found"
            )
        )

    # Chỉ metadata,
    # không trả result.json dài
    return dict(
        job
    )


# ============================================================
# JOB RESULT FROM MINIO
# ============================================================

@app.get(
    "/api/inference/jobs/{job_id}/result"
)
def get_inference_result(
    job_id: str,

    offset: int = Query(
        default=0,
        ge=0
    ),

    limit: int = Query(
        default=20,
        ge=1,
        le=500
    ),

    model_name: str | None = Query(
        default=None
    ),

    mission_code: str | None = Query(
        default=None
    ),

    filename: str | None = Query(
        default=None
    ),

    class_name: str | None = Query(
        default=None
    ),

    min_confidence: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0
    ),
):
    # ========================================================
    # GET JOB
    # ========================================================

    job = get_job(
        job_id
    )

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Inference job not found"
        )

    job = dict(
        job
    )

    # ========================================================
    # JOB NOT COMPLETED
    # ========================================================

    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail={
                "message":
                    "Inference job is not completed",

                "status":
                    job.get("status"),

                "progress":
                    job.get("progress"),
            }
        )

    # ========================================================
    # READ RESULT.JSON FROM MINIO
    # ========================================================

    object_name = (
        f"inference/"
        f"{job_id}/"
        f"result.json"
    )

    try:
        result_data = get_json_from_minio(
            object_name
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Cannot read result "
                f"from MinIO: {e}"
            )
        )

    # ========================================================
    # INPUT TYPE
    # ========================================================

    input_type = result_data.get(
        "input_type"
    )

    # ========================================================
    # LAND COVER GEOTIFF
    # ========================================================
    if input_type in {
        "LANDCOVER_GEOTIFF_PATH",
        "LANDCOVER_GEOTIFF_URL",
        "LANDCOVER_GEOTIFF_UPLOAD",
    }:
        # Kết quả land-cover đã là summary nhỏ, không cần flatten/pagination
        # như kết quả detection theo frame của YOLO.
        return result_data

    # ========================================================
    # MULTI VIDEO URL
    # ========================================================
    if input_type in (
            "VIDEO_URLS_MULTI",
            "VIDEO_MINIO_MULTI",
    ):

        videos = result_data.get(
            "videos",
            []
        )

        items = []

        # ========================================================
        # FLATTEN:
        #
        # video × model × frame
        # ========================================================

        for video in videos:

            if (
                    video.get(
                        "status"
                    )
                    != "completed"
            ):
                continue

            video_index = (
                video.get(
                    "index"
                )
            )

            video_url = video.get(
                "url"
            )

            bucket = video.get(
                "bucket"
            )

            object_key = video.get(
                "object_key"
            )

            for model in video.get(
                    "models",
                    []
            ):

                current_model_name = (
                    model.get(
                        "model_name"
                    )
                )

                current_model_version = (
                    model.get(
                        "model_version"
                    )
                )

                current_mission_code = (
                    model.get(
                        "mission_code"
                    )
                )

                current_mission_id = (
                    model.get(
                        "mission_id"
                    )
                )

                if (
                        model_name is not None
                        and
                        current_model_name
                        != model_name
                ):
                    continue

                if (
                        mission_code is not None
                        and
                        current_mission_code
                        != mission_code
                ):
                    continue

                annotated_video_url = (
                    model.get(
                        "annotated_video_url"
                    )
                )

                for frame in model.get(
                        "frames",
                        []
                ):

                    detections = (
                        frame.get(
                            "detections",
                            []
                        )
                    )

                    filtered = []

                    for detection in detections:

                        if (
                                class_name
                                is not None
                                and
                                detection.get(
                                    "class_name"
                                )
                                != class_name
                        ):
                            continue

                        if (
                                min_confidence
                                is not None
                                and
                                float(
                                    detection.get(
                                        "confidence",
                                        0
                                    )
                                )
                                < min_confidence
                        ):
                            continue

                        filtered.append(
                            detection
                        )

                    if (
                            (
                                    class_name
                                    is not None
                                    or
                                    min_confidence
                                    is not None
                            )
                            and
                            not filtered
                    ):
                        continue

                    items.append(
                        {
                            "video_index":
                                video_index,

                            "video_url":
                                video_url,

                            "mission_id":
                                current_mission_id,

                            "mission_code":
                                current_mission_code,

                            "model_name":
                                current_model_name,

                            "model_version":
                                current_model_version,

                            "annotated_video_url":
                                annotated_video_url,

                            "frame":
                                frame.get(
                                    "frame"
                                ),

                            "time_seconds":
                                frame.get(
                                    "time_seconds"
                                ),

                            "detection_count":
                                len(
                                    filtered
                                ),

                            "detections":
                                filtered,
                        }
                    )

        # ========================================================
        # PAGINATION
        # ========================================================

        total = len(
            items
        )

        page = items[
               offset:
               offset + limit
               ]

        returned_detection_count = sum(
            item[
                "detection_count"
            ]
            for item
            in page
        )

        has_more = (
                offset
                + len(page)
                < total
        )

        return {

            "job_id":
                job_id,

            "status":
                result_data.get(
                    "status"
                ),

            "input_type":
                "VIDEO_URLS_MULTI",

            "requested_video_count":
                result_data.get(
                    "requested_video_count",
                    0
                ),

            "video_count":
                result_data.get(
                    "video_count",
                    0
                ),

            "failed_video_count":
                result_data.get(
                    "failed_video_count",
                    0
                ),

            "model_count":
                result_data.get(
                    "model_count",
                    0
                ),

            "generate_annotated_videos":
                result_data.get(
                    "generate_annotated_videos",
                    False
                ),

            "processed_frames":
                result_data.get(
                    "processed_frames",
                    0
                ),

            "total_detection_count":
                result_data.get(
                    "detection_count",
                    0
                ),

            "returned_detection_count":
                returned_detection_count,

            "errors":
                result_data.get(
                    "errors",
                    []
                ),

            "filters": {
                "model_name":
                    model_name,

                "mission_code":
                    mission_code,

                "class_name":
                    class_name,

                "min_confidence":
                    min_confidence,
            },

            "pagination": {
                "offset":
                    offset,

                "limit":
                    limit,

                "total":
                    total,

                "returned":
                    len(
                        page
                    ),

                "has_more":
                    has_more,

                "next_offset":
                    (
                        offset
                        + len(page)
                        if has_more
                        else None
                    ),
            },

            "results":
                page,
        }

    # ========================================================
    # MULTI IMAGE
    #
    # support:
    # IMAGE_FILES_MULTI
    # IMAGE_URLS_MULTI
    # ========================================================

    if input_type in (
        "IMAGE_FILES_MULTI",
        "IMAGE_URLS_MULTI",
    ):

        models = result_data.get(
            "models",
            []
        )

        # Merged annotated images are stored once per input image
        # at result_data["images"], not inside each model result.
        merged_images = result_data.get(
            "images",
            []
        )

        merged_image_by_index = {
            merged_image.get("index"): merged_image
            for merged_image in merged_images
            if merged_image.get("index") is not None
        }

        items = []

        # ====================================================
        # FLATTEN:
        #
        # model x image => one result item
        # ====================================================

        for model in models:

            current_model_name = (
                model.get(
                    "model_name"
                )
            )

            current_model_version = (
                model.get(
                    "model_version"
                )
            )

            current_mission_code = (
                model.get(
                    "mission_code"
                )
            )

            current_mission_id = (
                model.get(
                    "mission_id"
                )
            )

            # ------------------------------------------------
            # MODEL FILTER
            # ------------------------------------------------

            if (
                model_name is not None
                and
                current_model_name
                != model_name
            ):
                continue

            # ------------------------------------------------
            # MISSION FILTER
            # ------------------------------------------------

            if (
                mission_code is not None
                and
                current_mission_code
                != mission_code
            ):
                continue

            # ------------------------------------------------
            # IMAGES
            # ------------------------------------------------

            for image in model.get(
                "images",
                []
            ):

                current_filename = (
                    image.get(
                        "filename"
                    )
                )

                current_url = (
                    image.get(
                        "url"
                    )
                )

                # --------------------------------------------
                # FILENAME FILTER
                # --------------------------------------------

                if (
                    filename is not None
                    and
                    current_filename
                    != filename
                ):
                    continue

                detections = (
                    image.get(
                        "detections",
                        []
                    )
                )

                filtered_detections = []

                # --------------------------------------------
                # DETECTION FILTERS
                # --------------------------------------------

                for detection in detections:

                    if (
                        class_name
                        is not None
                        and
                        detection.get(
                            "class_name"
                        )
                        != class_name
                    ):
                        continue

                    if (
                        min_confidence
                        is not None
                    ):
                        confidence = float(
                            detection.get(
                                "confidence",
                                0
                            )
                        )

                        if (
                            confidence
                            < min_confidence
                        ):
                            continue

                    filtered_detections.append(
                        detection
                    )

                # Nếu user filter detection,
                # và ảnh không còn detection nào
                # => bỏ khỏi response
                if (
                    (
                        class_name
                        is not None
                        or
                        min_confidence
                        is not None
                    )
                    and
                    not filtered_detections
                ):
                    continue

                # --------------------------------------------
                # MERGED ANNOTATED IMAGE
                # --------------------------------------------

                image_index = image.get(
                    "index"
                )

                merged_image = (
                    merged_image_by_index.get(
                        image_index,
                        {}
                    )
                )

                merged_annotated_image_url = (
                    merged_image.get(
                        "merged_annotated_image_url"
                    )
                )

                # --------------------------------------------
                # RESULT ITEM
                # --------------------------------------------

                item = {

                    "mission_id":
                        current_mission_id,

                    "mission_code":
                        current_mission_code,

                    "model_name":
                        current_model_name,

                    "model_version":
                        current_model_version,

                    "filename":
                        current_filename,

                    "width":
                        image.get(
                            "width"
                        ),

                    "height":
                        image.get(
                            "height"
                        ),

                    # Backward compatibility: client cũ vẫn đọc
                    # annotated_image_url; cả hai field trỏ tới
                    # cùng một merged annotated image.
                    "annotated_image_url":
                        merged_annotated_image_url,

                    "merged_annotated_image_url":
                        merged_annotated_image_url,

                    "detection_count":
                        len(
                            filtered_detections
                        ),

                    "detections":
                        filtered_detections,
                }

                # IMAGE_URLS_MULTI có original URL
                if (
                    input_type
                    == "IMAGE_URLS_MULTI"
                ):
                    item[
                        "index"
                    ] = image.get(
                        "index"
                    )

                    item[
                        "url"
                    ] = current_url

                items.append(
                    item
                )

        # ====================================================
        # PAGINATION
        # ====================================================

        total = len(
            items
        )

        page = items[
            offset:
            offset + limit
        ]

        returned_detection_count = (
            sum(
                item.get(
                    "detection_count",
                    0
                )
                for item in page
            )
        )

        has_more = (
            offset
            + len(page)
            < total
        )

        next_offset = (
            offset + len(page)
            if has_more
            else None
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        response = {
            "job_id":
                job_id,

            "status":
                result_data.get(
                    "status"
                ),

            "input_type":
                input_type,

            "image_count":
                result_data.get(
                    "image_count",
                    0
                ),

            "model_count":
                result_data.get(
                    "model_count",
                    0
                ),

            "total_inference_units":
                result_data.get(
                    "total_inference_units",
                    0
                ),

            "generate_annotated_images":
                result_data.get(
                    "generate_annotated_images",
                    False
                ),

            "total_detection_count":
                result_data.get(
                    "detection_count",
                    0
                ),

            "returned_detection_count":
                returned_detection_count,

            "filters": {
                "model_name":
                    model_name,

                "mission_code":
                    mission_code,

                "filename":
                    filename,

                "class_name":
                    class_name,

                "min_confidence":
                    min_confidence,
            },

            "pagination": {
                "offset":
                    offset,

                "limit":
                    limit,

                "total":
                    total,

                "returned":
                    len(page),

                "has_more":
                    has_more,

                "next_offset":
                    next_offset,
            },

            "images":
                merged_images,

            "results":
                page,
        }

        # ====================================================
        # URL-SPECIFIC METADATA
        # ====================================================

        if (
            input_type
            == "IMAGE_URLS_MULTI"
        ):
            response[
                "requested_image_count"
            ] = result_data.get(
                "requested_image_count",
                result_data.get(
                    "image_count",
                    0
                )
            )

            response[
                "failed_image_count"
            ] = result_data.get(
                "failed_image_count",
                0
            )

            response[
                "download_errors"
            ] = result_data.get(
                "download_errors",
                []
            )

        return response

    # ==========================================
    # RTSP STREAM
    # ==========================================

    if input_type == "RTSP_STREAM":

        items = []

        models = result_data.get(
            "models",
            []
        )

        for model in models:

            current_model_name = (
                model.get(
                    "model_name"
                )
            )

            current_model_version = (
                model.get(
                    "model_version"
                )
            )

            current_mission_code = (
                model.get(
                    "mission_code"
                )
            )

            current_mission_id = (
                model.get(
                    "mission_id"
                )
            )

            # model filter
            if (
                    model_name is not None
                    and
                    current_model_name
                    != model_name
            ):
                continue

            # mission filter
            if (
                    mission_code is not None
                    and
                    current_mission_code
                    != mission_code
            ):
                continue

            annotated_video_url = (
                model.get(
                    "annotated_video_url"
                )
            )

            for frame in model.get(
                    "frames",
                    []
            ):

                detections = frame.get(
                    "detections",
                    []
                )

                filtered_detections = []

                for detection in detections:

                    if (
                            class_name is not None
                            and
                            detection.get(
                                "class_name"
                            )
                            != class_name
                    ):
                        continue

                    if (
                            min_confidence
                            is not None
                    ):
                        confidence = float(
                            detection.get(
                                "confidence",
                                0
                            )
                        )

                        if (
                                confidence
                                < min_confidence
                        ):
                            continue

                    filtered_detections.append(
                        detection
                    )

                if (
                        (
                                class_name is not None
                                or
                                min_confidence is not None
                        )
                        and
                        not filtered_detections
                ):
                    continue

                items.append(
                    {
                        "mission_id":
                            current_mission_id,

                        "mission_code":
                            current_mission_code,

                        "model_name":
                            current_model_name,

                        "model_version":
                            current_model_version,

                        "annotated_video_url":
                            annotated_video_url,

                        "frame":
                            frame.get(
                                "frame"
                            ),

                        "time_seconds":
                            frame.get(
                                "time_seconds"
                            ),

                        "detection_count":
                            len(
                                filtered_detections
                            ),

                        "detections":
                            filtered_detections,
                    }
                )

        # pagination
        total = len(
            items
        )

        page = items[
               offset:
               offset + limit
               ]

        returned_detection_count = sum(
            item.get(
                "detection_count",
                0
            )
            for item in page
        )

        has_more = (
                offset
                + len(page)
                < total
        )

        next_offset = (
            offset
            + len(page)
            if has_more
            else None
        )

        return {
            "job_id":
                job_id,

            "status":
                result_data.get(
                    "status"
                ),

            "input_type":
                "RTSP_STREAM",

            "rtsp_url":
                result_data.get(
                    "rtsp_url"
                ),

            "duration_seconds":
                result_data.get(
                    "duration_seconds"
                ),

            "frame_skip":
                result_data.get(
                    "frame_skip"
                ),

            "model_count":
                result_data.get(
                    "model_count",
                    0
                ),

            "generate_annotated_video":
                result_data.get(
                    "generate_annotated_video",
                    False
                ),

            "processed_frames":
                result_data.get(
                    "processed_frames",
                    0
                ),

            "total_detection_count":
                result_data.get(
                    "detection_count",
                    0
                ),

            "returned_detection_count":
                returned_detection_count,

            "filters": {
                "model_name":
                    model_name,

                "mission_code":
                    mission_code,

                "class_name":
                    class_name,

                "min_confidence":
                    min_confidence,
            },

            "pagination": {
                "offset":
                    offset,

                "limit":
                    limit,

                "total":
                    total,

                "returned":
                    len(page),

                "has_more":
                    has_more,

                "next_offset":
                    next_offset,
            },

            "results":
                page,
        }

    # ========================================================
    # VIDEO RESULT
    # ========================================================

    frames = result_data.get(
        "frames",
        []
    )

    filtered_frames = []

    for frame in frames:

        detections = frame.get(
            "detections",
            []
        )

        filtered = []

        for detection in detections:

            if (
                class_name is not None
                and
                detection.get(
                    "class_name"
                )
                != class_name
            ):
                continue

            if (
                min_confidence
                is not None
                and
                float(
                    detection.get(
                        "confidence",
                        0
                    )
                )
                < min_confidence
            ):
                continue

            filtered.append(
                detection
            )

        if (
            (
                class_name is None
                and
                min_confidence is None
            )
            or
            filtered
        ):
            filtered_frames.append(
                {
                    **frame,
                    "detections":
                        filtered,
                }
            )

    # ========================================================
    # VIDEO PAGINATION
    # ========================================================

    total = len(
        filtered_frames
    )

    page = filtered_frames[
        offset:
        offset + limit
    ]

    returned_detection_count = sum(
        len(
            frame.get(
                "detections",
                []
            )
        )
        for frame in page
    )

    has_more = (
        offset
        + len(page)
        < total
    )

    return {
        "job_id":
            job_id,

        "mission_id":
            result_data.get(
                "mission_id"
            ),

        "input_type":
            input_type,

        "model":
            result_data.get(
                "model"
            ),

        "video":
            result_data.get(
                "video"
            ),

        "annotated_video":
            result_data.get(
                "annotated_video"
            ),

        "summary": {
            "processed_frames":
                result_data.get(
                    "processed_frames"
                ),

            "total_detection_count":
                result_data.get(
                    "detection_count",
                    0
                ),

            "returned_detection_count":
                returned_detection_count,
        },

        "filters": {
            "class_name":
                class_name,

            "min_confidence":
                min_confidence,
        },

        "pagination": {
            "offset":
                offset,

            "limit":
                limit,

            "total":
                total,

            "returned":
                len(page),

            "has_more":
                has_more,

            "next_offset":
                (
                    offset
                    + len(page)
                    if has_more
                    else None
                ),
        },

        "frames":
            page,
    }

# NOTE:
# Legacy /api/inference/video/file/multi endpoint was removed because it
# referenced undefined create_model_job/process_video_multi symbols and
# had no safe parent/sub-job completion aggregation.
# Keep /api/inference/video/file for single-file inference and
# /api/inference/videos/url/multi for multi-video URL inference until a
# dedicated multi-file video worker task is implemented.

@app.post(
    "/api/inference/images/multi"
)
async def inference_images_multi(

    mission_ids:
        str = Form(...),

    generate_annotated_images:
        bool = Form(False),

    files:
        list[UploadFile] = File(...),
):

    # ========================================================
    # PARSE MISSION IDS
    # ========================================================

    try:

        mission_codes = (
            json.loads(
                mission_ids
            )
        )

    except Exception:

        raise HTTPException(
            status_code=400,

            detail=(
                "mission_ids must be "
                "JSON array. Example: "
                '["mission-fire",'
                '"mission-rust"]'
            )
        )

    if not isinstance(
        mission_codes,
        list
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "mission_ids "
                "must be array"
            )
        )

    mission_codes = [
        str(code).strip()
        for code in mission_codes
        if str(code).strip()
    ]

    # remove duplicates
    mission_codes = list(
        dict.fromkeys(
            mission_codes
        )
    )

    if not mission_codes:

        raise HTTPException(
            status_code=400,
            detail=(
                "mission_ids "
                "cannot be empty"
            )
        )

    if not files:

        raise HTTPException(
            status_code=400,
            detail=(
                "No image files "
                "uploaded"
            )
        )

    # ========================================================
    # GET ACTIVE MODELS
    # ========================================================

    models = (
        get_active_models_by_missions(
            mission_codes
        )
    )

    found_codes = {
        row[
            "mission_code"
        ]
        for row in models
    }

    missing = [
        code
        for code in mission_codes
        if code not in found_codes
    ]

    if missing:

        raise HTTPException(
            status_code=404,

            detail={
                "message":
                    "Some missions "
                    "have no active model",

                "missing":
                    missing,
            }
        )

    # ========================================================
    # JOB ID
    # ========================================================

    job_id = str(
        uuid.uuid4()
    )

    # ========================================================
    # UPLOAD DIRECTORY
    # ========================================================

    upload_dir = os.path.join(
        "/shared/"
        "mission_uploads",
        job_id
    )

    os.makedirs(
        upload_dir,
        exist_ok=True
    )

    uploaded_files = []

    # ========================================================
    # SAVE ALL IMAGES
    # ========================================================

    try:

        for index, file in enumerate(
            files
        ):

            original_name = (
                file.filename
                or
                f"image_{index}.jpg"
            )

            safe_name = (
                os.path.basename(
                    original_name
                )
            )

            # tránh 2 file trùng tên
            safe_name = (
                f"{index:04d}_"
                f"{safe_name}"
            )

            file_path = (
                os.path.join(
                    upload_dir,
                    safe_name
                )
            )

            with open(
                file_path,
                "wb"
            ) as output:

                shutil.copyfileobj(
                    file.file,
                    output
                )

            uploaded_files.append(
                {
                    "filename":
                        safe_name,

                    "original_filename":
                        original_name,

                    "path":
                        file_path,
                }
            )

    except Exception as e:

        # cleanup nếu upload lỗi
        for item in uploaded_files:

            try:
                os.remove(
                    item[
                        "path"
                    ]
                )

            except Exception:
                pass

        raise HTTPException(
            status_code=500,

            detail=(
                f"Cannot save "
                f"uploaded images: {e}"
            )
        )

    # ========================================================
    # CREATE PARENT JOB
    # ========================================================

    try:

        create_job(
            job_id=
                job_id,

            mission_id=
                None,

            model_name=
                "MULTI_MODEL",

            model_version=
                "v1",

            input_type=
                "IMAGE_FILES_MULTI",

            input_url=
                upload_dir,
        )

    except Exception as e:

        for item in uploaded_files:

            try:
                os.remove(
                    item[
                        "path"
                    ]
                )

            except Exception:
                pass

        raise HTTPException(
            status_code=500,

            detail=(
                f"Cannot create "
                f"inference job: {e}"
            )
        )

    # ========================================================
    # CELERY PAYLOAD
    # ========================================================

    model_payload = [

        {
            "mission_id":
                str(
                    model[
                        "mission_id"
                    ]
                ),

            "mission_code":
                model[
                    "mission_code"
                ],

            "model_name":
                model[
                    "model_name"
                ],

            "model_version":
                model[
                    "model_version"
                ],

            "model_path":
                model[
                    "model_path"
                ],

            "confidence_threshold":
                float(
                    model[
                        "confidence_threshold"
                    ]
                ),

            "iou_threshold":
                float(
                    model[
                        "iou_threshold"
                    ]
                ),
        }

        for model in models
    ]

    # ========================================================
    # QUEUE
    # ========================================================

    try:

        process_images_multi.apply_async(
            kwargs={
                "job_id":
                    job_id,

                "models":
                    model_payload,

                "images":
                    uploaded_files,

                "generate_annotated_images":
                    generate_annotated_images,

                "image_batch_size":
                    8,
            }
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,

            detail=(
                f"Cannot enqueue "
                f"inference task: {e}"
            )
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "job_id":
            job_id,

        "status":
            "queued",

        "input_type":
            "IMAGE_FILES_MULTI",

        "image_count":
            len(
                uploaded_files
            ),

        "model_count":
            len(
                models
            ),

        "total_inference_units":
            (
                len(
                    uploaded_files
                )
                *
                len(
                    models
                )
            ),

        "generate_annotated_images":
            generate_annotated_images,

        "models": [

            {
                "mission_code":
                    model[
                        "mission_code"
                    ],

                "model_name":
                    model[
                        "model_name"
                    ],

                "model_version":
                    model[
                        "model_version"
                    ],
            }

            for model in models
        ],

        "files": [

            {
                "filename":
                    item[
                        "filename"
                    ],

                "original_filename":
                    item[
                        "original_filename"
                    ],
            }

            for item
            in uploaded_files
        ],
    }

@app.post(
    "/api/inference/images/url/multi"
)
def inference_image_urls_multi(
    request: MultiImageUrlRequest
):
    # ========================================================
    # MISSIONS
    # ========================================================

    mission_codes = [
        code.strip()
        for code in request.mission_ids
        if code.strip()
    ]

    # remove duplicate
    mission_codes = list(
        dict.fromkeys(
            mission_codes
        )
    )

    if not mission_codes:
        raise HTTPException(
            status_code=400,
            detail="mission_ids cannot be empty"
        )

    # ========================================================
    # URLS
    # ========================================================

    image_urls = [
        str(url)
        for url in request.image_urls
    ]

    # remove duplicate URL
    image_urls = list(
        dict.fromkeys(
            image_urls
        )
    )

    if not image_urls:
        raise HTTPException(
            status_code=400,
            detail="image_urls cannot be empty"
        )

    if len(image_urls) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 image URLs per request"
        )

    # ========================================================
    # GET MODELS
    # ========================================================

    models = (
        get_active_models_by_missions(
            mission_codes
        )
    )

    found_codes = {
        model["mission_code"]
        for model in models
    }

    missing = [
        code
        for code in mission_codes
        if code not in found_codes
    ]

    if missing:
        raise HTTPException(
            status_code=404,
            detail={
                "message":
                    "Some missions have no active model",

                "missing":
                    missing,
            }
        )

    # ========================================================
    # JOB
    # ========================================================

    job_id = str(
        uuid.uuid4()
    )

    # Có thể lưu JSON URL vào input_url TEXT
    input_url = json.dumps(
        image_urls,
        ensure_ascii=False
    )

    create_job(
        job_id=job_id,

        # multi-model parent job
        mission_id=None,

        model_name="MULTI_MODEL",

        model_version="v1",

        input_type="IMAGE_URLS_MULTI",

        input_url=input_url,
    )

    # ========================================================
    # MODEL PAYLOAD
    # ========================================================

    model_payload = [
        {
            "mission_id":
                str(
                    model[
                        "mission_id"
                    ]
                ),

            "mission_code":
                model[
                    "mission_code"
                ],

            "model_name":
                model[
                    "model_name"
                ],

            "model_version":
                model[
                    "model_version"
                ],

            "model_path":
                model[
                    "model_path"
                ],

            "confidence_threshold":
                float(
                    model[
                        "confidence_threshold"
                    ]
                ),

            "iou_threshold":
                float(
                    model[
                        "iou_threshold"
                    ]
                ),
        }

        for model in models
    ]

    # ========================================================
    # CELERY
    # ========================================================

    process_image_urls_multi.apply_async(
        kwargs={
            "job_id":
                job_id,

            "models":
                model_payload,

            "image_urls":
                image_urls,

            "generate_annotated_images":
                request.generate_annotated_images,

            "image_batch_size":
                8,
        }
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "job_id":
            job_id,

        "status":
            "queued",

        "input_type":
            "IMAGE_URLS_MULTI",

        "image_count":
            len(image_urls),

        "model_count":
            len(models),

        "total_inference_units":
            (
                len(image_urls)
                * len(models)
            ),

        "generate_annotated_images":
            request.generate_annotated_images,

        "models": [
            {
                "mission_code":
                    model[
                        "mission_code"
                    ],

                "model_name":
                    model[
                        "model_name"
                    ],

                "model_version":
                    model[
                        "model_version"
                    ],
            }

            for model in models
        ],

        "image_urls":
            image_urls,
    }

@app.post(
    "/api/inference/videos/url/multi"
)
def inference_videos_url_multi(
    request: MultiVideoUrlRequest
):

    # ========================================================
    # MISSIONS
    # ========================================================

    mission_codes = [
        code.strip()
        for code
        in request.mission_ids
        if code.strip()
    ]

    mission_codes = list(
        dict.fromkeys(
            mission_codes
        )
    )

    if not mission_codes:
        raise HTTPException(
            status_code=400,
            detail="mission_ids cannot be empty"
        )

    # ========================================================
    # VIDEO URLS
    # ========================================================

    video_urls = [
        str(url)
        for url
        in request.video_urls
    ]

    video_urls = list(
        dict.fromkeys(
            video_urls
        )
    )

    if not video_urls:
        raise HTTPException(
            status_code=400,
            detail="video_urls cannot be empty"
        )

    if len(video_urls) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 videos per request"
        )

    # ========================================================
    # MODELS
    # ========================================================

    models = (
        get_active_models_by_missions(
            mission_codes
        )
    )

    found = {
        model["mission_code"]
        for model in models
    }

    missing = [
        code
        for code in mission_codes
        if code not in found
    ]

    if missing:
        raise HTTPException(
            status_code=404,
            detail={
                "message":
                    "Some missions have no active model",

                "missing":
                    missing,
            }
        )

    # ========================================================
    # JOB
    # ========================================================

    job_id = str(
        uuid.uuid4()
    )

    input_url = json.dumps(
        video_urls,
        ensure_ascii=False
    )

    create_job(
        job_id=job_id,

        mission_id=None,

        model_name="MULTI_MODEL",

        model_version="v1",

        input_type="VIDEO_URLS_MULTI",

        input_url=input_url,
    )

    # ========================================================
    # MODEL PAYLOAD
    # ========================================================

    model_payload = [

        {
            "mission_id":
                str(
                    model[
                        "mission_id"
                    ]
                ),

            "mission_code":
                model[
                    "mission_code"
                ],

            "model_name":
                model[
                    "model_name"
                ],

            "model_version":
                model[
                    "model_version"
                ],

            "model_path":
                model[
                    "model_path"
                ],

            "confidence_threshold":
                float(
                    model[
                        "confidence_threshold"
                    ]
                ),

            "iou_threshold":
                float(
                    model[
                        "iou_threshold"
                    ]
                ),
        }

        for model in models
    ]

    # ========================================================
    # CELERY
    # ========================================================

    process_videos_url_multi.apply_async(
        kwargs={
            "job_id":
                job_id,

            "models":
                model_payload,

            "video_urls":
                video_urls,

            "generate_annotated_videos":
                request.generate_annotated_videos,

            "frame_skip":
                request.frame_skip,

            "batch_size":
                request.batch_size,
        }
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "job_id":
            job_id,

        "status":
            "queued",

        "input_type":
            "VIDEO_URLS_MULTI",

        "video_count":
            len(video_urls),

        "model_count":
            len(models),

        "total_inference_units":
            (
                len(video_urls)
                * len(models)
            ),

        "generate_annotated_videos":
            request.generate_annotated_videos,

        "frame_skip":
            request.frame_skip,

        "batch_size":
            request.batch_size,

        "models": [
            {
                "mission_code":
                    model["mission_code"],

                "model_name":
                    model["model_name"],

                "model_version":
                    model["model_version"],
            }

            for model in models
        ],

        "video_urls":
            video_urls,
    }


@app.post(
    "/api/inference/videos/minio/multi"
)
def inference_videos_minio_multi(
    request: MultiMinioVideoRequest
):

    # ========================================================
    # MISSIONS
    # ========================================================

    mission_codes = [
        code.strip()
        for code
        in request.mission_ids
        if code.strip()
    ]

    mission_codes = list(
        dict.fromkeys(
            mission_codes
        )
    )

    if not mission_codes:

        raise HTTPException(
            status_code=400,
            detail=(
                "mission_ids cannot be empty"
            )
        )

    # ========================================================
    # VIDEOS
    # ========================================================

    videos = [
        {
            "bucket":
                item.bucket.strip(),

            "object_key":
                item.object_key.strip(),
        }

        for item
        in request.videos
    ]

    if not videos:

        raise HTTPException(
            status_code=400,
            detail="videos cannot be empty"
        )

    if len(videos) > 20:

        raise HTTPException(
            status_code=400,
            detail=(
                "Maximum 20 MinIO "
                "videos per request"
            )
        )

    # ========================================================
    # MODELS
    # ========================================================

    models = (
        get_active_models_by_missions(
            mission_codes
        )
    )

    found = {
        model[
            "mission_code"
        ]
        for model
        in models
    }

    missing = [
        code
        for code
        in mission_codes
        if code not in found
    ]

    if missing:

        raise HTTPException(
            status_code=404,

            detail={
                "message":
                    "Some missions "
                    "have no active model",

                "missing":
                    missing,
            }
        )

    # ========================================================
    # JOB
    # ========================================================

    job_id = str(
        uuid.uuid4()
    )

    input_url = json.dumps(
        videos,
        ensure_ascii=False
    )

    create_job(
        job_id=
            job_id,

        mission_id=
            None,

        model_name=
            "MULTI_MODEL",

        model_version=
            "v1",

        input_type=
            "VIDEO_MINIO_MULTI",

        input_url=
            input_url,
    )

    # ========================================================
    # MODEL PAYLOAD
    # ========================================================

    model_payload = [

        {
            "mission_id":
                str(
                    model[
                        "mission_id"
                    ]
                ),

            "mission_code":
                model[
                    "mission_code"
                ],

            "model_name":
                model[
                    "model_name"
                ],

            "model_version":
                model[
                    "model_version"
                ],

            "model_path":
                model[
                    "model_path"
                ],

            "confidence_threshold":
                float(
                    model[
                        "confidence_threshold"
                    ]
                ),

            "iou_threshold":
                float(
                    model[
                        "iou_threshold"
                    ]
                ),
        }

        for model
        in models
    ]

    # ========================================================
    # CELERY
    # ========================================================

    process_videos_minio_multi.apply_async(
        kwargs={
            "job_id":
                job_id,

            "models":
                model_payload,

            "videos":
                videos,

            "generate_annotated_videos":
                request.
                generate_annotated_videos,

            "frame_skip":
                request.frame_skip,

            "batch_size":
                request.batch_size,
        }
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "job_id":
            job_id,

        "status":
            "queued",

        "input_type":
            "VIDEO_MINIO_MULTI",

        "video_count":
            len(videos),

        "model_count":
            len(models),

        "total_inference_units":
            (
                len(videos)
                * len(models)
            ),

        "generate_annotated_videos":
            request.
            generate_annotated_videos,

        "frame_skip":
            request.frame_skip,

        "batch_size":
            request.batch_size,

        "models": [
            {
                "mission_code":
                    model[
                        "mission_code"
                    ],

                "model_name":
                    model[
                        "model_name"
                    ],

                "model_version":
                    model[
                        "model_version"
                    ],
            }

            for model
            in models
        ],

        "videos":
            videos,
    }


@app.post(
    "/api/inference/stream/rtsp"
)
def inference_rtsp(
    request: RtspInferenceRequest
):
    mission_codes = [
        code.strip()
        for code in request.mission_ids
        if code.strip()
    ]

    mission_codes = list(
        dict.fromkeys(
            mission_codes
        )
    )

    if not mission_codes:
        raise HTTPException(
            status_code=400,
            detail="mission_ids cannot be empty"
        )

    rtsp_url = request.rtsp_url.strip()

    if not rtsp_url.lower().startswith(
        "rtsp://"
    ):
        raise HTTPException(
            status_code=400,
            detail="rtsp_url must start with rtsp://"
        )

    models = (
        get_active_models_by_missions(
            mission_codes
        )
    )

    found = {
        model["mission_code"]
        for model in models
    }

    missing = [
        code
        for code in mission_codes
        if code not in found
    ]

    if missing:
        raise HTTPException(
            status_code=404,
            detail={
                "message":
                    "Some missions have no active model",

                "missing":
                    missing,
            }
        )

    job_id = str(
        uuid.uuid4()
    )

    create_job(
        job_id=job_id,
        mission_id=None,
        model_name="MULTI_MODEL",
        model_version="v1",
        input_type="RTSP_STREAM",
        input_url=rtsp_url,
    )

    model_payload = [
        {
            "mission_id":
                str(
                    model["mission_id"]
                ),

            "mission_code":
                model["mission_code"],

            "model_name":
                model["model_name"],

            "model_version":
                model["model_version"],

            "model_path":
                model["model_path"],

            "confidence_threshold":
                float(
                    model[
                        "confidence_threshold"
                    ]
                ),

            "iou_threshold":
                float(
                    model[
                        "iou_threshold"
                    ]
                ),
        }

        for model in models
    ]

    process_rtsp_stream.apply_async(
        kwargs={
            "job_id":
                job_id,

            "models":
                model_payload,

            "rtsp_url":
                rtsp_url,

            "duration_seconds":
                request.duration_seconds,

            "frame_skip":
                request.frame_skip,

            "batch_size":
                request.batch_size,

            "generate_annotated_video":
                request.generate_annotated_video,
        }
    )

    return {
        "job_id":
            job_id,

        "status":
            "queued",

        "input_type":
            "RTSP_STREAM",

        "rtsp_url":
            rtsp_url,

        "duration_seconds":
            request.duration_seconds,

        "model_count":
            len(models),

        "frame_skip":
            request.frame_skip,

        "batch_size":
            request.batch_size,

        "generate_annotated_video":
            request.generate_annotated_video,

        "models": [
            {
                "mission_code":
                    model["mission_code"],

                "model_name":
                    model["model_name"],

                "model_version":
                    model["model_version"],
            }

            for model in models
        ],
    }
