from pydantic import BaseModel, HttpUrl


class InferenceInput(BaseModel):
    type: str
    url: HttpUrl


class InferenceRequest(BaseModel):
    mission_id: str

    input: InferenceInput

    # Mặc định KHÔNG tạo annotated video
    generate_annotated_video: bool = False

from pydantic import BaseModel, HttpUrl, Field


class MultiImageUrlRequest(BaseModel):
    mission_ids: list[str] = Field(
        ...,
        min_length=1
    )

    image_urls: list[HttpUrl] = Field(
        ...,
        min_length=1,
        max_length=100
    )

    generate_annotated_images: bool = False

from pydantic import BaseModel, HttpUrl, Field


class MultiVideoUrlRequest(BaseModel):

    mission_ids: list[str] = Field(
        ...,
        min_length=1
    )

    video_urls: list[HttpUrl] = Field(
        ...,
        min_length=1,
        max_length=20
    )

    generate_annotated_videos: bool = False

    frame_skip: int = Field(
        default=5,
        ge=1,
        le=100
    )

    batch_size: int = Field(
        default=8,
        ge=1,
        le=32
    )

from pydantic import BaseModel, Field


class MinioVideoObject(BaseModel):
    bucket: str = Field(
        ...,
        min_length=1
    )

    object_key: str = Field(
        ...,
        min_length=1
    )


class MultiMinioVideoRequest(BaseModel):

    mission_ids: list[str] = Field(
        ...,
        min_length=1
    )

    videos: list[MinioVideoObject] = Field(
        ...,
        min_length=1,
        max_length=20
    )

    generate_annotated_videos: bool = False

    frame_skip: int = Field(
        default=5,
        ge=1,
        le=100
    )

    batch_size: int = Field(
        default=8,
        ge=1,
        le=32
    )

from pydantic import BaseModel, Field


class RtspInferenceRequest(BaseModel):
    mission_ids: list[str] = Field(
        ...,
        min_length=1
    )

    rtsp_url: str = Field(
        ...,
        min_length=1
    )

    duration_seconds: int = Field(
        default=60,
        ge=5,
        le=3600
    )

    frame_skip: int = Field(
        default=5,
        ge=1,
        le=100
    )

    batch_size: int = Field(
        default=8,
        ge=1,
        le=32
    )

    generate_annotated_video: bool = False