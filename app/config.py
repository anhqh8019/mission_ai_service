import os


class Settings:
    DB_HOST = os.getenv("DB_HOST", "postgres")
    DB_PORT = int(os.getenv("DB_PORT", "5532"))
    DB_NAME = os.getenv("DB_NAME", "uavdb")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

    REDIS_URL = os.getenv(
        "REDIS_URL",
        "redis://redis:6479/0"
    )

    MQTT_HOST = os.getenv(
        "MQTT_HOST",
        "mqtt"
    )

    MQTT_PORT = int(
        os.getenv("MQTT_PORT", "1883")
    )

    MQTT_USERNAME = os.getenv(
        "MQTT_USERNAME"
    )

    MQTT_PASSWORD = os.getenv(
        "MQTT_PASSWORD"
    )

    MINIO_ENDPOINT = os.getenv(
        "MINIO_ENDPOINT",
        "http://minio:9500"
    )

    MINIO_PUBLIC_ENDPOINT = os.getenv(
        "MINIO_PUBLIC_ENDPOINT",
        "http://10.1.53.88:9211"
    )

    MINIO_ACCESS_KEY = os.getenv(
        "MINIO_ACCESS_KEY",
        "minioadmin"
    )

    MINIO_SECRET_KEY = os.getenv(
        "MINIO_SECRET_KEY",
        "minioadmin"
    )

    MINIO_BUCKET = os.getenv(
        "MINIO_BUCKET",
        "inference-results"
    )

    FILE_SERVER_IMAGE_DIR = os.getenv(
        "FILE_SERVER_IMAGE_DIR",
        "/file_server/images",
    )

    FILE_SERVER_VIDEO_DIR = os.getenv(
        "FILE_SERVER_VIDEO_DIR",
        "/file_server/videos",
    )

    FILE_SERVER_GEOTIFF_DIR = os.getenv(
        "FILE_SERVER_GEOTIFF_DIR",
        "/file_server/geotiff",
    )

    FILE_SERVER_PUBLIC_URL = os.getenv(
        "FILE_SERVER_PUBLIC_URL",
        "http://10.1.53.88:9080",
    )


settings = Settings()
