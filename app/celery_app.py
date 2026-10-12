from celery import Celery
from .config import settings


celery_app = Celery(
    "mission_ai",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,

    # QUAN TRỌNG
    include=[
        "app.tasks"
    ]
)


celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    task_track_started=True,

    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    broker_connection_retry_on_startup=True,

    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
)

celery_app.conf.imports = (
    "app.tasks",
)