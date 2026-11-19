"""
Celery Application Instance for Poshan Production Pipeline.
Configured with Redis broker and result backend.
"""
import os
from celery import Celery

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")

DEFAULT_BROKER = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
DEFAULT_BACKEND = f"redis://{REDIS_HOST}:{REDIS_PORT}/1"

BROKER_URL = os.getenv("CELERY_BROKER_URL", DEFAULT_BROKER)
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", DEFAULT_BACKEND)

celery = Celery(
    "poshan_tasks",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["production.tasks"]
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    worker_send_task_events=True,  # Enable real-time event streaming for Flower dashboard
    task_send_sent_event=True,     # Track task queued timestamps and latency
    task_time_limit=180,  # 3 minutes maximum per screening task
    task_soft_time_limit=150,
    worker_prefetch_multiplier=1,  # Fair task distribution across CPU cores
    broker_connection_retry_on_startup=True,
    task_routes={
        "production.tasks.process_child_screening_task": {"queue": "poshan_screening"},
        "production.tasks.send_whatsapp_notification_task": {"queue": "poshan_notifications"}
    }
)

