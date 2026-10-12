import psycopg2
from psycopg2.extras import RealDictCursor

from .config import settings
from psycopg2.extras import RealDictCursor

def get_connection():

    return psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        database=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD
    )


def get_active_model_by_mission(
    mission_code: str
):

    conn = get_connection()

    try:

        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:

            cursor.execute(
                """
             SELECT
                    mm.id,
                    mm.mission_id,
                    mm.model_name,
                    mm.model_version,
                    mm.model_path,
                    mm.confidence_threshold,
                    mm.iou_threshold
                FROM mission_models mm
                JOIN missions m
                    ON m.id = mm.mission_id
                WHERE m.code = %s
                  AND mm.is_active = TRUE
                  AND m.status = 'ACTIVE'
                ORDER BY mm.created_at DESC
                LIMIT 1
                """,
                (mission_code,)
            )

            return cursor.fetchone()

    finally:
        conn.close()


def create_job(
    job_id,
    mission_id,
    model_name,
    model_version,
    input_type,
    input_url
):

    conn = get_connection()

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO inference_jobs (
                    id,
                    mission_id,
                    model_name,
                    model_version,
                    input_type,
                    input_url,
                    status
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    'queued'
                )
                """,
                (
                    job_id,
                    mission_id,
                    model_name,
                    model_version,
                    input_type,
                    input_url
                )
            )

        conn.commit()

    finally:
        conn.close()


def mark_running(
    job_id
):

    conn = get_connection()

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                UPDATE inference_jobs

                SET
                    status = 'running',
                    started_at = NOW()

                WHERE id = %s
                """,
                (job_id,)
            )

        conn.commit()

    finally:
        conn.close()


def update_progress(
    job_id,
    processed_frames,
    total_frames,
    detection_count,
    progress
):

    conn = get_connection()

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                UPDATE inference_jobs

                SET
                    processed_frames = %s,
                    total_frames = %s,
                    detection_count = %s,
                    progress = %s

                WHERE id = %s
                """,
                (
                    processed_frames,
                    total_frames,
                    detection_count,
                    progress,
                    job_id
                )
            )

        conn.commit()

    finally:
        conn.close()


def mark_completed(
    job_id,
    result_url,
    processed_frames,
    total_frames,
    detection_count
):

    conn = get_connection()

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                UPDATE inference_jobs

                SET
                    status = 'completed',
                    progress = 100,
                    result_url = %s,
                    processed_frames = %s,
                    total_frames = %s,
                    detection_count = %s,
                    completed_at = NOW()

                WHERE id = %s
                """,
                (
                    result_url,
                    processed_frames,
                    total_frames,
                    detection_count,
                    job_id
                )
            )

        conn.commit()

    finally:
        conn.close()


def mark_failed(
    job_id,
    error
):

    conn = get_connection()

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                UPDATE inference_jobs

                SET
                    status = 'failed',
                    error = %s,
                    completed_at = NOW()

                WHERE id = %s
                """,
                (
                    str(error),
                    job_id
                )
            )

        conn.commit()

    finally:
        conn.close()


def get_job(
    job_id
):

    conn = get_connection()

    try:

        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:

            cursor.execute(
                """
                SELECT *
                FROM inference_jobs
                WHERE id = %s
                """,
                (job_id,)
            )

            return cursor.fetchone()

    finally:
        conn.close()


def get_supported_models():
    conn = get_connection()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT
                    m.id AS mission_id,
                    m.code AS mission_code,
                    m.name AS mission_name,
                    mm.model_name,
                    mm.model_version,
                    mm.model_path,
                    mm.confidence_threshold,
                    mm.iou_threshold
                FROM missions m
                JOIN mission_models mm
                    ON mm.mission_id = m.id
                WHERE
                    m.status = 'ACTIVE'
                    AND mm.is_active = TRUE
                ORDER BY m.code, mm.model_version
            """)

            return cursor.fetchall()

    finally:
        conn.close()

def get_active_models_by_missions(
    mission_codes: list[str]
):
    conn = get_connection()

    try:
        cursor = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cursor.execute(
            """
            SELECT
                m.id AS mission_id,
                m.code AS mission_code,

                mm.model_name,
                mm.model_version,
                mm.model_path,

                mm.confidence_threshold,
                mm.iou_threshold

            FROM missions m

            JOIN mission_models mm
                ON mm.mission_id = m.id

            WHERE
                m.code = ANY(%s)
                AND m.status = 'ACTIVE'
                AND mm.is_active = TRUE

            ORDER BY m.code
            """,
            (
                mission_codes,
            )
        )

        return cursor.fetchall()

    finally:
        conn.close()

def get_active_models_by_missions(
    mission_codes: list[str]
):
    conn = get_connection()

    try:
        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cursor:

            cursor.execute(
                """
                SELECT
                    m.id AS mission_id,
                    m.code AS mission_code,
                    m.name AS mission_name,

                    mm.model_name,
                    mm.model_version,
                    mm.model_path,

                    mm.confidence_threshold,
                    mm.iou_threshold

                FROM missions m

                JOIN mission_models mm
                    ON mm.mission_id = m.id

                WHERE
                    m.code = ANY(%s)
                    AND m.status = 'ACTIVE'
                    AND mm.is_active = TRUE

                ORDER BY
                    m.code,
                    mm.model_version
                """,
                (
                    mission_codes,
                )
            )

            return cursor.fetchall()

    finally:
        conn.close()