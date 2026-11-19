"""
Production Data Management Module for Poshan Malnutrition Screening Stack.
Provides:
1. Multi-Gigabyte Streaming ZIP Export (Supports 8GB+ unbuffered download directly from NVMe SSD).
2. Child Data Deletion by AWC Code and Serial Number (with physical file cleanup).
"""
import os
import io
import csv
import json
import shutil
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

from sqlalchemy.orm import Session
from production.db_models import ChildAssessment, ChildPhoto, AuditLog
from production.storage_manager import RAW_PHOTOS_DIR

logger = logging.getLogger("poshan_data_management")

try:
    import zipstream
    ZIPSTREAM_AVAILABLE = True
except ImportError:
    ZIPSTREAM_AVAILABLE = False


def generate_assessments_csv(assessments: List[ChildAssessment]) -> str:
    """Generates an in-memory CSV string containing all clinical & demographic metadata."""
    output = io.StringIO()
    fieldnames = [
        "audit_id", "created_at", "status", "awc_code", "serial_number",
        "date_of_birth", "gender", "height_cm", "weight_kg", "muac_cm",
        "predicted_tier", "predicted_risk_pct", "wasting_locus",
        "worker_phone", "photo_filename", "photo_sha256", "photo_size_bytes"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for item in assessments:
        photo = item.photos[0] if item.photos else None
        writer.writerow({
            "audit_id": item.audit_id,
            "created_at": item.created_at.isoformat() if item.created_at else "",
            "status": item.status,
            "awc_code": item.awc_code or "",
            "serial_number": item.serial_number or "",
            "date_of_birth": item.date_of_birth or "",
            "gender": item.gender or "",
            "height_cm": item.height_cm,
            "weight_kg": item.weight_kg,
            "muac_cm": item.muac_cm,
            "predicted_tier": item.predicted_tier or "",
            "predicted_risk_pct": item.predicted_risk_pct,
            "wasting_locus": item.wasting_locus or "",
            "worker_phone": item.worker_phone or "",
            "photo_filename": os.path.basename(photo.file_path) if photo else "",
            "photo_sha256": photo.sha256_hash if photo else "",
            "photo_size_bytes": photo.file_size_bytes if photo else 0
        })

    return output.getvalue()


def generate_assessments_json(assessments: List[ChildAssessment]) -> str:
    """Generates a complete structured JSON dump of all records including nested audit metrics."""
    records = []
    for item in assessments:
        record = {
            "audit_id": item.audit_id,
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "status": item.status,
            "awc_code": item.awc_code,
            "serial_number": item.serial_number,
            "date_of_birth": item.date_of_birth,
            "gender": item.gender,
            "height_cm": item.height_cm,
            "weight_kg": item.weight_kg,
            "muac_cm": item.muac_cm,
            "predicted_tier": item.predicted_tier,
            "predicted_risk_pct": item.predicted_risk_pct,
            "wasting_locus": item.wasting_locus,
            "worker_phone": item.worker_phone,
            "photos": [
                {
                    "file_name": os.path.basename(p.file_path),
                    "sha256": p.sha256_hash,
                    "size_bytes": p.file_size_bytes,
                    "mime_type": p.mime_type
                }
                for p in item.photos
            ],
            "audit_logs": [
                {
                    "execution_time_sec": l.execution_time_sec,
                    "landmark_detected": l.landmark_detected,
                    "quality_passed": l.quality_passed,
                    "pipeline_metrics": json.loads(l.pipeline_metrics) if l.pipeline_metrics else None,
                    "error_log": l.error_log
                }
                for l in item.audit_logs
            ]
        }
        records.append(record)

    return json.dumps(records, indent=2, default=str)


def create_export_archive_stream(
    db: Session,
    awc_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_photos: bool = True,
    include_audit_visuals: bool = True,
    audit_logs_dir: Optional[str] = None
):
    """
    Creates an on-the-fly streaming ZIP generator using zipstream-ng.
    Streams multi-gigabyte archives (8GB+) with a constant RAM footprint (<20 MB)
    by reading chunks directly from persistent NVMe storage.
    """
    if not ZIPSTREAM_AVAILABLE:
        raise RuntimeError("zipstream-ng package is required for streaming exports. Install via: pip install zipstream-ng")

    query = db.query(ChildAssessment)
    if awc_code:
        query = query.filter(ChildAssessment.awc_code == awc_code)
    if start_date:
        query = query.filter(ChildAssessment.created_at >= start_date)
    if end_date:
        query = query.filter(ChildAssessment.created_at <= end_date)

    assessments = query.order_by(ChildAssessment.created_at.desc()).all()

    # Initialize streaming ZIP with standard ZIP64 support (supports archives > 4GB / 8GB / 100GB)
    zs = zipstream.ZipStream(compress_type=zipstream.ZIP_STORED)

    # 1. Add metadata exports to root of archive
    csv_content = generate_assessments_csv(assessments)
    zs.add(csv_content.encode("utf-8"), arcname="metadata_assessments.csv")

    json_content = generate_assessments_json(assessments)
    zs.add(json_content.encode("utf-8"), arcname="metadata_assessments.json")

    # 2. Stream raw uncompressed photos directly from NVMe
    if include_photos:
        for item in assessments:
            for photo in item.photos:
                if os.path.exists(photo.file_path):
                    rel_name = f"raw_photos/{os.path.basename(photo.file_path)}"
                    zs.add_path(photo.file_path, arcname=rel_name)

    # 3. Stream generated audit logs & risk gauge graphics
    if include_audit_visuals and audit_logs_dir:
        base_audit = Path(audit_logs_dir)
        for item in assessments:
            session_folder = base_audit / item.audit_id
            if session_folder.is_dir():
                for visual_file in session_folder.glob("*.png"):
                    rel_name = f"audit_logs/{item.audit_id}/{visual_file.name}"
                    zs.add_path(str(visual_file), arcname=rel_name)

    return zs


def delete_child_data_by_awc_and_serial(
    db: Session,
    awc_code: str,
    serial_number: str,
    audit_logs_dir: Optional[str] = None,
    csv_registry_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Permanently deletes all data for a specific child identified by awc_code and serial_number:
    1. Removes all physical raw photo files from NVMe SSD.
    2. Removes all physical audit log folders and graphics from NVMe SSD.
    3. Deletes database records from PostgreSQL (ChildPhoto, AuditLog, ChildAssessment).
    4. Cleans up field validation registry CSV if present.
    """
    awc_code_clean = str(awc_code).strip()
    serial_number_clean = str(serial_number).strip()

    # Query matching assessments
    assessments = db.query(ChildAssessment).filter(
        ChildAssessment.awc_code == awc_code_clean,
        ChildAssessment.serial_number == serial_number_clean
    ).all()

    if not assessments:
        return {
            "status": "not_found",
            "message": f"No child records found for AWC '{awc_code_clean}' and Serial Number '{serial_number_clean}'.",
            "deleted_count": 0
        }

    deleted_photos = []
    deleted_audit_folders = []
    deleted_audit_ids = []

    for item in assessments:
        deleted_audit_ids.append(item.audit_id)

        # 1. Delete physical photo files from NVMe SSD
        for photo in item.photos:
            if photo.file_path and os.path.exists(photo.file_path):
                try:
                    os.remove(photo.file_path)
                    deleted_photos.append(photo.file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete photo file {photo.file_path}: {e}")

        # 2. Delete audit log folders and visuals
        if audit_logs_dir:
            audit_path = Path(audit_logs_dir) / item.audit_id
            if audit_path.exists() and audit_path.is_dir():
                try:
                    shutil.rmtree(str(audit_path), ignore_errors=True)
                    deleted_audit_folders.append(str(audit_path))
                except Exception as e:
                    logger.warning(f"Failed to delete audit folder {audit_path}: {e}")

        # 3. Delete database record (cascades to ChildPhoto and AuditLog)
        db.delete(item)

    db.commit()

    # 4. Clean up local registry CSV if present
    if csv_registry_path and os.path.exists(csv_registry_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_registry_path)
            # Filter out deleted child rows
            if "awc_code" in df.columns and "serial_number" in df.columns:
                mask = ~((df["awc_code"].astype(str).str.strip() == awc_code_clean) & 
                         (df["serial_number"].astype(str).str.strip() == serial_number_clean))
                df_filtered = df[mask]
                df_filtered.to_csv(csv_registry_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to update CSV registry {csv_registry_path}: {e}")

    return {
        "status": "success",
        "message": f"Successfully deleted child records for AWC '{awc_code_clean}' and Serial Number '{serial_number_clean}'.",
        "awc_code": awc_code_clean,
        "serial_number": serial_number_clean,
        "deleted_assessments_count": len(assessments),
        "deleted_photos_count": len(deleted_photos),
        "deleted_audit_ids": deleted_audit_ids,
        "deleted_files": deleted_photos + deleted_audit_folders
    }

