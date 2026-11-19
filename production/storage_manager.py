"""
Storage Manager for Approach B: NVMe Filesystem + PostgreSQL Metadata.
Preserves raw uncompressed binary image bytes directly on persistent NVMe storage
and logs exact file path, file size, MIME type, and SHA-256 integrity hash in PostgreSQL.
"""
import os
import hashlib
from typing import Tuple
from sqlalchemy.orm import Session
from production.db_models import ChildPhoto

# Configurable storage path (mapped to persistent NVMe host directory via Docker bind mount)
RAW_PHOTOS_DIR = os.getenv("RAW_PHOTOS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_photos"))

def ensure_storage_dir(dir_path: str = None) -> str:
    path = dir_path or RAW_PHOTOS_DIR
    os.makedirs(path, exist_ok=True)
    return path

def save_raw_photo_lossless(
    raw_bytes: bytes, 
    audit_id: str, 
    db: Session,
    extension: str = "jpg",
    mime_type: str = "image/jpeg"
) -> Tuple[ChildPhoto, str]:
    """
    Saves binary image bytes directly to persistent NVMe disk without re-encoding or compression.
    Computes SHA-256 integrity checksum and records entry in PostgreSQL.
    
    Returns:
        (ChildPhoto model instance, absolute file path on disk)
    """
    if not raw_bytes or len(raw_bytes) == 0:
        raise ValueError("Cannot save empty or null byte buffer.")

    target_dir = ensure_storage_dir()
    
    # Compute SHA-256 hash directly on incoming binary payload
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
    file_size = len(raw_bytes)
    
    filename = f"{audit_id}.{extension.lstrip('.')}"
    file_path = os.path.join(target_dir, filename)

    # Write raw uncompressed bytes directly to disk
    with open(file_path, "wb") as f:
        f.write(raw_bytes)
        f.flush()
        os.fsync(f.fileno())  # Flush OS buffer directly to NVMe SSD

    # Verify physical file integrity immediately
    disk_size = os.path.getsize(file_path)
    if disk_size != file_size:
        raise IOError(f"Disk write corrupted! Expected {file_size} bytes, got {disk_size} bytes.")

    # Record metadata in PostgreSQL
    photo_record = ChildPhoto(
        audit_id=audit_id,
        file_path=file_path,
        sha256_hash=sha256_hash,
        file_size_bytes=file_size,
        mime_type=mime_type
    )
    db.add(photo_record)
    db.commit()
    db.refresh(photo_record)

    return photo_record, file_path


def verify_photo_integrity(photo_record: ChildPhoto) -> bool:
    """
    Verifies that the file on disk matches its recorded SHA-256 hash bit-for-bit.
    Guarantees zero data loss, silent corruption, or post-upload re-compression.
    """
    if not os.path.exists(photo_record.file_path):
        return False

    with open(photo_record.file_path, "rb") as f:
        disk_bytes = f.read()

    computed_hash = hashlib.sha256(disk_bytes).hexdigest()
    return computed_hash == photo_record.sha256_hash

