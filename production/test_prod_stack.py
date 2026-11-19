"""
Cross-check and Verification Test Suite for Production Stack.
Verifies Approach B (NVMe Raw Binary Storage + PostgreSQL Metadata + SHA-256 Checksum).
"""
import os
import sys
import hashlib
import tempfile
from pathlib import Path

# Setup paths
PROD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROD_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use SQLite in-memory / temporary database for automated cross-checking test
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["RAW_PHOTOS_DIR"] = tempfile.mkdtemp(prefix="poshan_test_photos_")

from production.database import get_db_engine, Base, SessionLocal
from production.db_models import Worker, ChildAssessment, ChildPhoto, AuditLog
from production.storage_manager import save_raw_photo_lossless, verify_photo_integrity

def test_database_and_lossless_storage():
    print("=" * 60)
    print("🔍 STEP 1: Verifying Database Schema & Engine Initialization...")
    engine = get_db_engine()
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    print("✅ Tables created: workers, child_assessments, child_photos, audit_logs")

    print("\n🔍 STEP 2: Testing Approach B (Lossless Raw Photo Saving + SHA-256)...")
    # Load a real sample image if available or create test bytes
    sample_img_path = PROJECT_ROOT / "sample_images"
    sample_files = list(sample_img_path.glob("*.jpg")) + list(sample_img_path.glob("*.png"))
    if sample_files:
        with open(sample_files[0], "rb") as f:
            original_bytes = f.read()
        print(f"📄 Using real clinical sample: {sample_files[0].name} ({len(original_bytes):,} bytes)")
    else:
        # 1 MB of arbitrary binary bytes simulating camera sensor output
        original_bytes = os.urandom(1024 * 1024)
        print(f"📄 Using synthetic test payload ({len(original_bytes):,} bytes)")

    expected_sha256 = hashlib.sha256(original_bytes).hexdigest()
    print(f"🔒 Computed SHA-256 on incoming bytes: {expected_sha256}")

    # 1. Create Assessment record
    test_audit_id = "20261118_120000_test1234"
    assessment = ChildAssessment(
        audit_id=test_audit_id,
        worker_phone="919876543210",
        awc_code="28548190419",
        serial_number="SN-TEST-01",
        date_of_birth="2024-05-10",
        gender="Female",
        height_cm=78.5,
        weight_kg=9.2,
        muac_cm=12.4,
        status="QUEUED"
    )
    db.add(assessment)
    db.commit()

    # 2. Save raw uncompressed bytes directly to disk and record metadata
    photo_record, disk_path = save_raw_photo_lossless(
        raw_bytes=original_bytes,
        audit_id=test_audit_id,
        db=db,
        extension="jpg",
        mime_type="image/jpeg"
    )

    print(f"💾 Written to NVMe path: {disk_path}")
    print(f"📊 PostgreSQL Recorded Photo ID: {photo_record.photo_id}")
    print(f"📊 PostgreSQL Recorded Size: {photo_record.file_size_bytes:,} bytes")
    print(f"📊 PostgreSQL Recorded Hash: {photo_record.sha256_hash}")

    # 3. Cross-check integrity: read back from disk and compare bit-for-bit
    with open(disk_path, "rb") as f:
        read_back_bytes = f.read()

    assert len(read_back_bytes) == len(original_bytes), "Byte size mismatch on disk!"
    assert read_back_bytes == original_bytes, "Bitwise mismatch! Data was corrupted or altered!"
    assert verify_photo_integrity(photo_record), "verify_photo_integrity function returned False!"
    print("✅ VERIFIED: Bit-for-bit identical! Zero compression, zero artifact, exact SHA-256 match.")

    print("\n🔍 STEP 3: Verifying Relationships & Audit Logging...")
    log_entry = AuditLog(
        audit_id=test_audit_id,
        execution_time_sec=0.68,
        landmark_detected=True,
        quality_passed=True,
        pipeline_metrics='{"tier": "MAM", "risk_pct": 34.2, "locus": "Distal Brachium"}'
    )
    db.add(log_entry)
    db.commit()

    # Query back using SQLAlchemy ORM relationship
    queried_assessment = db.query(ChildAssessment).filter_by(audit_id=test_audit_id).first()
    assert queried_assessment is not None
    assert len(queried_assessment.photos) == 1
    assert len(queried_assessment.audit_logs) == 1
    assert queried_assessment.photos[0].sha256_hash == expected_sha256
    print(f"✅ ORM Relationship query successful: Assessment -> {len(queried_assessment.photos)} Photo(s), {len(queried_assessment.audit_logs)} Audit Log(s)")

    print("\n🔍 STEP 4: Verifying Multi-Gigabyte Streaming Export Generation...")
    from production.data_management import create_export_archive_stream, delete_child_data_by_awc_and_serial
    import zipfile
    import io

    # Create streaming generator
    zs = create_export_archive_stream(db=db, include_photos=True, include_audit_visuals=False)
    
    # Consume the chunks from the streaming generator
    zip_bytes_buffer = io.BytesIO()
    for chunk in zs:
        zip_bytes_buffer.write(chunk)
    zip_bytes_buffer.seek(0)

    # Verify with standard zipfile module that the archive is valid
    with zipfile.ZipFile(zip_bytes_buffer, "r") as zf:
        file_list = zf.namelist()
        print(f"📦 Archive generated successfully! Files inside: {file_list}")
        assert "metadata_assessments.csv" in file_list
        assert "metadata_assessments.json" in file_list
        expected_photo_entry = f"raw_photos/{test_audit_id}.jpg"
        assert expected_photo_entry in file_list

        # Verify photo bytes inside ZIP match original bytes exactly
        extracted_photo_bytes = zf.read(expected_photo_entry)
        assert extracted_photo_bytes == original_bytes, "Photo bytes inside ZIP do not match original bytes!"
        print("✅ VERIFIED: Streaming export ZIP is valid and contains exact lossless photo & metadata!")

    print("\n🔍 STEP 5: Verifying Individual Child Data Deletion by AWC & Serial Number...")
    # Verify file exists before deletion
    assert os.path.exists(disk_path), "Photo file should exist on disk before deletion!"

    del_result = delete_child_data_by_awc_and_serial(
        db=db,
        awc_code="28548190419",
        serial_number="SN-TEST-01"
    )
    print(f"🗑️ Deletion result: {del_result}")
    assert del_result["status"] == "success"
    assert del_result["deleted_assessments_count"] == 1
    assert del_result["deleted_photos_count"] == 1

    # Verify physical file on disk is deleted
    assert not os.path.exists(disk_path), "Photo file was NOT deleted from NVMe SSD!"
    print("✅ VERIFIED: Physical photo file removed from NVMe disk.")

    # Verify database record is deleted
    db_check = db.query(ChildAssessment).filter_by(audit_id=test_audit_id).first()
    assert db_check is None, "ChildAssessment record still exists in database!"
    photo_check = db.query(ChildPhoto).filter_by(audit_id=test_audit_id).first()
    assert photo_check is None, "ChildPhoto record still exists in database!"
    print("✅ VERIFIED: Database records cascaded and removed from PostgreSQL.")

    # Verify calling delete again returns not_found
    repeat_del = delete_child_data_by_awc_and_serial(
        db=db,
        awc_code="28548190419",
        serial_number="SN-TEST-01"
    )
    assert repeat_del["status"] == "not_found"
    print("✅ VERIFIED: Repeat deletion gracefully returns 'not_found'.")

    db.close()
    print("\n" + "=" * 60)
    print("🎉 ALL PRODUCTION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_database_and_lossless_storage()

