"""
Celery Task Pipeline for Poshan Child Malnutrition Screening.
Executes the computer vision + geometric morphometrics pipeline asynchronously,
persists diagnostic results to PostgreSQL, and dispatches WhatsApp reports to frontline workers.
"""
import os
import sys
import time
import json
import shutil
import logging
import requests
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so existing ML modules can be imported seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from production.celery_app import celery
from production.database import SessionLocal
from production.db_models import ChildAssessment, ChildPhoto, AuditLog
from production.storage_manager import verify_photo_integrity

# Import domain logic from core system without modifying local files
from app import (
    _execute_mobile_pipeline,
    format_whatsapp_triage_text,
    format_whatsapp_error_text,
    send_meta_whatsapp_image,
    send_meta_whatsapp_message,
    send_meta_whatsapp_flow_message,
    append_field_validation_record,
    AUDIT_LOGS_DIR
)

logger = logging.getLogger("poshan_celery_worker")
logger.setLevel(logging.INFO)

@celery.task(bind=True, max_retries=2, default_retry_delay=5)
def process_child_screening_task(
    self,
    audit_id: str,
    phone_number_id: str,
    sender_phone: str,
    flow_data: dict,
    flow_id: str = None
):
    """
    Asynchronous screening worker:
    1. Loads raw uncompressed photo from NVMe disk and validates SHA-256 hash against PostgreSQL.
    2. Runs Computer Vision & Geometric Morphometrics pipeline.
    3. Updates assessment and audit logs in PostgreSQL.
    4. Dispatches the WhatsApp risk gauge image report and follow-up Flow CTA to the worker.
    """
    logger.info(f"🚀 [Task Start] Processing screening audit_id={audit_id} for sender={sender_phone}")
    start_time = time.time()
    db = SessionLocal()

    try:
        # Step 1: Retrieve assessment & photo record from PostgreSQL
        assessment = db.query(ChildAssessment).filter(ChildAssessment.audit_id == audit_id).first()
        if not assessment:
            raise ValueError(f"Assessment record '{audit_id}' not found in database.")

        photo_record = db.query(ChildPhoto).filter(ChildPhoto.audit_id == audit_id).first()
        if not photo_record:
            raise ValueError(f"Photo record for audit_id='{audit_id}' not found in database.")

        # Step 2: Verify zero-loss integrity of raw binary bytes on NVMe SSD
        if not verify_photo_integrity(photo_record):
            raise IOError(f"Integrity check failed: File {photo_record.file_path} does not match recorded SHA-256 hash.")

        with open(photo_record.file_path, "rb") as f:
            raw_img_bytes = f.read()

        assessment.status = "PROCESSING"
        db.commit()

        # Step 3: Execute Computer Vision + Classification Pipeline
        source_label = f"flow_{flow_data.get('serial_number') or flow_data.get('awc_code') or 'child'}"
        diag = _execute_mobile_pipeline(
            raw_img_bytes,
            source_name=source_label,
            include_visuals=False,
            status_callback=None
        )

        exec_duration = round(time.time() - start_time, 3)
        base_url = os.environ.get("PUBLIC_APP_URL") or "https://poshan.ap.gov.in"
        append_field_validation_record(flow_data, diag, base_url=base_url)

        # Step 4: Extract clinical indicators & results
        is_rejected = (diag.get("status") == "quality_rejected")
        tier = "REJECTED" if is_rejected else diag.get("tier", "UNKNOWN")
        risk_pct = diag.get("risk_probability") or diag.get("predicted_risk_pct")
        if risk_pct is not None:
            risk_pct = round(float(risk_pct) * 100 if float(risk_pct) <= 1.0 else float(risk_pct), 1)
        wasting_locus = diag.get("primary_wasting_locus") or diag.get("primary_indicator") or "N/A"

        # Update Assessment in PostgreSQL
        assessment.predicted_tier = tier
        assessment.predicted_risk_pct = risk_pct
        assessment.wasting_locus = wasting_locus
        assessment.status = "REJECTED" if is_rejected else "COMPLETED"
        db.commit()

        # Record detailed diagnostic metrics in PostgreSQL audit_logs
        log_entry = AuditLog(
            audit_id=audit_id,
            execution_time_sec=exec_duration,
            landmark_detected=diag.get("landmark_detected", True),
            quality_passed=(not is_rejected),
            pipeline_metrics=json.dumps(diag, default=str),
            error_log=None
        )
        db.add(log_entry)
        db.commit()

        # Step 5: Format bilingual WhatsApp Triage text
        triage_reply = format_whatsapp_triage_text(diag)

        # Step 6: Dispatch Risk Gauge Speedometer Image with Report Caption
        if not is_rejected:
            # 1. Resolve audit directory from pipeline execution & ensure target dir exists
            diag_audit_id = diag.get("audit_log", {}).get("audit_id") or audit_id
            target_audit_dir = Path(AUDIT_LOGS_DIR) / audit_id
            target_audit_dir.mkdir(parents=True, exist_ok=True)

            # Copy all generated audit files into target_audit_dir if diag created a distinct directory
            if diag_audit_id != audit_id:
                src_audit_dir = Path(AUDIT_LOGS_DIR) / diag_audit_id
                if src_audit_dir.exists():
                    for item in src_audit_dir.iterdir():
                        dest_file = target_audit_dir / item.name
                        if not dest_file.exists():
                            try:
                                shutil.copy2(item, dest_file)
                            except Exception:
                                pass

            gauge_file = target_audit_dir / "07_risk_gauge_speedometer.png"
            if not gauge_file.exists():
                try:
                    from risk_gauge_visualizer import generate_risk_gauge_speedometer
                    from audit_logger import _decode_b64_to_bytes
                    prob_val = float(risk_pct) / 100.0 if risk_pct is not None else 0.5
                    gauge_b64 = generate_risk_gauge_speedometer(probability=prob_val, tier=tier)
                    with open(gauge_file, "wb") as f:
                        f.write(_decode_b64_to_bytes(gauge_b64))
                except Exception as _g_ex:
                    logger.warning(f"Could not synthesize risk gauge image: {_g_ex}")

            # 2. Upload image directly to Meta WhatsApp Media CDN (high reliability, no crawling)
            meta_token = os.environ.get("META_ACCESS_TOKEN") or os.environ.get("TWILIO_AUTH_TOKEN")
            media_id = None
            if gauge_file.exists() and meta_token:
                try:
                    up_url = f"https://graph.facebook.com/v21.0/{phone_number_id}/media"
                    up_headers = {"Authorization": f"Bearer {meta_token}"}
                    with open(gauge_file, "rb") as f:
                        files = {
                            "file": (gauge_file.name, f, "image/png"),
                            "type": (None, "image/png"),
                            "messaging_product": (None, "whatsapp")
                        }
                        up_resp = requests.post(up_url, headers=up_headers, files=files, timeout=30)
                        if up_resp.status_code in (200, 201):
                            media_id = up_resp.json().get("id")
                            logger.info(f"✅ [Meta WA] Uploaded risk gauge to Meta Media CDN (ID: {media_id})")
                        else:
                            logger.warning(f"Meta Media Upload failed HTTP {up_resp.status_code}: {up_resp.text}")
                except Exception as _up_ex:
                    logger.warning(f"Meta Media Upload exception: {_up_ex}")

            # 3. Deliver image message via Meta Cloud API using media ID
            delivered_visual = False
            if media_id and meta_token:
                try:
                    msg_url = f"https://graph.facebook.com/v21.0/{phone_number_id}/messages"
                    msg_headers = {
                        "Authorization": f"Bearer {meta_token}",
                        "Content-Type": "application/json"
                    }
                    msg_payload = {
                        "messaging_product": "whatsapp",
                        "recipient_type": "individual",
                        "to": sender_phone,
                        "type": "image",
                        "image": {
                            "id": media_id,
                            "caption": triage_reply[:1024]
                        }
                    }
                    msg_resp = requests.post(msg_url, headers=msg_headers, json=msg_payload, timeout=25)
                    if msg_resp.status_code in (200, 201):
                        delivered_visual = True
                        logger.info(f"✅ Delivered Speedometer Risk Gauge to {sender_phone} via Meta Media ID: {media_id}")
                    else:
                        logger.warning(f"Failed to send image by ID ({msg_resp.status_code}): {msg_resp.text}")
                except Exception as _m_ex:
                    logger.warning(f"Exception sending image message: {_m_ex}")

            if not delivered_visual:
                send_meta_whatsapp_message(phone_number_id, sender_phone, triage_reply)
        else:
            send_meta_whatsapp_message(phone_number_id, sender_phone, triage_reply)

        # Step 7: Send Follow-up Action Button (New Assessment or Retake Photo)
        effective_flow_id = flow_id or os.environ.get("META_FLOW_ID")
        if effective_flow_id:
            if is_rejected:
                retake_data = {
                    "awc_code": str(flow_data.get("awc_code", "") or ""),
                    "serial_number": str(flow_data.get("serial_number", "") or ""),
                    "date_of_birth": str(flow_data.get("date_of_birth", "") or datetime.now().strftime("%Y-%m-%d")),
                    "gender": str(flow_data.get("gender", "") or "Female"),
                    "height_cm": str(flow_data.get("height_cm", "") or "75.0"),
                    "weight_kg": str(flow_data.get("weight_kg", "") or "9.0"),
                    "muac_cm": str(flow_data.get("muac_cm", "") or "12.5"),
                    "nutritional_status": str(flow_data.get("nutritional_status", "") or "mam")
                }
                send_meta_whatsapp_flow_message(
                    phone_number_id=phone_number_id,
                    recipient_phone=sender_phone,
                    flow_id=effective_flow_id,
                    screen_id="DEMOGRAPHICS_SCREEN",
                    header_text="ఫోటో మార్పిడి / Retake Photo",
                    body_text=(
                        "📸 *కొత్త ఫోటో తీయండి / Retake Child Photo*\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "పిల్లల కొలతలు మరియు వివరాలు ముందే నింపబడ్డాయి (Auto-filled).\n"
                        "దయచేసి క్రింది బటన్ నొక్కి వివరాలు సమీక్షించి, కొత్త ఫోటోను సమర్పించండి.\n"
                        "(Child details are auto-filled. Tap below to review and re-capture a fresh photo.)"
                    ),
                    cta_text="📸 Retake Photo",
                    initial_data=retake_data
                )
            else:
                send_meta_whatsapp_flow_message(
                    phone_number_id=phone_number_id,
                    recipient_phone=sender_phone,
                    flow_id=effective_flow_id,
                    screen_id="DEMOGRAPHICS_SCREEN",
                    header_text="పోషణ్ తనిఖీ / Child Registration",
                    body_text=(
                        "✅ *తదుపరి పిల్లల తనిఖీ / Next Screening*\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "ఈ పిల్లవాని నివేదిక విజయవంతంగా నమోదైంది. తదుపరి పిల్లల ఆరోగ్య తనిఖీని ప్రారంభించడానికి క్రింది బటన్ నొక్కండి.\n"
                        "(Report saved successfully. Tap below to begin assessment for the next child.)"
                    ),
                    cta_text="➕ New Assessment"
                )

        logger.info(f"✅ [Task Complete] audit_id={audit_id} duration={exec_duration}s tier={tier}")
        return {"status": "success", "audit_id": audit_id, "tier": tier, "duration": exec_duration}

    except Exception as exc:
        exec_duration = round(time.time() - start_time, 3)
        logger.error(f"❌ [Task Error] audit_id={audit_id}: {exc}", exc_info=True)
        if 'assessment' in locals() and assessment:
            assessment.status = "FAILED"
            db.commit()
            error_log = AuditLog(
                audit_id=audit_id,
                execution_time_sec=exec_duration,
                landmark_detected=False,
                quality_passed=False,
                pipeline_metrics=None,
                error_log=str(exc)
            )
            db.add(error_log)
            db.commit()

        err_msg = format_whatsapp_error_text(exc)
        send_meta_whatsapp_message(phone_number_id, sender_phone, err_msg)
        raise self.retry(exc=exc)
    finally:
        db.close()

