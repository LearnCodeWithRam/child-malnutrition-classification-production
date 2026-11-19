"""
field_validation_manager.py
===========================
Field Testing & Clinical Validation Intake System for ASHA / Anganwadi Workers.

Captures comprehensive child metadata, ground-truth physical anthro measurements (MUAC tape, height, weight),
left-arm photographs, vision biometrics, and AI ensemble prediction metrics into a persistent master CSV registry.
"""

import os
import csv
import json
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

CURRENT_DIR = Path(__file__).resolve().parent
AUDIT_LOGS_DIR = CURRENT_DIR / "audit_logs"
FIELD_REGISTRY_CSV = CURRENT_DIR / "field_validation_registry.csv"

# Canonical CSV Columns for Model Validation & Longitudinal Epidemiological Tracking
FIELD_CSV_COLUMNS = [
    "timestamp_iso",
    "session_id",
    "serial_number",
    "awc_code",
    "locality",
    "child_name",
    "child_mother_name",
    "date_of_birth",
    "age_month",
    "gender",
    "height_cm",
    "weight_kg",
    "muac_tape_cm",
    "worker_phone",
    "triage_tier",
    "nutritional_category",
    "primary_ensemble_risk_pct",
    "cloud_egat_risk_pct",
    "primary_wasting_locus",
    "vision_estimated_muac_cm",
    "muac_error_cm",
    "total_arm_length_cm",
    "upper_arm_length_cm",
    "forearm_length_cm",
    "quality_score",
    "qa_status",
    "lime_fidelity_r2",
    "original_image_path",
    "contour_overlay_path",
    "risk_gauge_speedometer_path",
    "interactive_report_url",
    "audit_dir"
]


def init_field_registry_csv():
    """Initializes master field CSV registry with canonical headers if not already present."""
    if not FIELD_REGISTRY_CSV.exists() or FIELD_REGISTRY_CSV.stat().st_size == 0:
        with open(FIELD_REGISTRY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(FIELD_CSV_COLUMNS)


def append_field_validation_record(
    metadata: Dict[str, Any],
    pipeline_diag: Dict[str, Any],
    base_url: str = ""
) -> Dict[str, Any]:
    """
    Appends a completed field assessment record to the master CSV registry and
    persists field_metadata.json inside the session's audit directory.
    """
    init_field_registry_csv()

    audit_summary = pipeline_diag.get("audit_log", {})
    audit_id = audit_summary.get("audit_id") or pipeline_diag.get("session_id", "")
    if not audit_id:
        audit_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the exact audit logs directory path requested for field validation
    folder_rel = rf"MC-ONC-Based-Classification\Full_Demo_MC_ONC_Classification\audit_logs\{audit_id}"
    orig_img = rf"MC-ONC-Based-Classification\Full_Demo_MC_ONC_Classification\audit_logs\{audit_id}\01_original_image.jpg"
    contour_img = rf"MC-ONC-Based-Classification\Full_Demo_MC_ONC_Classification\audit_logs\{audit_id}\02_contour_overlay.png"
    gauge_img = rf"MC-ONC-Based-Classification\Full_Demo_MC_ONC_Classification\audit_logs\{audit_id}\07_risk_gauge_speedometer.png"
    disk_audit_dir = AUDIT_LOGS_DIR / audit_id

    ens_res = pipeline_diag.get("offline_ensemble", {})
    egat_res = pipeline_diag.get("cloud_egat", {})
    bio_res = pipeline_diag.get("biometrics", {})
    lime_res = pipeline_diag.get("lime_explanation", {})
    q_score = pipeline_diag.get("quality_score", 100.0)

    # Physical Ground-truth values
    try:
        tape_muac = float(metadata.get("muac_cm", 0.0) or 0.0)
    except (ValueError, TypeError):
        tape_muac = None

    try:
        vision_muac = float(bio_res.get("vision_estimated_muac_cm", 0.0) or 0.0)
    except (ValueError, TypeError):
        vision_muac = None

    muac_error = None
    if tape_muac is not None and vision_muac is not None and tape_muac > 0 and vision_muac > 0:
        muac_error = round(vision_muac - tape_muac, 2)

    tier = ens_res.get("triage_tier", "UNKNOWN")
    category = ens_res.get("category", "Normal")
    field_nutri = metadata.get("nutritional_status")
    if field_nutri:
        category = f"{category} [Field: {str(field_nutri).upper()}]"
    prob_ens = round(float(ens_res.get("probability_ensemble", 0.0)) * 100.0, 1)
    prob_egat = round(float(egat_res.get("backup_probability_egat", 0.0)) * 100.0, 1) if egat_res else 0.0

    locus = egat_res.get("primary_wasting_site")
    if not locus or locus in ["N/A", "N/A (Backup Disabled)"]:
        locus = lime_res.get("primary_wasting_locus") or "Mid-Upper Arm / MUAC Site (L16-L15)"

    clean_base = base_url.rstrip("/") if base_url else ""
    report_url = f"{clean_base}/report/{audit_id}" if clean_base else f"/report/{audit_id}"

    # Auto-calculate age in months from date_of_birth if not explicitly provided
    calc_age_month = metadata.get("age_month", "")
    if not calc_age_month and metadata.get("date_of_birth"):
        dob_str = str(metadata.get("date_of_birth", "")).strip()
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
            try:
                dob_dt = datetime.strptime(dob_str, fmt)
                now_dt = datetime.now()
                diff_m = (now_dt.year - dob_dt.year) * 12 + (now_dt.month - dob_dt.month)
                if now_dt.day < dob_dt.day:
                    diff_m -= 1
                calc_age_month = str(max(0, diff_m))
                break
            except ValueError:
                continue

    # Saved artifact relative paths using exact audit logs directory requested
    # folder_rel, orig_img, contour_img, gauge_img already created above
    record_row = [
        datetime.now().isoformat(),
        audit_id,
        metadata.get("serial_number", ""),
        metadata.get("awc_code", ""),
        metadata.get("locality", ""),
        metadata.get("child_name", ""),
        metadata.get("child_mother_name", ""),
        metadata.get("date_of_birth", ""),
        calc_age_month,
        metadata.get("gender", ""),
        metadata.get("height_cm", ""),
        metadata.get("weight_kg", ""),
        tape_muac if tape_muac is not None else "",
        metadata.get("worker_phone", ""),
        tier,
        category,
        prob_ens,
        prob_egat,
        locus,
        vision_muac if vision_muac is not None else "",
        muac_error if muac_error is not None else "",
        bio_res.get("total_arm_length_cm", 28.5),
        bio_res.get("upper_arm_length_cm", ""),
        bio_res.get("forearm_length_cm", ""),
        q_score,
        "PASS" if pipeline_diag.get("status") == "success" else "QUALITY_REJECTED",
        lime_res.get("surrogate_fidelity_r2", ""),
        orig_img,
        contour_img,
        gauge_img,
        report_url,
        folder_rel
    ]

    with open(FIELD_REGISTRY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(record_row)

    # Also persist inside the session's audit folder for complete provenance
    if disk_audit_dir and disk_audit_dir.exists():
        field_meta_file = disk_audit_dir / "field_metadata.json"
        meta_payload = {
            "metadata": metadata,
            "tape_muac_cm": tape_muac,
            "vision_estimated_muac_cm": vision_muac,
            "muac_error_cm": muac_error,
            "recorded_at": datetime.now().isoformat()
        }
        with open(field_meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

    return {
        "status": "saved",
        "session_id": audit_id,
        "csv_path": str(FIELD_REGISTRY_CSV),
        "report_url": report_url,
        "primary_ensemble_risk_pct": prob_ens,
        "triage_tier": tier,
        "muac_error_cm": muac_error
    }


def generate_field_intake_html(worker_phone: str = "918349819726", default_awc: str = "AWC-AP-042") -> str:
    """
    Renders a responsive, mobile-first bilingual (English + Telugu) field intake form
    specifically designed for ASHA / Anganwadi frontline health workers.
    """
    today_date = datetime.now().strftime("%Y-%m-%d")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Poshan Field Assessment Intake | అంగన్‌వాడీ ఫీల్డ్ నమోదు</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Sans+Telugu:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --primary: #0284c7;
      --primary-dark: #0369a1;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --focus-ring: rgba(2, 132, 199, 0.25);
      --success: #10b981;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Plus Jakarta Sans', 'Noto Sans Telugu', sans-serif;
      background: var(--bg);
      color: var(--text-main);
      padding: 16px 12px 60px;
      display: flex;
      justify-content: center;
    }}
    .container {{
      width: 100%;
      max-width: 580px;
      background: var(--card-bg);
      border-radius: 20px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.03);
      border: 1px solid var(--border);
      overflow: hidden;
    }}
    .header {{
      background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
      color: #ffffff;
      padding: 24px 20px;
      text-align: center;
      position: relative;
    }}
    .header h1 {{
      font-size: 20px;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin-bottom: 4px;
    }}
    .header p {{
      font-size: 13px;
      opacity: 0.9;
      font-family: 'Noto Sans Telugu', sans-serif;
    }}
    .badge {{
      display: inline-block;
      background: rgba(255, 255, 255, 0.2);
      border-radius: 9999px;
      padding: 4px 12px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }}
    form {{
      padding: 20px 18px;
    }}
    .section-title {{
      font-size: 13px;
      font-weight: 700;
      color: var(--primary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin: 18px 0 12px;
      display: flex;
      align-items: center;
      gap: 6px;
      border-bottom: 2px solid #f1f5f9;
      padding-bottom: 6px;
    }}
    .form-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .col-span-2 {{
      grid-column: span 2;
    }}
    .form-group {{
      display: flex;
      flex-direction: column;
      gap: 5px;
    }}
    label {{
      font-size: 12.5px;
      font-weight: 600;
      color: #334155;
    }}
    label .te {{
      font-size: 11px;
      color: var(--text-muted);
      font-family: 'Noto Sans Telugu', sans-serif;
      font-weight: 400;
    }}
    input, select {{
      width: 100%;
      height: 44px;
      padding: 8px 12px;
      border-radius: 10px;
      border: 1.5px solid var(--border);
      background: #ffffff;
      font-size: 14px;
      color: var(--text-main);
      font-family: inherit;
      transition: all 0.2s ease;
    }}
    input:focus, select:focus {{
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px var(--focus-ring);
    }}
    .photo-card {{
      background: #f0fdf4;
      border: 2px dashed #86efac;
      border-radius: 14px;
      padding: 16px;
      text-align: center;
      cursor: pointer;
      position: relative;
      transition: all 0.2s ease;
    }}
    .photo-card:hover {{
      background: #dcfce7;
      border-color: #4ade80;
    }}
    .photo-card input[type="file"] {{
      position: absolute;
      top: 0; left: 0; width: 100%; height: 100%;
      opacity: 0;
      cursor: pointer;
    }}
    .photo-icon {{
      font-size: 32px;
      margin-bottom: 6px;
    }}
    .photo-prompt {{
      font-size: 13px;
      font-weight: 700;
      color: #166534;
    }}
    .photo-sub {{
      font-size: 11px;
      color: #15803d;
      margin-top: 2px;
      font-family: 'Noto Sans Telugu', sans-serif;
    }}
    #image_preview {{
      display: none;
      max-width: 100%;
      max-height: 220px;
      margin: 12px auto 0;
      border-radius: 10px;
      border: 2px solid #86efac;
      object-fit: contain;
    }}
    .btn-submit {{
      margin-top: 24px;
      width: 100%;
      height: 52px;
      background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
      color: #ffffff;
      border: none;
      border-radius: 12px;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35);
      transition: transform 0.1s ease, box-shadow 0.1s ease;
    }}
    .btn-submit:active {{
      transform: scale(0.98);
      box-shadow: 0 2px 6px rgba(2, 132, 199, 0.25);
    }}
    .btn-submit:disabled {{
      background: #94a3b8;
      cursor: not-allowed;
      box-shadow: none;
    }}
    .overlay {{
      display: none;
      position: fixed;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(15, 23, 42, 0.7);
      backdrop-filter: blur(4px);
      z-index: 999;
      justify-content: center;
      align-items: center;
      flex-direction: column;
      color: #ffffff;
      padding: 24px;
      text-align: center;
    }}
    .spinner {{
      width: 48px;
      height: 48px;
      border: 4px solid rgba(255, 255, 255, 0.3);
      border-top-color: #38bdf8;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin-bottom: 16px;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  </style>
</head>
<body>

<div class="container">
  <div class="header">
    <div class="badge">Field Validation Survey • ఫీల్డ్ సర్వే</div>
    <h1>Poshan Child Assessment</h1>
    <p>పోషణ్ పిల్లల పరీక్ష మరియు పోషకాహార నమోదు ఫారం</p>
  </div>

  <form id="intakeForm" action="/api/v1/submit-field-intake" method="POST" enctype="multipart/form-data">
    
    <!-- Section 1: Location & Clinic Registry -->
    <div class="section-title">📍 1. AWC details / అంగన్‌వాడీ కేంద్రం</div>
    <div class="form-grid">
      <div class="form-group">
        <label>AWC Code <span class="te">/ అంగన్‌వాడీ కోడ్</span></label>
        <input type="text" name="awc_code" value="{default_awc}" required placeholder="e.g. AWC-AP-042">
      </div>
      <div class="form-group">
        <label>AWC Name <span class="te">/ క్రమ సంఖ్య </span></label>
        <input type="text" name="serial_number" placeholder="e.g. SN-048" required>
      </div>
    </div>

    <!-- Section 2: Child Identity -->
    <div class="section-title">👶 2. Child Information / బిడ్డ వివరాలు</div>
    <div class="form-grid">
      <div class="form-group">
        <label>Date of Birth <span class="te">/ పుట్టిన తేదీ (No future date)</span></label>
        <input type="date" name="date_of_birth" id="dobInput" max="{today_date}" required>
      </div>
      <div class="form-group">
        <label>Gender <span class="te">/ లింగం</span></label>
        <select name="gender" required>
          <option value="Female">Female / బాలిక</option>
          <option value="Male">Male / బాలుడు</option>
        </select>
      </div>
    </div>

    <!-- Section 3: Field Physical Measurements -->
    <div class="section-title">📏 3. Physical Anthropometrics / క్షేత్ర కొలతలు</div>
    <div class="form-grid">
      <div class="form-group">
        <label>Height (cm) <span class="te">/ ఎత్తు (సెం.మీ)</span></label>
        <input type="number" step="0.1" name="height_cm" placeholder="e.g. 82.5" required>
      </div>
      <div class="form-group">
        <label>Weight (kg) <span class="te">/ బరువు (కిలోలు)</span></label>
        <input type="number" step="0.1" name="weight_kg" placeholder="e.g. 10.4" required>
      </div>
      <div class="form-group col-span-2">
        <label>Physical Tape MUAC (cm) <span class="te">/ టేపుతో కొలిచిన MUAC (సెం.మీ)</span></label>
        <input type="number" step="0.1" name="muac_cm" placeholder="e.g. 12.8" required>
      </div>
    </div>

    <!-- Section 4: Left Arm Photograph Capture -->
    <div class="section-title">📸 4. Bare Left Arm Photo / ఎడమ చేయి ఫోటో</div>
    <div class="photo-card" id="photoDropArea">
      <input type="file" name="left_arm_image" id="fileInput" accept="image/*" capture="environment" required>
      <div class="photo-icon">📷</div>
      <div class="photo-prompt">Take Photo with Camera or Browse</div>
      <div class="photo-sub">కెమెరాతో ఎడమ చేయి ఫోటో తీయండి (భుజం నుండి మణికట్టు వరకు)</div>
      <img id="image_preview" alt="Left Arm Preview">
    </div>

    <!-- Submit Button -->
    <button type="submit" class="btn-submit" id="submitBtn">
      <span>🚀 Get Result / ఫలితాన్ని పొందండి</span>
    </button>
  </form>
</div>

<!-- Loading Overlay -->
<div class="overlay" id="loadingOverlay">
  <div class="spinner"></div>
  <h2 style="font-size: 18px; margin-bottom: 6px;">Processing Child Arm Photo...</h2>
  <p style="font-size: 13px; opacity: 0.9; font-family: 'Noto Sans Telugu', sans-serif;">
    నాణ్యత తనిఖీ, 22 ల్యాండ్‌మార్క్స్ మరియు రిస్క్ గేజ్ లెక్కిస్తున్నాము...
  </p>
</div>

<script>
  // Calculate age in months automatically when DOB is chosen and block future dates
  const dobInput = document.getElementById('dobInput');
  const ageInput = document.getElementById('ageMonthInput');
  const todayStr = new Date().toISOString().split('T')[0];
  dobInput.max = todayStr;
  dobInput.addEventListener('change', () => {{
    if (!dobInput.value) return;
    if (dobInput.value > todayStr) {{
      alert("⚠️ Date of Birth cannot be in the future / పుట్టిన తేదీ భవిష్యత్తులో ఉండకూడదు.");
      dobInput.value = todayStr;
    }}
    const dob = new Date(dobInput.value);
    const today = new Date();
    let months = (today.getFullYear() - dob.getFullYear()) * 12 + (today.getMonth() - dob.getMonth());
    if (today.getDate() < dob.getDate()) months--;
    if (months < 0) months = 0;
    ageInput.value = months;
  }});

  // Live image preview
  const fileInput = document.getElementById('fileInput');
  const imgPreview = document.getElementById('image_preview');
  fileInput.addEventListener('change', () => {{
    if (fileInput.files && fileInput.files[0]) {{
      const reader = new FileReader();
      reader.onload = (e) => {{
        imgPreview.src = e.target.result;
        imgPreview.style.display = 'block';
      }};
      reader.readAsDataURL(fileInput.files[0]);
    }}
  }});

  // Form submission spinner
  const form = document.getElementById('intakeForm');
  const overlay = document.getElementById('loadingOverlay');
  const submitBtn = document.getElementById('submitBtn');
  form.addEventListener('submit', () => {{
    overlay.style.display = 'flex';
    submitBtn.disabled = true;
  }});
</script>

</body>
</html>
"""

