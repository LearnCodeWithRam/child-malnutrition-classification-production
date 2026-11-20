# PediaMorph AI - Mobile Integration REST API

A lightweight, high-performance REST API designed specifically for **mobile applications (Flutter, React Native, Android/Kotlin, iOS/Swift)** to perform frontline child acute malnutrition screening using child left-arm photographs.

---

## 🌟 Core Highlights

1. **NO CHECKERBOARD REQUIRED**:
   - In field deployment, children are photographed in natural environments without calibration mats.
   - The frontline computer vision pipeline uses MediaPipe Pose and Multiclass Arm Segmentation to detect the arm and automatically scales measurements using canonical pediatric anthropometrics ($28.5\text{ cm}$ canonical arm length).
   - Procrustes superimposition (fGPA) renders the 36D shape analysis strictly scale- and rotation-invariant.

2. **Built for Mobile Integration**:
   - Accepts already-clicked photos via **Multipart File Upload**, **Base64 JSON strings**, or **Local File Paths**.
   - Sub-second execution time ($<0.7\text{s}$ total on standard CPU).
   - Returns clean, standardized JSON with actionable triage tiers (🟢 Green, 🟡 Yellow, 🔴 Red), hex color codes, biometrics, and 36D shape residuals.

3. **Cloud EGAT Architecture (Offline Mobile Ensemble To Be Installed Separately)**:
   - **Cloud REST API (This Service)**: Hosts the **17-node Edge-Augmented Graph Attention Network (EGAT)** with 9 transverse calipers for deep deformation attention, primary wasting locus localization, and explainable wireframe heatmaps.
   - **Offline Edge Ensemble (Mobile Side)**: The 40% Shrinkage LDA + 60% XGBoost ensemble will be installed directly on mobile handsets separately for offline screening.
   - **Residuals Returned**: This API returns the standardized 36D shape residuals in the response so mobile clients have them available if needed.
   - **Multimodal Escalation**: Secondary resolver for borderline Yellow triage cases using physical tape MUAC ($>89.7\%$ clinical accuracy).

4. **Zero Cloud Infrastructure Overhead**:
   - No Celery, No Redis, No S3, No Docker needed. Self-contained FastAPI service.

---

## 📏 Zero-Checkerboard Metric Scaling & Shape Invariance Logic

### 1. Anatomical Arm Anchor
MediaPipe detects the child's skeletal joints: **Shoulder** and **Wrist**. The pixel distance between them is calibrated to the standard pediatric arm length (**28.5 cm** for 6–59 month children):
$$\text{px\_per\_cm} = \frac{\text{Pixel Distance (Shoulder to Wrist)}}{28.5\text{ cm}}$$

### 2. Procrustes Scale Invariance (The Real Secret)
Geometric Morphometrics does not rely on absolute scale or camera distance. During Procrustes Superimposition (fGPA), the shape is divided by its Centroid Size ($CS$):
$$\mathbf{X}_{\text{shape}} = \frac{\mathbf{X} - \mathbf{\bar{X}}}{CS}$$

This completely cancels out zoom level, camera distance, and resolution.

### 3. What the Model Evaluates
The AI classifies malnutrition by **relative shape deformation**, not raw pixels:
* **Bicep-to-forearm thickness ratio**
* **Lateral contour concavity** (muscle wasting curves)
* **Relative width differences across the 9 transverse calipers** (e.g., MUAC site wasting relative to joint bones).

### 4. Scientific & Empirical Justification for 28.5 cm Canonical Arm Length
The **28.5 cm** anchor is rigorously grounded in both empirical ground truth from the clinical cohort and international pediatric standards:

1. **Empirical Cohort Ground Truth ($N = 3,868$ Clinical Records)**:
   Analysis of the ground-truth checkerboard-scaled annotations in [`final_cleaned_dataset_augmented_2x_mc1934_onc1934.csv`](file:///E:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/MC-ONC-Based-Classification/final_cleaned_dataset_augmented_2x_mc1934_onc1934.csv) demonstrates that 28.5 cm is the exact mathematical center of the child population distribution:
   * **Inter-Landmark Distance (`ILD_Shoulder_Wrist`)**: Mean = **$28.19\text{ cm}$**, Median = **$28.78\text{ cm}$** (Midpoint = **$28.49\text{ cm} \approx \mathbf{28.5\text{ cm}}$**).
   * **Wrist Center $X$-Coordinate ($X_3$ relative to Shoulder at $X=0$)**: Mean = **$27.86\text{ cm}$**, Median = **$28.43\text{ cm}$** ($\approx \mathbf{28.5\text{ cm}}$).
   * **Real Non-Augmented Cohort ($N = 2,901$)**: Mean ILD = **$28.25\text{ cm}$**, Median $X_3$ = **$28.64\text{ cm}$**.

2. **Pediatric Anthropometry (WHO & CDC NHANES Standards)**:
   * **Median Child Length**: In the WHO Multicentre Growth Reference Study (MGRS) for children aged 6 to 59 months, the median recumbent length/stature is **$85.5\text{ cm}$** (center of the 24–36 month cohort).
   * **Upper Extremity Proportion Ratio**: Human pediatric anatomical standards indicate that upper limb length (Acromion to Stylion) consistently represents **$\approx 33.3\%$ of stature**:
     $$\text{Arm Length} = 85.5\text{ cm} \times 0.333 = \mathbf{28.47\text{ cm}} \approx \mathbf{28.5\text{ cm}}$$
   * **Segment Summation**: Upper arm (Acromion to Olecranon) is $\approx 13.8\text{ cm}$, Forearm (Olecranon to Stylion) is $\approx 14.7\text{ cm}$ ($13.8 + 14.7 = \mathbf{28.5\text{ cm}}$).

3. **Centroid Size ($CS$) Distribution Matching**:
   * Calibrating to 28.5 cm total arm length places the shape Centroid Size at **$CS \approx 41.2$**.
   * In the training dataset, median Centroid Size is **$CS = 41.4$** (Mean = $40.8$).
   * This guarantees that ontogenetic size decoupling on $\ln(CS)$ operates directly in the sweet spot of the trained allometric regression function without numerical extrapolation.

---

## 🚀 Running the API Server

```bash
cd Full_Demo_MC_ONC_Classification
python run_demo.py
```
* Server starts at: `http://127.0.0.1:8000`
* Interactive Swagger API Tester: `http://127.0.0.1:8000/docs`
* ReDoc Documentation: `http://127.0.0.1:8000/redoc`

---

## 📡 Mobile API Endpoints

### 1. `POST /api/v1/diagnose` (Multipart Form Upload)
Used when the mobile app uploads an image directly from the camera or gallery.

* **Content-Type**: `multipart/form-data`
* **Form Field**: `file` (Binary image file: JPEG/PNG)
* **Query Parameter**: `include_visuals=true` (optional, default `true`)

#### Mobile Example (`curl`):
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/diagnose?include_visuals=false" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@child_arm_photo.jpg"
```

#### Flutter Example (`dio`):
```dart
var formData = FormData.fromMap({
  'file': await MultipartFile.fromFile(imagePath, filename: 'child_arm.jpg'),
});
var response = await dio.post('http://<server-ip>:8000/api/v1/diagnose', data: formData);
print(response.data['triage_summary']['tier']); // GREEN, YELLOW, or RED
```

---

### 2. `POST /api/v1/diagnose-base64` (JSON Base64 Payload)
Used when mobile apps compress or encode the photo in memory before transmission.

* **Content-Type**: `application/json`
* **Request Body**:
```json
{
  "image_base64": "/9j/4AAQSkZJRgABAQAAAQ...",
  "client_id": "mobile_device_001",
  "include_visuals": false
}
```

---

### 3. `POST /api/v1/diagnose-path` (Local File Path)
Used for local testing or server batch evaluation of already-stored images.

* **Request Body**:
```json
{
  "image_path": "E:/data/child_photos/photo1.jpg",
  "include_visuals": false
}
```

---

### 4. `POST /api/v1/multimodal-resolve` (Borderline Yellow Escalation)
When a child falls into the 🟡 **Yellow Borderline Zone** (25% - 50% probability), the mobile app prompts the health worker to input the child's sex and physical tape MUAC (cm).

* **Request Body**:
```json
{
  "residuals_36": [-0.012, 0.004, ...],
  "gender": "M",
  "tape_muac_cm": 12.5
}
```

* **Response**:
```json
{
  "status": "success",
  "tape_muac_cm": 12.5,
  "multimodal_probability": 0.9992,
  "multimodal_triage": "RED (Confirmed Malnourished)",
  "clinical_confidence": ">89.7% Accuracy (Multimodal Protocol)"
}
```

---

### 5. Automatic Audit Logging & Quality Check (QA) Persistence
For every request processed by the frontline pipeline, an audit record is automatically saved under `audit_logs/<timestamp_audit_id>/` with the following validation files:

1. `01_original_image.jpg`: The original input photo.
2. `02_contour_overlay.png`: Visual arm overlay with contour lines and 22 numbered landmark pins.
3. `03_cartesian_plot.png`: Cartesian metric wireframe plot with upper/lower edges and calipers.
4. `04_side_by_side_quality_check.png`: Side-by-side composite comparison (Arm Overlay + Cartesian Plot).
5. `05_egat_attention_heatmap.png`: Cloud EGAT 17-node graph attention wireframe heatmap.
6. `landmarks_22.json`: Exact formatted 22-landmark list (pixel coordinates, Cartesian cm, colors, categories).
7. `audit_summary.json`: Comprehensive QA summary (execution time, biometrics, scale, QA status `PASS`/`FLAGGED`).

#### Audit Inspection Endpoints:
* `GET /api/v1/audit-logs?limit=50`: Lists recent audit logs with QA status and triage summary.
* `GET /api/v1/audit-logs/{audit_id}`: Retrieves full QA audit manifest for a session.
* `GET /api/v1/audit-logs/{audit_id}/file/{filename}`: Downloads or renders any specific audit file.


---

## 📋 Standard Mobile API Response Structure

```json
{
  "status": "success",
  "source": "photo.jpg",
  "timing_sec": {
    "phase1_vision": 0.539,
    "phase2_classification": 0.137,
    "total": 0.676
  },
  "triage_summary": {
    "tier": "YELLOW",
    "color_hex": "#f59e0b",
    "badge": "🟡 YELLOW: Moderate Risk (Borderline / Early-Stage MAM)",
    "category": "Borderline At-Risk (Early MAM / Watchlist)",
    "clinical_action": "Anganwadi Watchlist registration, nutrition counseling, and immediate manual tape MUAC check.",
    "malnutrition_probability": 0.3059,
    "is_yellow_borderline": true
  },
  "biometrics": {
    "arm_side_detected": "Left",
    "scale_px_per_cm": 25.4,
    "total_arm_length_cm": 28.5,
    "upper_arm_length_cm": 14.38,
    "forearm_length_cm": 14.2,
    "vision_estimated_muac_cm": 14.58,
    "centroid_size": 41.2
  },
  "cloud_egat": {
    "probability_egat": 0.9327,
    "primary_wasting_locus": "Proximal Forearm (L8-L7)",
    "calipers_attention": [
      {
        "caliper_id": 1,
        "name": "Proximal Shoulder Closure (L1-L4)",
        "pair": "L1 - L4",
        "attention_weight": 0.0892,
        "is_muac_site": false
      },
      {
        "caliper_id": 4,
        "name": "Mid-Upper Arm / MUAC Site (L16-L15)",
        "pair": "L16 - L15",
        "attention_weight": 0.1341,
        "is_muac_site": true
      }
    ]
  },
  "residuals_36": [ ... ],
  "landmarks_22": [ ... ],
  "visualizations": {
    "contour_overlay_base64": "data:image/png;base64,...",
    "cartesian_plot_base64": "data:image/png;base64,...",
    "egat_attention_heatmap_base64": "data:image/png;base64,..."
  }
}
```

---

## 🧪 Developer Verification & Testing

Developers can test the API immediately using preloaded clinical samples without uploading:
1. `GET /api/v1/samples` - Lists all 5 bundled sample arm photos.
2. `POST /api/v1/diagnose-sample/{filename}` - Runs end-to-end diagnosis on a sample photo.
3. Open `http://127.0.0.1:8000/docs` to test all endpoints interactively in the Swagger UI.

---

## 🤖 Frontline WhatsApp Chatbot & Loosely Coupled Modular Architecture

### 1. Architectural Blueprint
To empower frontline health workers (ASHA / Anganwadi workers) in rural or low-resource settings without requiring them to install or learn a separate native mobile application, the system natively integrates with **WhatsApp**.

```
[ ASHA / Frontline Worker ]
           │
           │  1. Takes photo of child's bare left arm (NO CHECKERBOARD)
           ▼
[ WhatsApp Mobile App ]
           │
           │  2. Sends image message via 4G/3G data
           ▼
[ Meta WhatsApp Cloud API / Twilio Gateway ]
           │
           │  3. Webhook Event (GET / POST /api/v1/whatsapp-webhook)
           ▼
[ PediaMorph AI FastAPI Server (run_demo.py) ]
    │
    ├─► Phase 1 Vision: MediaPipe Pose + Arm Segmentation (Scale via 28.5 cm canonical anchor)
    │
    ├─► Phase 2 Classification: Loosely Coupled Modular Engine (modular_models.py)
    │     │
    │     ├── [Tier 1 Primary]: 40% Shrinkage LDA + 60% XGBoost Ensemble (Sub-second triage)
    │     └── [Tier 2 Backup]:  17-Node Cloud EGAT Graph Attention Network (Secondary cross-check)
    │
    ├─► Optional Escalation: Multimodal 38D XGBoost (if worker replies with "MUAC <value>")
    │
    ▼
[ Formatted Clinical Triage Card with Badge & Action Guide ]
           │
           │  4. Outbound WhatsApp Message (< 1 sec)
           ▼
[ ASHA Worker WhatsApp Screen: 🔴 RED / 🟡 YELLOW / 🟢 GREEN ]
```

### 2. Loosely Coupled Modular Models (`modular_models.py`)
All predictors implement a unified `BasePredictor` interface, allowing complete architectural independence:

| Predictor Class | Model Type | Core Specialty & Metrics | Standalone Inspection Key |
| :--- | :--- | :--- | :--- |
| `ShrinkageLDAPredictor` | Ledoit-Wolf Regularized Linear Discriminant | **High Sensitivity ($72.3\%$)** for subtle soft-tissue wasting detection | `results["individual_models"]["shrinkage_lda"]` |
| `XGBoostShapePredictor` | Gradient Boosted Decision Trees on 36D | **High Specificity ($84.3\%$)** suppressing false alarms on healthy children | `results["individual_models"]["xgboost_shape"]` |
| `CloudEGATPredictor` | 17-Node Edge-Augmented Graph Attention Net | **Secondary Backup Verification**: 9-caliper attention heatmaps & wasting locus | `results["individual_models"]["cloud_egat"]` |
| `OfflineEnsemblePredictor` | Soft-Voting Blend ($0.40 \times \text{LDA} + 0.60 \times \text{XGB}$) | **Primary Frontline Triage**: Calibrated `[0.20, 0.50]` cutoffs ($>88.5\%$ real sensitivity) | `results["grouped_models"]["offline_ensemble_lda_xgboost"]` |
| `MultimodalResolver` | 38D Late-Fusion XGBoost Classifier | **Secondary Escalation**: Resolves borderline Yellow cases with tape MUAC ($>89.7\%$ accuracy) | `results["multimodal_confirmation"]` |

### 3. One-Line Toggles: Add, Remove, or Switch Models On Demand
You can instantly enable, disable, or remove Cloud EGAT or change the primary decision engine with a single line of code:

#### A. Toggle EGAT Backup (Global Switch in `classifier_engine.py`):
```python
ENABLE_EGAT_BACKUP: bool = True  # Set to False to completely remove EGAT with 0ms graph overhead
```

#### B. Toggle EGAT Per-Request (In Python Code):
```python
results = engine.diagnose_full_pipeline(landmarks, enable_egat=False)
```

#### C. Change Primary Decision Model (Global Switch in `modular_models.py`):
```python
DEFAULT_PRIMARY_MODEL: str = "ensemble"  # Options: 'ensemble', 'lda', 'xgboost', 'egat'
```

### 4. Evaluating Individual vs. Grouped Performance
Calling `diagnose_full_pipeline()` returns both standalone model diagnostics and grouped ensemble metrics side-by-side:

```python
from classifier_engine import MalnutritionClassificationEngine

engine = MalnutritionClassificationEngine(enable_egat_backup=True)
res = engine.diagnose_full_pipeline(raw_landmarks)

# 1. Standalone Individual Performance
p_lda  = res["individual_models"]["shrinkage_lda"]["probability_mc"]
p_xgb  = res["individual_models"]["xgboost_shape"]["probability_mc"]
p_egat = res["individual_models"]["cloud_egat"]["probability_egat"]

# 2. Grouped Ensemble Performance
p_ens  = res["grouped_models"]["offline_ensemble_lda_xgboost"]["probability_ensemble"]
tier   = res["grouped_models"]["offline_ensemble_lda_xgboost"]["triage_tier"]

# 3. Model Concordance / Secondary Verification
agreed = res["models_in_agreement"]  # True if Primary Triage and Backup EGAT concur
```

### 5. Running the WhatsApp Service & Local CLI Simulator

#### Step 1: Start the API Gateway
```bash
python run_demo.py
```
*Listens on `http://127.0.0.1:8000` with active Meta Webhook endpoints at `/api/v1/whatsapp-webhook`.*

#### Step 2: Run the Local WhatsApp Chatbot Simulator
Test the complete ASHA worker experience locally without needing Meta developer credentials:
```bash
python whatsapp_bot_service.py
```

#### Sample WhatsApp Chatbot Output:
```
📋 *PEDIAMORPH AI - CHILD NUTRITION TRIAGE*
━━━━━━━━━━━━━━━━━━━━━━━━━━
🆔 *Session:* `SESSION-20261026-001`
🎯 *Triage Status:*
🔴 RED: Critical Risk / Severe Acute Malnutrition (SAM)

📊 *Primary Ensemble Risk:* *91.8%*
   • Shrinkage LDA (Sensitivity): 99.4%
   • XGBoost Shape (Specificity): 86.8%

🔬 *Backup Cloud EGAT Cross-Check:* 10.6%
📍 *Primary Wasting Indicator:* Mid Forearm (L6-L5)

📏 *Estimated Dimensions:*
• Vision Estimated MUAC: ~11.2 cm
• Total Arm Length: 21.4 cm

🩺 *Clinical Action Guide for ASHA / Health Worker:*
Immediate urgent clinical referral to Primary Health Centre (PHC) or Nutrition Rehabilitation Centre (NRC).

━━━━━━━━━━━━━━━━━━━━━━━━━━
_PediaMorph AI Frontline Pediatric Health Assistant_
```

---

## 🛡️ Pre-Flight Quality Guard & Arm Posture Inspection Module (`image_quality_guard.py`)

To prevent inaccurate clinical diagnoses or pipeline errors caused by unstandardized, blurry, or improperly positioned child photographs, PediaMorph AI incorporates a **loosely coupled pre-flight quality guard** before images reach the Phase 2 classification engine.

### 1. Clinical Rationale: Why Pre-Flight Inspection is Essential
The downstream classifier relies on **Procrustes Superimposition (fGPA)** and **Allometric Residual Modeling** over 22 anatomical Cartesian landmarks. If:
- The child's left arm is **bent at the elbow** (e.g., $145^\circ$ or $125^\circ$),
- The **shoulder landmark drifts inward toward the neck** (due to head turning or clothing),
- **Adult fingers grip or squeeze the wrist**, collapsing the wrist landmarks into a pinched triangle, or
- **Jewelry, sacred threads, or black bangles** disrupt the skin boundary,

the Procrustes registration artificially warps and stretches coordinates, creating large artificial shape residuals ($\text{residuals}_{36}$). This can result in **false positives** (e.g., misdiagnosing a normal child as SAM / Red) or pipeline failures.

The pre-flight guard acts as an **intelligent clinical safety gate**: invalid photos are intercepted immediately and rejected with clear, actionable retake instructions returned to the user or frontline health worker before calling the classifier.

---

### 2. Multi-Stage Verification Architecture

```
[Raw Child Photo Input]
         │
         ▼
[Stage 1: Optical Quality Verification]
  • Sharpness / Blur: Laplacian variance (Var(∇²I) ≥ 340 required)
  • Illumination & Contrast: Rejects underexposed (< 45) or glare (> 232)
         │ PASS
         ▼
[Stage 2: Arm Straightness & Geometry]
  • Elbow Stretch Angle: θ_elbow ≥ 162.0° required (rejects bent arms)
  • Geometric Stretch Ratio: Dist(Shoulder, Wrist) / (Upper + Forearm) ≥ 0.960
         │ PASS
         ▼
[Stage 3: Joint Positioning Integrity Trio]
  • Shoulder (L21, L1, L22): Anchored to true acromion; flags neck displacement (sum-to-total ratio ≤ 1.15) & necklace cord occlusions
  • Elbow (L2, L18, L17): Anatomical upper/forearm ratio strictly bounded (0.80 – 1.32) & flags contour cutting under skin crease
  • Wrist (L3, L19, L20): Caliper width ratio (0.45 – 1.15), flags adult fingers gripping wrist & black bangle occlusions
         │ PASS
         ▼
[Stage 4: Contour Skin-Edge Adherence]
  • Samples boundary normals along upper and lower contours
  • Verifies alignment with infant YCrCb skin mask (≥ 65% adherence)
  • Ensures complete perpendicular scan resolution (≥ 19/22 landmarks)
         │
    ┌────┴────────────────────────────────┐
    │                                     │
[PASS: All Standards Met]       [FAIL: Quality Issue Detected]
    │                                     │
    ▼                                     ▼
[Phase 2 Dual-Tier Classifier]  [Return Rejection Response Immediately]
• 40% LDA + 60% XGBoost          • Status: "quality_rejected"
• Cloud EGAT Attention            • Detailed list of critical failures
• Real-time LIME Explainer        • Specific customer-friendly retake instructions
• Clinical Triage Badge           • Annotated diagnostic image overlay with red callouts
```

---

### 3. Detected Flaws & Clinical Failure Codes

| Failure Code | Observed Cause | Clinical Impact | Customer Retake Instruction |
| :--- | :--- | :--- | :--- |
| `ARM_BENT_AT_ELBOW` | Child's left arm is bent at elbow ($\theta < 162^\circ$) | Warps Procrustes superimposition and distorts caliper lengths | *"Gently hold the child's left arm completely straight and flat along the surface. Do not allow the elbow to bend."* |
| `IMAGE_BLURRY` | Camera motion or out-of-focus defocus ($\text{Var}(\nabla^2 I) < 340$) | Blurs skin edges and introduces perpendicular scanning jitter | *"Hold the camera steady in good lighting, tap the child's arm on the phone screen to focus, and retake the photo."* |
| `SHOULDER_LANDMARK_DISPLACED_NECK` | Shoulder marker shifted inward onto neck/ear | Artificially doubles upper-arm segment length | *"Ensure child's neck and chest are unobstructed. The shoulder joint (deltoid) must be clearly visible and not covered by clothing."* |
| `SHOULDER_JEWELRY_CORD_OCCLUSION` | Amulets, strings, or necklaces crossing shoulder | Pinches shoulder semi-landmarks ($L_1 \to L_{22}$) | *"Please remove necklaces, sacred threads, amulets, or strings from the child's neck and shoulder before taking the photo."* |
| `ELBOW_LANDMARK_DISPLACED` | Upper-arm to forearm ratio outside $[0.80, 1.32]$ | Mislocates mid-upper arm / MUAC measurement locus | *"Ensure child's arm is fully uncovered from shoulder to wrist so the elbow joint line is clearly identifiable."* |
| `ELBOW_CONTOUR_UNDERSEGMENTED` | Contour cut under the true skin edge at the elbow crease | Artificially thins the elbow epicondyle width | *"Ensure good lighting with no deep shadows under the elbow crease and keep the arm flat."* |
| `WRIST_COLLAPSED_ADULT_FINGERS` | Adult fingers holding/squeezing wrist ($W_{\text{wrist}} / W_{\text{fore}} < 0.45$) | Distorts distal termination into a pinched triangle | *"Do not hold or squeeze the child's wrist with fingers. Lay the wrist open and flat against the mat; support gently only by fingertips if needed."* |
| `WRIST_BANGLE_OR_JEWELRY_OCCLUSION` | Wrist caliper abnormally wide ($> 1.15 \times$ forearm) | Distorts distal caliper width | *"Remove all bracelets, threads, and jewelry from the child's wrist before taking the photo."* |
| `BLACK_BANGLE_DETECTED` | High-contrast dark band ($> 28\%$ dark pixels) across wrist | Breaks contour tracking and distorts forearm width | *"Please remove all black bangles and bracelets from the child's left wrist."* |
| `CONTOUR_PERPENDICULAR_SCAN_FAILED` | Fewer than 19 of 22 landmarks resolved | Incomplete shape geometry for 36D residual calculation | *"Ensure clean arm background with good illumination. Check for necklaces, sleeves, or shadows breaking the arm contour."* |

---

### 4. Standalone Python Usage

```python
from image_quality_guard import inspect_image_and_contour_quality
from contour_extractor import extract_arm_contour_and_landmarks

# 1. Run contour extraction
p1 = extract_arm_contour_and_landmarks("child_arm_photo.jpg")

# 2. Inspect image quality and arm posture
quality = inspect_image_and_contour_quality("child_arm_photo.jpg", contour_result=p1)

if not quality["is_valid"]:
    print(f"❌ Image Rejected! Score: {quality['quality_score']}/100")
    for failure in quality["critical_failures"]:
        print(f"  • [{failure['code']}]: {failure['message']}")
        print(f"    Action: {failure['customer_instruction']}")
else:
    print(f"✅ Image Approved! Score: {quality['quality_score']}/100")
```

---

### 5. API Response Payloads

#### When Image is Rejected (Quality Gate Intercepts Before Classifier):
```json
{
  "status": "quality_rejected",
  "is_valid": false,
  "source": "photo_upload.jpg",
  "message": "Image quality or child arm posture does not meet clinical standards. Photo rejected before classification to prevent false diagnosis.",
  "quality_score": 35.0,
  "critical_failures": [
    {
      "code": "ARM_BENT_AT_ELBOW",
      "severity": "CRITICAL",
      "message": "Child's left arm is bent at 149.4° (required straight stretch >= 162°). Stretch ratio: 0.967.",
      "customer_instruction": "Gently hold the child's left arm completely straight and flat along the surface. Do not allow the elbow to bend."
    },
    {
      "code": "ELBOW_LANDMARK_DISPLACED",
      "severity": "CRITICAL",
      "message": "Elbow landmark position is anatomically abnormal (Upper-arm / Forearm ratio: 0.61, valid range: 0.80-1.32).",
      "customer_instruction": "Ensure child's arm is fully uncovered from shoulder to wrist so the elbow joint line is clearly identifiable."
    }
  ],
  "warnings": [],
  "customer_instructions": [
    "Gently hold the child's left arm completely straight and flat along the surface. Do not allow the elbow to bend.",
    "Ensure child's arm is fully uncovered from shoulder to wrist so the elbow joint line is clearly identifiable."
  ],
  "quality_metrics": {
    "blur_laplacian_var": 1896.0,
    "mean_brightness": 126.9,
    "contrast_std": 69.0,
    "elbow_angle_deg": 149.4,
    "arm_stretch_ratio": 0.967,
    "upper_to_forearm_ratio": 0.61,
    "sum_to_total_ratio": 1.03
  },
  "visualizations": {
    "diagnostic_overlay_base64": "data:image/png;base64,..."
  }
}
```

#### When Image Passes Quality Inspection:
The classifier runs as usual, and `quality_inspection` metrics are embedded in the response alongside the diagnostic triage summary:
```json
{
  "status": "success",
  "triage_summary": {
    "tier": "GREEN",
    "badge": "🟢 GREEN: Low Risk / Normal Nutritional Status",
    "malnutrition_probability": 0.082
  },
  "quality_inspection": {
    "status": "passed",
    "quality_score": 100.0,
    "warnings": [],
    "metrics": {
      "blur_laplacian_var": 605.2,
      "elbow_angle_deg": 171.7,
      "arm_stretch_ratio": 0.997,
      "upper_to_forearm_ratio": 1.01
    }
  }
}
```

---

### 6. Frontline Bilingual Support (English + Telugu / తెలుగు)
To overcome language barriers for ASHA and Anganwadi workers in Andhra Pradesh and Telangana, all quality rejection issues, retake guidance, and clinical triage responses are formatted in **Bilingual English + Telugu side-by-side**. 

Workers immediately see what problem occurred and exactly how to fix it in their native language:
* **Arm Bent at Elbow**: `ARM_BENT_AT_ELBOW / చేయి మోచేయి వద్ద వంగి ఉంది`
  * EN: *Gently hold child's left arm completely straight and flat along the surface.*
  * TE: *పిల్లల ఎడమ చేతిని మోచేయి వంచకుండా పూర్తిగా తిన్నగా చాపపై లేదా నేలపై ఉంచి ఫోటో తీయండి.*
* **Black Bangles / Charms**: `BLACK_BANGLE_DETECTED / నల్ల గాజు / దారం గుర్తించబడింది`
  * EN: *Please remove all black bangles and bracelets from the child's left wrist.*
  * TE: *పిల్లల ఎడమ చేతికి ఉన్న నల్ల గాజులు, దారాలు తీసివేసి మళ్లీ ఫోటో తీయండి.*
* **Necklaces / Sacred Threads**: `SHOULDER_JEWELRY_CORD_OCCLUSION / మెడలోని తాళ్లు / గొలుసులు భుజంపై పడ్డాయి`
  * EN: *Remove necklaces, sacred threads, amulets, or strings from child's neck and shoulder.*
  * TE: *మెడలోని గొలుసులు, నల్లతాళ్లు, తావీదులు భుజంపై పడకుండా తీసివేసి ఫోటో తీయండి.*

---

### 7. Frontline Worker Authentication & Daily Quota Guard (`worker_auth.py`)

To prevent unauthorized strangers from texting the WhatsApp number and consuming Twilio / Meta API budgets and server resources, a self-service authentication layer is integrated directly into the webhook.

#### 1. Self-Service 4-Digit PIN Activation (`JOIN 1234`)
- When any unregistered phone number texts or sends a photo, they receive:
  ```text
  🔒 ACCESS RESTRICTED / అనుమతి అవసరం
  This service is reserved for registered frontline ASHA / Anganwadi staff.
  ఈ సేవ రిజిస్టర్ అయిన ఆశా / అంగన్‌వాడీ కార్యకర్తలకు మాత్రమే.

  Please reply with your secret Clinic PIN to activate:
  👉 JOIN 1234
  ```
- Once the worker replies `JOIN 1234` (or just `1234`), their phone number is permanently activated and saved in `authorized_workers.json`.
- Zero AI vision inferences are executed for unregistered numbers.

#### 2. Daily Rate Limiting (Configurable Quota)
- Default: **25 scans per day** per registered worker (configurable via `.env`).
- If a worker exceeds 25 scans in a day, the bot returns:
  ```text
  ⏳ DAILY SCAN LIMIT REACHED / రోజువారీ పరిమితి ముగిసింది
  You have used your maximum quota of 25 scans for today.
  Your daily quota will reset tomorrow morning at 00:00.
  ```

#### 3. Configuration in `.env`
```ini
# Frontline Worker Authentication & Rate Limiting
REGISTRATION_PIN=12345
DAILY_SCAN_LIMIT=25
ADMIN_API_KEY=poshan_admin_secret_2026
```

#### 4. Deregistering Workers & Discontinuing Chat via REST API
Administrators can remove any registered phone number to immediately discontinue their chat and revoke access to child screening.

- **Endpoints**: `DELETE /api/v1/auth/workers/{phone}` or `POST /api/v1/auth/workers/remove` (also aliased under `/api/v1/admin/workers/...`)
- **Headers**: `X-Admin-Key: <ADMIN_API_KEY>` (or query param `?admin_key=...`)

```bash
# Method A: Via DELETE request (Path Parameter)
curl -X DELETE "https://pediamorph-ai.tech/api/v1/auth/workers/918349819726" \
     -H "X-Admin-Key: poshan_admin_secret_2026"

# Method B: Via POST request (JSON Body)
curl -X POST "https://pediamorph-ai.tech/api/v1/auth/workers/remove" \
     -H "Content-Type: application/json" \
     -H "X-Admin-Key: poshan_admin_secret_2026" \
     -d '{"phone": "918349819726"}'
```

**JSON Response:**
```json
{
  "status": "success",
  "message": "Phone number '918349819726' was successfully removed. Chat is discontinued until re-registered with PIN.",
  "phone": "918349819726",
  "re_registration_required": "JOIN 12345"
}
```

#### 5. Listing All Registered Workers
To audit which frontline workers are registered and their usage statistics:
```bash
curl -X GET "https://pediamorph-ai.tech/api/v1/auth/workers" \
     -H "X-Admin-Key: poshan_admin_secret_2026"
```

#### 6. Direct Admin Registration via API
```bash
curl -X POST "https://pediamorph-ai.tech/api/v1/auth/workers/register" \
     -H "Content-Type: application/json" \
     -H "X-Admin-Key: poshan_admin_secret_2026" \
     -d '{"phone": "918349819726"}'
```

#### 7. Self-Service WhatsApp Logout & PIN Re-Registration
- **WhatsApp Self-Deregistration**: Frontline staff can text **`UNREGISTER`** or **`LOGOUT`** directly in the WhatsApp chat. The bot immediately deactivates their account and confirms deactivation.
- **Immediate Discontinuation**: Once removed (via API or WhatsApp command), any further message from that number triggers the `Access Restricted` lock prompt (`🔒 అనుమతి అవసరం`), completely discontinuing all screening features.
- **Re-Registration**: To resume access, the worker must send **`JOIN 12345`** (or `JOIN 1234`) with the secret PIN.



