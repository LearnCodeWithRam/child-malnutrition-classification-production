"""
image_quality_guard.py
======================
Frontline Image Quality, Child Arm Posture, Joint Positioning & Contour Adherence Inspection Guard.

Designed for PediaMorph AI:
Performs comprehensive pre-flight verification before images are passed to the
Phase 2 Dual-Tier Morphometric Classification Engine.

Bilingual Support (English + Telugu / తెలుగు) for frontline ASHA / Anganwadi workers in AP & Telangana.

Verifications:
1. Optical Quality: Sharpness / Blur (Laplacian variance), Exposure (under/over), Contrast.
2. Arm Straightness: 2D Elbow bend angle (straight stretch >= 162 deg required).
3. Joint Positioning Trio:
   - Shoulder (L21, L1, L22): Anchored to true anatomical acromion; detects inward neck displacement
     and necklace/cord occlusions crossing L1/L21/L22.
   - Elbow (L2, L18, L17): Centered at joint crease; checks upper-arm-to-forearm anatomical ratio (0.80 - 1.32)
     and detects contour under-segmentation (cutting under skin crease at elbow).
   - Wrist (L3, L19, L20): Caliper width, anti-collapse test (detects adult fingers squeezing wrist),
     wrist orientation (pronation/supination), and bangle/jewelry occlusion.
4. Contour Skin-Edge Adherence: Verifies that the extracted contour tightly follows the true skin boundary
   and does not cut under the skin crease (under-segmentation) or leak into checkerboard tiles/clothes.
5. Customer-Friendly Guidance: Returns structured diagnostic rejection reasons and clear retake instructions
   in both English and Telugu (తెలుగు).
"""

import os
import cv2
import numpy as np
import base64
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===========================================================================
# 🎛️ Configurable Clinical Quality Thresholds
# ===========================================================================
# 1. Optical Quality
MIN_BLUR_LAPLACIAN_VAR: float = 340.0   # Below this = blurry / defocus / motion blur
MIN_BRIGHTNESS: float = 45.0            # Underexposed if below
MAX_BRIGHTNESS: float = 232.0           # Overexposed / glare if above
MIN_CONTRAST_STD: float = 26.0          # Washed-out / low dynamic range if below

# 2. Arm Straightness & Geometry
MIN_ELBOW_ANGLE_DEG: float = 162.0      # Arm must be stretched straight
MIN_STRETCH_RATIO: float = 0.960        # Shoulder-Wrist distance / (Upper + Forearm length)
MIN_LANDMARK_COUNT: int = 19            # Below this indicates perpendicular scanning failed

# 3. Joint Positioning Bounds
MIN_UPPER_FOREARM_RATIO: float = 0.80   # Upper arm cm / Forearm cm
MAX_UPPER_FOREARM_RATIO: float = 1.32   # Upper arm cm / Forearm cm
MAX_SUM_TO_TOTAL_RATIO: float = 1.15    # (Upper + Forearm) / Total - flags shoulder at neck
MIN_ELBOW_TO_UPPER_WIDTH_RATIO: float = 0.65  # Flags contour cutting under skin at elbow
MIN_SHOULDER_TO_PROX_WIDTH_RATIO: float = 0.55 # Flags necklace cords crossing shoulder

# 4. Wrist Positioning & Anti-Collapse / Jewelry Detection
MIN_WRIST_WIDTH_PX: float = 10.0        # Collapsed wrist line if below
MIN_WRIST_TO_FOREARM_RATIO: float = 0.45  # Pinched by adult fingers if below
MAX_WRIST_TO_FOREARM_RATIO: float = 1.35  # Gross anatomical outlier / loose bangle if above (relaxed from 1.15)
SUSPICIOUS_WRIST_RATIO_MIN: float = 1.22  # Between 1.22 and 1.35 requires optical confirmation (dark pixels)
MIN_BANGLE_DARK_PIXEL_RATIO: float = 0.18 # Dark pixel ratio threshold for black thread / bangle

# 5. Contour Skin Adherence
MIN_SKIN_CONTOUR_ADHERENCE: float = 0.65  # Fraction of contour points matching true skin edge

# 6. Overall Quality Score Pass Threshold (>= 80% required to pass)
MIN_QUALITY_SCORE_THRESHOLD: float = 80.0


# ===========================================================================
# 🌐 Telugu Translation Dictionary (తెలుగు అనువాదాలు)
# ===========================================================================
TELUGU_TRANSLATIONS: Dict[str, Dict[str, str]] = {
  "ARM_BENT_AT_ELBOW": {
    "title": "చేయి మోచేయి వద్ద వంగి ఉంది / Arm Bent at Elbow",
    "message": "పిల్లల ఎడమ చేయి మోచేయి వద్ద వంగి ఉంది. కొలత సరిగ్గా రావడానికి చేయి తిన్నగా ఉండాలి. / The child's left arm is bent at the elbow. The arm must be straight for accurate measurement.",
    "instruction": "పిల్లల ఎడమ చేతిని మోచేయి వంచకుండా పూర్తిగా తిన్నగా చాపపై లేదా నేలపై ఉంచి ఫోటో తీయండి. / Keep the child's left arm fully straight on a mat or floor and retake the photo."
  },

  "IMAGE_BLURRY": {
    "title": "ఫోటో మసకగా ఉంది / Blurry Photo",
    "message": "ఫోటో మసకగా వచ్చింది. కెమెరా కదలడం లేదా ఫోకస్ సరిగా లేకపోవడం కారణం కావచ్చు. / The photo is blurry due to camera movement or poor focus.",
    "instruction": "మంచి వెలుతురులో కెమెరాను స్థిరంగా పట్టుకుని, పిల్లల చేతిపై ఫోకస్ చేసి ఫోటో తీయండి. / Hold the camera steady in good light, focus on the child's arm, and retake the photo."
  },

  "IMAGE_UNDEREXPOSED": {
    "title": "ఫోటో చీకటిగా ఉంది / Photo Too Dark",
    "message": "ఫోటో చాలా చీకటిగా ఉంది. చేయి స్పష్టంగా కనిపించడం లేదు. / The image is too dark and the arm is not clearly visible.",
    "instruction": "మంచి వెలుతురు లేదా పగటి వెలుగు ఉన్న చోట ఫోటో తీయండి. / Retake the photo in daylight or a well-lit area."
  },

  "IMAGE_OVEREXPOSED": {
    "title": "ఎక్కువ వెలుతురు ఉంది / Too Much Light",
    "message": "ఫోటోపై ఎక్కువ వెలుతురు లేదా మెరుపు పడింది. / Excessive light or glare is affecting the image.",
    "instruction": "నేరుగా ఫ్లాష్ లేదా ఎండ వెలుతురు పడకుండా ఫోటో తీయండి. / Avoid direct flash or strong sunlight when taking the photo."
  },

  "LOW_CONTRAST": {
    "title": "చేయి స్పష్టంగా కనిపించడం లేదు / Low Contrast",
    "message": "చేయి మరియు నేపథ్యం మధ్య తేడా తక్కువగా ఉంది. / The arm does not clearly stand out from the background.",
    "instruction": "సాదా రంగు ఉన్న చాప లేదా నేపథ్యంపై చేయిని ఉంచి ఫోటో తీయండి. / Place the arm on a plain-colored surface and retake the photo."
  },

  "PRIMARY_JOINTS_MISSING": {
    "title": "భుజం, మోచేయి లేదా మణికట్టు కనిపించడం లేదు / Arm Joints Not Visible",
    "message": "భుజం, మోచేయి లేదా మణికట్టు స్పష్టంగా కనిపించడం లేదు. / The shoulder, elbow, or wrist is not clearly visible.",
    "instruction": "భుజం నుండి మణికట్టు వరకు ఎడమ చేయి మొత్తం ఫోటోలో కనిపించేలా తీయండి. / Ensure the entire left arm from shoulder to wrist is visible in the photo."
  },

  "SHOULDER_LANDMARK_DISPLACED_NECK": {
    "title": "భుజం భాగం స్పష్టంగా కనిపించడం లేదు / Shoulder Not Clearly Visible",
    "message": "భుజం భాగం బట్టలు లేదా ఇతర అడ్డంకుల వల్ల సరిగా కనిపించడం లేదు. / The shoulder area is not clearly visible due to clothing or obstruction.",
    "instruction": "భుజం పూర్తిగా కనిపించేలా బట్టలు లేదా ఇతర అడ్డంకులను తొలగించి ఫోటో తీయండి. / Ensure the shoulder is fully visible before taking the photo."
  },

  "SHOULDER_JEWELRY_CORD_OCCLUSION": {
    "title": "తాళ్లు లేదా గొలుసులు అడ్డుగా ఉన్నాయి / Chains or Threads Covering Shoulder",
    "message": "మెడలోని గొలుసులు, తాళ్లు లేదా తావీదులు భుజం భాగాన్ని కప్పుతున్నాయి. / Chains, threads, or amulets are covering the shoulder area.",
    "instruction": "మెడలోని తాళ్లు, గొలుసులు లేదా తావీదులు తొలగించి ఫోటో తీయండి. / Remove chains, threads, or amulets before taking the photo."
  },

  "ELBOW_LANDMARK_DISPLACED": {
    "title": "మోచేయి భాగం స్పష్టంగా కనిపించడం లేదు / Elbow Not Clearly Visible",
    "message": "మోచేయి ప్రాంతం సరిగా గుర్తించబడలేదు. / The elbow region could not be identified correctly.",
    "instruction": "చేయి మొత్తం భుజం నుండి మణికట్టు వరకు స్పష్టంగా కనిపించేలా ఫోటో తీయండి. / Ensure the full arm from shoulder to wrist is clearly visible."
  },

  "ELBOW_CONTOUR_UNDERSEGMENTED": {
    "title": "మోచేయి వద్ద నీడలు ఉన్నాయి / Shadows Around Elbow",
    "message": "మోచేయి భాగంలో నీడలు ఉండటం వల్ల చేయి అంచులు స్పష్టంగా కనిపించడం లేదు. / Shadows around the elbow are affecting arm detection.",
    "instruction": "మంచి వెలుతురులో చేయిని తిన్నగా ఉంచి మళ్లీ ఫోటో తీయండి. / Keep the arm straight and retake the photo in good lighting."
  },

  "WRIST_COLLAPSED_ADULT_FINGERS": {
    "title": "మణికట్టును పట్టుకున్నారు / Wrist Being Held",
    "message": "మణికట్టును చేతితో పట్టుకోవడం వల్ల చేయి ఆకారం మారింది. / The wrist is being held, affecting the arm shape.",
    "instruction": "మణికట్టును పట్టుకోకుండా చేయిని సహజంగా చాపపై ఉంచండి. / Do not hold the wrist. Let the arm rest naturally on the mat."
  },

  "WRIST_BANGLE_OR_JEWELRY_OCCLUSION": {
    "title": "మణికట్టుకు గాజులు లేదా కడియాలు ఉన్నాయి / Bangles or Jewelry on Wrist",
    "message": "గాజులు, కడియాలు లేదా తాళ్లు కొలతను ప్రభావితం చేస్తున్నాయి. / Bangles, bracelets, or threads are affecting the measurement.",
    "instruction": "మణికట్టుపై ఉన్న గాజులు, కడియాలు మరియు తాళ్లు తొలగించి ఫోటో తీయండి. / Remove all bangles, bracelets, and threads from the wrist before taking the photo."
  },

  "BLACK_BANGLE_DETECTED": {
    "title": "నల్ల గాజు లేదా దారం గుర్తించబడింది / Black Bangle or Thread Detected",
    "message": "నల్ల గాజు లేదా దారం చేయి కొలతకు అడ్డంకిగా ఉంది. / A black bangle or thread is interfering with arm measurement.",
    "instruction": "నల్ల గాజులు లేదా దారాలు తొలగించి మళ్లీ ఫోటో తీయండి. / Remove the black bangle or thread and retake the photo."
  },

  "CONTOUR_PERPENDICULAR_SCAN_FAILED": {
    "title": "చేయి అంచులు పూర్తిగా గుర్తించబడలేదు / Arm Boundary Not Detected",
    "message": "చేయి మొత్తం స్పష్టంగా గుర్తించలేకపోయాము. / The arm boundary could not be detected correctly.",
    "instruction": "మంచి వెలుతురులో, చేయి చుట్టూ అడ్డంకులు లేకుండా ఫోటో తీయండి. / Retake the photo in good lighting with no obstructions around the arm."
  },

  "CONTOUR_SKIN_EDGE_DEVIATION": {
    "title": "చేయి అంచులు స్పష్టంగా లేవు / Arm Edge Not Clear",
    "message": "నేపథ్యం లేదా నీడల కారణంగా చేయి అంచులు స్పష్టంగా కనిపించడం లేదు. / The arm edges are unclear due to shadows or background patterns.",
    "instruction": "సాదా రంగు నేపథ్యంపై చేయిని ఉంచి ఫోటో తీయండి. / Place the arm on a plain background and retake the photo."
  },

  "CONTOUR_EXTRACTION_FAILED": {
    "title": "చేయి గుర్తించబడలేదు / Arm Not Detected",
    "message": "ఫోటోలో పిల్లల చేయి స్పష్టంగా కనిపించడం లేదు. / The child's arm could not be clearly detected in the photo.",
    "instruction": "ఎడమ చేయి మొత్తం కనిపించేలా మళ్లీ ఫోటో తీయండి. / Retake the photo ensuring the entire left arm is visible."
  }
}



def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def _load_image_bgr(image_input: Union[bytes, str, Path, np.ndarray]) -> np.ndarray:
    """Safely ingests image input into a BGR numpy array."""
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Unable to read image from path: {image_input}")
        return img
    elif isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unable to decode image from bytes.")
        return img
    elif isinstance(image_input, np.ndarray):
        return image_input.copy()
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")


def _make_failure(code: str, message_en: str, instruction_en: str) -> Dict[str, str]:
    """Helper to package failure info with dual English + Telugu translations."""
    te_info = TELUGU_TRANSLATIONS.get(code, {})
    message_te = te_info.get("message_te") or te_info.get("message", "")
    instruction_te = te_info.get("instruction_te") or te_info.get("instruction", "")
    title_te = te_info.get("title_te") or te_info.get("title", "")

    return {
        "code": code,
        "severity": "CRITICAL",
        "title_te": title_te,
        "message": message_en,
        "message_te": message_te,
        "customer_instruction": instruction_en,
        "customer_instruction_te": instruction_te,
        "bilingual_instruction": f"{instruction_en} / {instruction_te}" if instruction_te else instruction_en
    }


def inspect_image_and_contour_quality(
    image_input: Union[bytes, str, Path, np.ndarray],
    contour_result: Optional[dict] = None,
    min_quality_score: float = MIN_QUALITY_SCORE_THRESHOLD
) -> dict:
    """
    Executes complete multi-stage inspection on raw image and extracted contour/landmarks.
    Includes bilingual English + Telugu guidance for frontline workers.
    """
    image_bgr = _load_image_bgr(image_input)
    h_img, w_img = image_bgr.shape[:2]

    critical_failures: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    metrics: Dict[str, Union[float, int, bool, str]] = {}

    # -----------------------------------------------------------------------
    # Stage 1: Optical Quality Checks (Blur, Lighting, Contrast)
    # -----------------------------------------------------------------------
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))
    std_contrast = float(np.std(gray))

    metrics["blur_laplacian_var"] = round(lap_var, 2)
    metrics["mean_brightness"] = round(mean_brightness, 2)
    metrics["contrast_std"] = round(std_contrast, 2)

    # 1.1 Blur / Defocus
    if lap_var < MIN_BLUR_LAPLACIAN_VAR:
        critical_failures.append(_make_failure(
            "IMAGE_BLURRY",
            f"Image sharpness is low ({lap_var:.1f} vs required {MIN_BLUR_LAPLACIAN_VAR:.0f}). Camera motion or out-of-focus blur detected.",
            "Hold camera steady in good light, tap child's arm on screen to focus, and retake photo."
        ))
    elif lap_var < MIN_BLUR_LAPLACIAN_VAR * 1.35:
        warnings.append({
            "code": "IMAGE_SLIGHT_BLUR",
            "severity": "WARNING",
            "message": f"Image is slightly soft ({lap_var:.1f}). Recommend steadying camera for optimal precision."
        })

    # 1.2 Brightness / Illumination
    if mean_brightness < MIN_BRIGHTNESS:
        critical_failures.append(_make_failure(
            "IMAGE_UNDEREXPOSED",
            f"Photograph is too dark (brightness: {mean_brightness:.1f} / 255).",
            "Move to a well-lit area or use daylight. Do not capture in dark or shadowy rooms."
        ))
    elif mean_brightness > MAX_BRIGHTNESS:
        critical_failures.append(_make_failure(
            "IMAGE_OVEREXPOSED",
            f"Photograph has excessive glare / overexposure (brightness: {mean_brightness:.1f} / 255).",
            "Avoid direct flash reflection or harsh glare directly onto the child's skin."
        ))

    # 1.3 Contrast
    if std_contrast < MIN_CONTRAST_STD:
        warnings.append({
            "code": "LOW_CONTRAST",
            "severity": "WARNING",
            "message": f"Low contrast detected ({std_contrast:.1f}). Child skin vs background separation may be hazy."
        })

    # If no contour_result was passed, we need it to check posture and landmarks
    if contour_result is None:
        try:
            from contour_extractor import extract_arm_contour_and_landmarks
            contour_result = extract_arm_contour_and_landmarks(image_bgr)
        except Exception as e:
            critical_failures.append(_make_failure(
                "CONTOUR_EXTRACTION_FAILED",
                f"Could not detect child arm contour: {str(e)}",
                "Ensure the child's entire left arm is visible and unobstructed from shoulder to hand."
            ))

    # -----------------------------------------------------------------------
    # Stage 2: Arm Straightness & Elbow Bend Angle
    # -----------------------------------------------------------------------
    lms_raw = contour_result.get("landmarks", []) if contour_result else []
    lm_dict = {lm["Landmark_ID"]: np.array([float(lm["X_Pixel"]), float(lm["Y_Pixel"])]) for lm in lms_raw}
    
    elbow_angle = 180.0
    stretch_ratio = 1.0
    upper_fore_ratio = 1.0
    sum_to_total_ratio = 1.0

    has_shoulder = 21 in lm_dict
    has_elbow = 2 in lm_dict
    has_wrist = 3 in lm_dict

    if has_shoulder and has_elbow and has_wrist:
        p_sh = lm_dict[21]
        p_el = lm_dict[2]
        p_wr = lm_dict[3]

        v_upper = p_sh - p_el
        v_fore = p_wr - p_el
        d_upper = float(np.linalg.norm(v_upper))
        d_fore = float(np.linalg.norm(v_fore))
        d_total = float(np.linalg.norm(p_wr - p_sh))

        if d_upper > 0 and d_fore > 0:
            cos_a = np.dot(v_upper, v_fore) / (d_upper * d_fore)
            elbow_angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
            stretch_ratio = d_total / (d_upper + d_fore + 1e-6)

        metrics["elbow_angle_deg"] = round(elbow_angle, 1)
        metrics["arm_stretch_ratio"] = round(stretch_ratio, 3)

        # 2.1 Check Elbow Angle (Straight Stretch)
        if elbow_angle < MIN_ELBOW_ANGLE_DEG or stretch_ratio < MIN_STRETCH_RATIO:
            critical_failures.append(_make_failure(
                "ARM_BENT_AT_ELBOW",
                f"Child's left arm is bent at {elbow_angle:.1f}° (required straight stretch >= {MIN_ELBOW_ANGLE_DEG:.0f}°). Stretch ratio: {stretch_ratio:.3f}.",
                "Gently hold child's left arm completely straight and flat along the surface. Do not allow elbow to bend."
            ))
    else:
        critical_failures.append(_make_failure(
            "PRIMARY_JOINTS_MISSING",
            "One or more core joints (Shoulder, Elbow, Wrist) could not be located.",
            "Ensure the full left arm from shoulder down to hand is clearly visible in the camera frame."
        ))

    # -----------------------------------------------------------------------
    # Stage 3: Joint Positioning Trio (Shoulder, Elbow, Wrist)
    # -----------------------------------------------------------------------
    upper_cm = float(contour_result.get("upper_arm_cm") or 0.0) if contour_result else 0.0
    fore_cm = float(contour_result.get("forearm_cm") or 0.0) if contour_result else 0.0
    tot_cm = float(contour_result.get("total_arm_cm") or 1.0) if contour_result else 1.0

    if fore_cm > 0:
        upper_fore_ratio = upper_cm / fore_cm
    if tot_cm > 0:
        sum_to_total_ratio = (upper_cm + fore_cm) / tot_cm

    metrics["upper_to_forearm_ratio"] = round(upper_fore_ratio, 2)
    metrics["sum_to_total_ratio"] = round(sum_to_total_ratio, 2)

    # 3.1 Shoulder Positioning (L21 vs Neck & sum-to-total ratio)
    if sum_to_total_ratio > MAX_SUM_TO_TOTAL_RATIO:
        critical_failures.append(_make_failure(
            "SHOULDER_LANDMARK_DISPLACED_NECK",
            f"Shoulder landmark displaced inward toward neck or outside anatomical joint (segment sum ratio: {sum_to_total_ratio:.2f} > {MAX_SUM_TO_TOTAL_RATIO}).",
            "Ensure child's neck and chest are unobstructed. The shoulder joint (deltoid) must be clearly visible and not covered by clothing or necklaces."
        ))

    # Check for neck cords / jewelry crossing near shoulder (L1, L21, L22)
    if 1 in lm_dict and 22 in lm_dict:
        sh_top = lm_dict[1]
        sh_bot = lm_dict[22]
        sh_w = float(np.linalg.norm(sh_top - sh_bot))
        metrics["shoulder_caliper_width_px"] = round(sh_w, 1)

        prox_w = float(np.linalg.norm(lm_dict[12] - lm_dict[4])) if (12 in lm_dict and 4 in lm_dict) else 1.0
        sh_prox_ratio = sh_w / prox_w
        metrics["shoulder_to_proximal_ratio"] = round(sh_prox_ratio, 2)

        if sh_prox_ratio < MIN_SHOULDER_TO_PROX_WIDTH_RATIO:
            critical_failures.append(_make_failure(
                "SHOULDER_JEWELRY_CORD_OCCLUSION",
                f"Necklace cords, amulets, or strings detected draping across shoulder / armpit (width ratio: {sh_prox_ratio:.2f} < {MIN_SHOULDER_TO_PROX_WIDTH_RATIO}).",
                "Please remove necklaces, sacred threads, amulets, or strings from the child's neck and shoulder before taking the photo."
            ))

    # 3.2 Elbow Positioning (L2 Anatomical Ratio, Centrality & Contour Under-Segmentation)
    if upper_fore_ratio < MIN_UPPER_FOREARM_RATIO or upper_fore_ratio > MAX_UPPER_FOREARM_RATIO:
        critical_failures.append(_make_failure(
            "ELBOW_LANDMARK_DISPLACED",
            f"Elbow landmark position is anatomically abnormal (Upper-arm / Forearm ratio: {upper_fore_ratio:.2f}, valid range: {MIN_UPPER_FOREARM_RATIO:.2f}-{MAX_UPPER_FOREARM_RATIO:.2f}).",
            "Ensure child's arm is fully uncovered from shoulder to wrist so the elbow joint line is clearly identifiable."
        ))

    # Check elbow caliper width vs upper arm width (flags contour cutting under skin edge at elbow)
    if 18 in lm_dict and 17 in lm_dict:
        w_elbow = float(np.linalg.norm(lm_dict[18] - lm_dict[17]))
        metrics["elbow_width_px"] = round(w_elbow, 1)

        w_upper_ref = float(np.linalg.norm(lm_dict[14] - lm_dict[13])) if (14 in lm_dict and 13 in lm_dict) else 1.0
        elbow_to_upper_ratio = w_elbow / w_upper_ref
        metrics["elbow_to_upper_width_ratio"] = round(elbow_to_upper_ratio, 2)

        if elbow_to_upper_ratio < MIN_ELBOW_TO_UPPER_WIDTH_RATIO:
            critical_failures.append(_make_failure(
                "ELBOW_CONTOUR_UNDERSEGMENTED",
                f"Contour severely cut under the skin edge at the elbow crease (elbow/upper arm width ratio {elbow_to_upper_ratio:.2f} < {MIN_ELBOW_TO_UPPER_WIDTH_RATIO}).",
                "Ensure good lighting with no deep shadows under the elbow crease and keep the arm flat."
            ))

    if 18 in lm_dict and 17 in lm_dict and has_elbow:
        el_top = lm_dict[18]
        el_bot = lm_dict[17]
        p_el = lm_dict[2]
        d_top = np.linalg.norm(p_el - el_top)
        d_bot = np.linalg.norm(p_el - el_bot)
        if max(d_top, d_bot) > 0:
            el_skew = abs(d_top - d_bot) / max(d_top, d_bot)
            metrics["elbow_center_skew"] = round(float(el_skew), 2)
            if el_skew > 0.60:
                warnings.append({
                    "code": "ELBOW_CENTER_SKEWED",
                    "severity": "WARNING",
                    "message": f"Elbow centerline marker skewed by {el_skew*100:.0f}% towards one skin edge."
                })

    # 3.3 Wrist Positioning, Anti-Collapse & Occlusion (L3, L19, L20)
    has_wrist_caliper = 20 in lm_dict and 19 in lm_dict
    dark_ratio = 0.0

    # Optical check: Sample brightness in wrist zone for dark bangle / thread detection
    if has_wrist and 3 in lm_dict:
        p_wr = lm_dict[3].astype(int)
        x_min = max(0, p_wr[0] - 25)
        x_max = min(w_img, p_wr[0] + 25)
        y_min = max(0, p_wr[1] - 25)
        y_max = min(h_img, p_wr[1] + 25)
        roi_wrist = gray[y_min:y_max, x_min:x_max]
        if roi_wrist.size > 0:
            dark_ratio = float(np.sum(roi_wrist < 40) / roi_wrist.size)
        metrics["wrist_dark_pixel_ratio"] = round(dark_ratio, 2)

    if has_wrist_caliper:
        w_wrist_px = float(np.linalg.norm(lm_dict[20] - lm_dict[19]))
        metrics["wrist_width_px"] = round(w_wrist_px, 1)

        # Compare to representative forearm width (use maximum of distal and mid forearm)
        forearm_widths = []
        for pair in [(6, 5), (8, 7)]:
            if pair[0] in lm_dict and pair[1] in lm_dict:
                forearm_widths.append(float(np.linalg.norm(lm_dict[pair[0]] - lm_dict[pair[1]])))

        ref_forearm_w = max(forearm_widths) if forearm_widths else None

        if ref_forearm_w and ref_forearm_w > 0:
            wrist_to_fore_ratio = w_wrist_px / ref_forearm_w
            metrics["wrist_to_forearm_ratio"] = round(wrist_to_fore_ratio, 2)

            # Pinched / Collapsed wrist (adult fingers holding child wrist)
            if wrist_to_fore_ratio < MIN_WRIST_TO_FOREARM_RATIO or w_wrist_px < MIN_WRIST_WIDTH_PX:
                critical_failures.append(_make_failure(
                    "WRIST_COLLAPSED_ADULT_FINGERS",
                    f"Wrist landmarks pinched or collapsed (ratio {wrist_to_fore_ratio:.2f} < {MIN_WRIST_TO_FOREARM_RATIO}). Adult fingers detected grasping child wrist or wrist turned.",
                    "Do not hold or squeeze child's wrist with fingers. Lay the wrist open and flat against the mat; support gently only by fingertips if needed."
                ))
            elif wrist_to_fore_ratio > MAX_WRIST_TO_FOREARM_RATIO:
                # Gross geometric enlargement: prominent loose bangle/bracelet
                critical_failures.append(_make_failure(
                    "WRIST_BANGLE_OR_JEWELRY_OCCLUSION",
                    f"Wrist width abnormally enlarged (ratio {wrist_to_fore_ratio:.2f} > {MAX_WRIST_TO_FOREARM_RATIO}). Bangle, charm, or bracelet detected.",
                    "Remove all black bangles, bracelets, threads, and jewelry from the child's wrist before taking the photo."
                ))
            elif wrist_to_fore_ratio > SUSPICIOUS_WRIST_RATIO_MIN and dark_ratio > MIN_BANGLE_DARK_PIXEL_RATIO:
                # Moderate widening confirmed by high-contrast dark bangle band
                critical_failures.append(_make_failure(
                    "WRIST_BANGLE_OR_JEWELRY_OCCLUSION",
                    f"Bangle or wrist charm detected (width ratio {wrist_to_fore_ratio:.2f} with {dark_ratio*100:.0f}% dark pixels).",
                    "Remove all black bangles, bracelets, threads, and jewelry from the child's wrist before taking the photo."
                ))
            elif wrist_to_fore_ratio > SUSPICIOUS_WRIST_RATIO_MIN:
                # Slightly flared wrist without dark band is natural anatomy: emit warning only, do not block!
                warnings.append({
                    "code": "WRIST_SLIGHTLY_WIDE",
                    "severity": "WARNING",
                    "message": f"Wrist width slightly flared (ratio {wrist_to_fore_ratio:.2f}). Clear of dark bangles; consistent with anatomical hand posture."
                })

    # Standalone check for dark bangle band (even if width didn't grossly expand)
    if dark_ratio > 0.28:
        critical_failures.append(_make_failure(
            "BLACK_BANGLE_DETECTED",
            f"High-contrast dark band detected around wrist ({dark_ratio*100:.0f}% dark pixels). Black bangle or wrist charm detected.",
            "Please remove all black bangles and bracelets from the child's left wrist."
        ))

    # -----------------------------------------------------------------------
    # Stage 4: Contour Skin-Edge Adherence & Completeness
    # -----------------------------------------------------------------------
    extracted_lm_count = len(lms_raw)
    metrics["extracted_landmarks_count"] = extracted_lm_count

    if extracted_lm_count < MIN_LANDMARK_COUNT:
        critical_failures.append(_make_failure(
            "CONTOUR_PERPENDICULAR_SCAN_FAILED",
            f"Only {extracted_lm_count}/22 landmarks could be resolved. Contour did not follow complete arm boundary (missing semi-landmarks).",
            "Ensure clean arm background with good illumination. Check for necklaces, sleeves, or shadows breaking the arm contour."
        ))

    # Skin color verification using YCrCb
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 130, 75]), np.array([255, 175, 130]))

    # Test contour adherence along upper and lower edges
    upper_seq = [1, 12, 14, 16, 18, 10, 8, 6, 20]
    lower_seq = [4, 13, 15, 17, 9, 7, 5, 19]

    adherence_hits = 0
    adherence_total = 0

    for seq in [upper_seq, lower_seq]:
        for i in range(len(seq) - 1):
            id_a, id_b = seq[i], seq[i+1]
            if id_a in lm_dict and id_b in lm_dict:
                pt_a = lm_dict[id_a].astype(int)
                pt_b = lm_dict[id_b].astype(int)
                for alpha in np.linspace(0.1, 0.9, 5):
                    sample_x = int(pt_a[0] * (1 - alpha) + pt_b[0] * alpha)
                    sample_y = int(pt_a[1] * (1 - alpha) + pt_b[1] * alpha)
                    if 0 <= sample_x < w_img and 0 <= sample_y < h_img:
                        adherence_total += 1
                        patch = skin_mask[max(0, sample_y-3):min(h_img, sample_y+4),
                                          max(0, sample_x-3):min(w_img, sample_x+4)]
                        if patch.size > 0 and np.mean(patch) > 30:
                            adherence_hits += 1

    adherence_score = (adherence_hits / adherence_total) if adherence_total > 0 else 0.0
    metrics["contour_skin_adherence_score"] = round(float(adherence_score), 2)

    if adherence_total > 0 and adherence_score < MIN_SKIN_CONTOUR_ADHERENCE:
        critical_failures.append(_make_failure(
            "CONTOUR_SKIN_EDGE_DEVIATION",
            f"Contour did not properly follow the child's skin edge ({adherence_score*100:.0f}% adherence < {MIN_SKIN_CONTOUR_ADHERENCE*100:.0f}% required). Contour cutting under skin at elbow or leaking into background.",
            "Ensure child's arm is laid flat against a clean surface without heavy shadows under the elbow or background pattern interference."
        ))

    # -----------------------------------------------------------------------
    # Decision Engine: Overall Quality Score (0-100) & Pass Threshold (>= 80%)
    # -----------------------------------------------------------------------
    score = 100.0

    # 1. Optical quality penalties
    if lap_var < MIN_BLUR_LAPLACIAN_VAR:
        score -= (25.0 if lap_var < 200.0 else 15.0)
    if mean_brightness < MIN_BRIGHTNESS or mean_brightness > MAX_BRIGHTNESS:
        score -= 20.0
    if std_contrast < MIN_CONTRAST_STD:
        score -= 5.0

    # 2. Arm straightness & geometry penalties
    if elbow_angle < MIN_ELBOW_ANGLE_DEG:
        score -= (30.0 if elbow_angle < 155.0 else 15.0)
    if stretch_ratio < MIN_STRETCH_RATIO:
        score -= 10.0

    # 3. Joint bounds & positioning penalties
    if upper_fore_ratio < MIN_UPPER_FOREARM_RATIO or upper_fore_ratio > MAX_UPPER_FOREARM_RATIO:
        score -= 15.0
    if sum_to_total_ratio > MAX_SUM_TO_TOTAL_RATIO:
        score -= 15.0

    # 4. Jewelry & occlusion penalties
    has_bangle = any(f.get("code") in ["WRIST_BANGLE_OR_JEWELRY_OCCLUSION", "BLACK_BANGLE_DETECTED"] for f in critical_failures)
    if has_bangle:
        score -= 25.0
    has_shoulder_jewelry = any(f.get("code") == "SHOULDER_JEWELRY_CORD_OCCLUSION" for f in critical_failures)
    if has_shoulder_jewelry:
        score -= 20.0
    has_pinched_wrist = any(f.get("code") == "WRIST_COLLAPSED_ADULT_FINGERS" for f in critical_failures)
    if has_pinched_wrist:
        score -= 25.0

    # 5. Landmark & contour adherence penalties
    if extracted_lm_count < MIN_LANDMARK_COUNT:
        score -= 25.0
    if adherence_score < MIN_SKIN_CONTOUR_ADHERENCE:
        score -= 20.0

    # 6. Fatal unrecoverable pipeline failures
    has_fatal_pipeline_failure = any(f.get("code") in ["PRIMARY_JOINTS_MISSING", "CONTOUR_EXTRACTION_FAILED"] for f in critical_failures)
    if has_fatal_pipeline_failure:
        score -= 40.0

    # Warnings minor deduction
    score -= len(warnings) * 2.0
    quality_score = max(0.0, min(100.0, round(score, 1)))

    # Acceptance Logic:
    # Image is valid if quality_score >= min_quality_score (default 80.0%)
    # and no fatal unrecoverable pipeline errors (e.g. missing core joints).
    is_valid = (quality_score >= min_quality_score) and (not has_fatal_pipeline_failure)
    status = "APPROVED" if is_valid else "REJECTED_QUALITY_FAILURE"

    # If score passes >= 80%, minor issues become informational warnings rather than blocking triage
    active_critical_failures = [] if is_valid else critical_failures

    # -----------------------------------------------------------------------
    # Generate Diagnostic Visual Overlay
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")

    # Show RGB image
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.axis("off")

    # Banner at top
    banner_color = "#10b981" if is_valid else "#ef4444"
    if is_valid:
        banner_text = f"APPROVED (Quality: {quality_score:.1f}% >= {min_quality_score:.0f}%)"
    else:
        err_code = active_critical_failures[0]['code'] if active_critical_failures else "LOW_QUALITY"
        banner_text = f"REJECTED: {err_code} (Quality: {quality_score:.1f}% < {min_quality_score:.0f}%)"

    ax.text(w_img * 0.5, h_img * 0.05, banner_text,
            color="#ffffff", backgroundcolor=banner_color,
            fontsize=9.5, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="square,pad=0.35", facecolor=banner_color, edgecolor="none"))

    # Draw elbow angle arc
    if has_shoulder and has_elbow and has_wrist:
        p_sh = lm_dict[21]
        p_el = lm_dict[2]
        p_wr = lm_dict[3]

        arm_color = "#10b981" if elbow_angle >= MIN_ELBOW_ANGLE_DEG else "#ef4444"
        ax.plot([p_sh[0], p_el[0], p_wr[0]], [p_sh[1], p_el[1], p_wr[1]],
                color=arm_color, linewidth=2.5, linestyle="--", zorder=4)

        angle_lbl = f"{elbow_angle:.1f}° (STRAIGHT)" if elbow_angle >= MIN_ELBOW_ANGLE_DEG else f"{elbow_angle:.1f}° [BENT - REJECTED]"
        ax.annotate(angle_lbl, (p_el[0], p_el[1]),
                    xytext=(0, -22), textcoords="offset points",
                    color=arm_color, fontweight="bold", fontsize=9,
                    ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e293b", edgecolor=arm_color))

    # Mark joints
    if has_shoulder:
        sh_color = "#ef4444" if sum_to_total_ratio > MAX_SUM_TO_TOTAL_RATIO else "#10b981"
        ax.scatter(lm_dict[21][0], lm_dict[21][1], color=sh_color, s=70, edgecolors="#ffffff", zorder=5)
        ax.annotate("Shoulder", (lm_dict[21][0], lm_dict[21][1]), xytext=(-10, -12), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=sh_color)

    if has_elbow:
        el_color = "#10b981" if (MIN_UPPER_FOREARM_RATIO <= upper_fore_ratio <= MAX_UPPER_FOREARM_RATIO) else "#ef4444"
        ax.scatter(lm_dict[2][0], lm_dict[2][1], color=el_color, s=70, edgecolors="#ffffff", zorder=5)
        ax.annotate("Elbow", (lm_dict[2][0], lm_dict[2][1]), xytext=(0, 12), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=el_color)

    if has_wrist:
        wr_color = "#10b981" if is_valid else "#f59e0b"
        ax.scatter(lm_dict[3][0], lm_dict[3][1], color=wr_color, s=70, edgecolors="#ffffff", zorder=5)
        ax.annotate("Wrist", (lm_dict[3][0], lm_dict[3][1]), xytext=(10, -12), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=wr_color)

    # Customer Instruction Callout on Image
    if not is_valid and len(active_critical_failures) > 0:
        first_issue = active_critical_failures[0]
        callout_text = f"Action Required:\n{first_issue['customer_instruction']}"
        ax.text(w_img * 0.5, h_img * 0.93, callout_text,
                color="#ffffff", fontsize=8.5, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#b91c1c", edgecolor="#ffffff", alpha=0.92))

    plt.tight_layout()
    diagnostic_overlay_b64 = _fig_to_base64(fig)

    bilingual_instructions = []
    if not is_valid:
        for f in active_critical_failures:
            if f.get("customer_instruction_te"):
                bilingual_instructions.append(f"{f['customer_instruction']} / {f['customer_instruction_te']}")
            else:
                bilingual_instructions.append(f['customer_instruction'])

    return {
        "is_valid": is_valid,
        "quality_score": quality_score,
        "min_quality_score_threshold": min_quality_score,
        "status": status,
        "critical_failures": active_critical_failures,
        "warnings": warnings,
        "metrics": metrics,
        "customer_instructions": bilingual_instructions,
        "diagnostic_overlay_base64": diagnostic_overlay_b64
    }
