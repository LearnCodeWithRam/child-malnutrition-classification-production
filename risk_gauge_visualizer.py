"""
risk_gauge_visualizer.py
========================
Frontline Malnutrition Risk Scale & Speedometer Gauge Visualizer for PediaMorph AI.

Provides:
1. High-Resolution 5-Zone Speedometer Risk Gauge Dial:
   Matches the clinical Child risk scale:
     * LOW RISK (0% - 20%): Green (#00a651)
     * MEDIUM LOW RISK (20% - 40%): Yellow (#fff200)
     * MEDIUM RISK (40% - 60%): Orange (#f7941d)
     * MEDIUM HIGH RISK (60% - 80%): Red/Coral (#ed1c24)
     * HIGH RISK (80% - 100%): Crimson (#be1e2d)
   - Tapered needle pointer with hollow circular pivot hub, pointing to the exact ensemble risk.
   - Clinical triage badge and percentage readout.
   - Encodes directly to Base64 image data URL for APIs, Web, and WhatsApp media.

2. WhatsApp-Optimized Text-Based Risk Scale Meter:
   - High-readability emoji scale bar for instant field recognition by frontline workers.
   - Dynamic cursor pointer showing exact location on the LOW -> CRITICAL spectrum.
"""

import io
import base64
from pathlib import Path
from typing import Union
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, Polygon


def generate_risk_gauge_speedometer(
    probability: float,
    tier: str = "RED",
    title: str = "POSHAN APPLICATION • RISK METER"
) -> str:
    """
    Renders the exact 5-zone semi-circular speedometer risk gauge dial and returns Base64 data URL.
    
    Args:
        probability: Float ensemble risk probability (0.0 to 1.0).
        tier: Diagnostic tier ('GREEN', 'YELLOW', 'RED').
        title: Top header title.
        
    Returns:
        Base64 encoded PNG data URI string ('data:image/png;base64,...').
    """
    p = np.clip(float(probability), 0.0, 1.0)
    p_pct = p * 100.0

    fig, ax = plt.subplots(figsize=(8.0, 5.0), facecolor='#ffffff', dpi=180)
    ax.set_aspect('equal')
    ax.axis('off')

    # 5 Equal 36-degree Segments (0 to 180 degrees)
    zones = [
        (180, 144, '#00a651', 'LOW\nRISK', '#e8f5e9'),
        (144, 108, '#fff200', 'MEDIUM\nLOW\nRISK', '#fffde7'),
        (108, 72,  '#f7941d', 'MEDIUM\nRISK', '#fff3e0'),
        (72,  36,  '#ed1c24', 'MEDIUM\nHIGH\nRISK', '#ffebee'),
        (36,  0,   '#be1e2d', 'HIGH\nRISK', '#fce4ec')
    ]

    r_outer = 1.0
    r_inner = 0.55

    # Draw 5 curved wedge blocks with white boundary borders
    for th1, th2, col, lbl, badge_bg in zones:
        w = Wedge((0, 0), r_outer, th2, th1, width=(r_outer - r_inner),
                  facecolor=col, edgecolor='#ffffff', lw=4.0)
        ax.add_patch(w)

        # Position badge label in arc
        mid_th = np.radians((th1 + th2) / 2)
        lr = r_inner + (r_outer - r_inner) * 0.50
        lx = lr * np.cos(mid_th)
        ly = lr * np.sin(mid_th)

        bbox_props = dict(boxstyle='square,pad=0.25', facecolor=badge_bg, edgecolor='none', alpha=0.95)
        ax.text(lx, ly, lbl, color='#1e293b', fontsize=8.5, fontweight='heavy',
                ha='center', va='center', bbox=bbox_props, linespacing=1.1)

    # Needle geometry: 0% -> 180 deg (left), 100% -> 0 deg (right)
    needle_angle_deg = 180.0 * (1.0 - p)
    needle_rad = np.radians(needle_angle_deg)
    needle_len = 0.92

    tip_x = needle_len * np.cos(needle_rad)
    tip_y = needle_len * np.sin(needle_rad)

    # Pivot circle outer & inner radius (hollow circular hub matching user's image)
    r_pivot_out = 0.22
    r_pivot_in = 0.11

    perp_rad = needle_rad + np.pi / 2
    w_base = 0.065
    b1_x = r_pivot_out * 0.9 * np.cos(needle_rad) + w_base * np.cos(perp_rad)
    b1_y = r_pivot_out * 0.9 * np.sin(needle_rad) + w_base * np.sin(perp_rad)
    b2_x = r_pivot_out * 0.9 * np.cos(needle_rad) - w_base * np.cos(perp_rad)
    b2_y = r_pivot_out * 0.9 * np.sin(needle_rad) - w_base * np.sin(perp_rad)

    # Pointer blade
    blade = Polygon([[tip_x, tip_y], [b1_x, b1_y], [b2_x, b2_y]],
                    closed=True, facecolor='#333333', edgecolor='#222222', lw=1.5, zorder=6)
    ax.add_patch(blade)

    # Donut pivot hub
    ax.add_patch(Circle((0, 0), r_pivot_out, facecolor='#333333', edgecolor='#ffffff', lw=2.5, zorder=7))
    ax.add_patch(Circle((0, 0), r_pivot_in, facecolor='#ffffff', edgecolor='#333333', lw=2.0, zorder=8))

    # Tier color & description
    tier_colors = {'GREEN': '#00a651', 'YELLOW': '#f7941d', 'RED': '#be1e2d'}
    tc = tier_colors.get(str(tier).upper(), '#be1e2d')

    tier_labels = {
        'GREEN': 'NORMAL NUTRITIONAL STATUS (LOW RISK)',
        'YELLOW': 'MODERATE RISK (MAM / AT-RISK MONITORING)',
        'RED': 'CRITICAL RISK (SAM / URGENT NRC REFERRAL)'
    }
    badge_label = tier_labels.get(str(tier).upper(), 'MALNUTRITION EVALUATION')

    # Value and status readout below gauge
    ax.text(0, -0.32, f'PREDICTED RISK: {p_pct:.1f}%', fontsize=18, fontweight='heavy',
            color='#1e293b', ha='center', va='top')

    bbox_tier = dict(boxstyle='round,pad=0.35', facecolor=tc, edgecolor='none', alpha=0.95)
    ax.text(0, -0.48, badge_label, fontsize=9.5, fontweight='heavy',
            color='#ffffff', ha='center', va='top', bbox=bbox_tier)

    # Top title
    ax.text(0, 1.15, title, fontsize=11, fontweight='heavy',
            color='#1e293b', ha='center', va='center')

    ax.set_xlim(-1.20, 1.20)
    ax.set_ylim(-0.64, 1.25)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"


def save_risk_gauge_image(probability: float, tier: str, output_path: Union[str, Path]) -> Path:
    """Saves the risk gauge PNG directly to disk."""
    data_uri = generate_risk_gauge_speedometer(probability, tier)
    b64_data = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(base64.b64decode(b64_data))
    return out


def format_text_risk_scale(probability: float, tier: str = "RED") -> str:
    """
    Constructs a visual text-based Risk Scale Meter for WhatsApp / Chat UI matching the 5-zone gauge.
    
    Example Output:
    📈 *Risk Scale Meter / రిస్క్ స్కేల్:*
    [ 🟢 Low ── 🟡 Med-Low ── 🟠 Medium ── 🔴 Med-High ── 🛑 High ]
    [ 🟩🟩 🟨🟨 🟧🟧 🟥🟥 🛑🛑 ] ◄ *75.9%* ◄ *CRITICAL (SAM)*
    • Risk Zone / స్థితి: 🔴 *CRITICAL SAM / తీవ్రమైన పోషకాహార లోపం*
    """
    p = np.clip(float(probability), 0.0, 1.0)
    pct = p * 100.0

    # Determine visual zone badge and localized label
    if pct < 20.0:
        zone_badge = "🟢 *LOW RISK / సాధారణ స్థితి*"
        zone_pointer = "◄ *LOW (ఆరోగ్యం)*"
    elif pct < 40.0:
        zone_badge = "🟡 *MEDIUM LOW RISK / స్వల్ప రిస్క్*"
        zone_pointer = "◄ *MED-LOW*"
    elif pct < 60.0:
        zone_badge = "🟠 *MEDIUM RISK (MAM) / మధ్యస్థ పోషకాహార లోపం*"
        zone_pointer = "◄ *MEDIUM (MAM)*"
    elif pct < 80.0:
        zone_badge = "🔴 *MEDIUM HIGH RISK (SAM) / తీవ్రమైన పోషకాహార లోపం*"
        zone_pointer = "◄ *MED-HIGH (SAM)*"
    else:
        zone_badge = "🛑 *HIGH CRITICAL SAM / అత్యంత తీవ్రమైన లోపం*"
        zone_pointer = "◄ *HIGH (CRITICAL)*"

    # 5-zone scale bar with 2 blocks per zone (10 blocks total)
    scale_bar = "🟩🟩 🟨🟨 🟧🟧 🟥🟥 🛑🛑"

    text_meter = (
        f"📈 *Risk Scale Meter / రిస్క్ స్కేల్:*\n"
        f"[ 🟢 Low ── 🟡 Med-Low ── 🟠 Medium ── 🔴 Med-High ── 🛑 High ]\n"
        f"[ {scale_bar} ] ◄ *{pct:.1f}%* {zone_pointer}\n"
        f"• Risk Zone / స్థితి: {zone_badge}"
    )
    return text_meter


def generate_interactive_html_report(session_id: str, record: dict, base_url: str = "") -> str:
    """
    Generates a standalone, responsive HTML5 report page with an interactive animated
    5-zone speedometer risk gauge, live counter, and clinical diagnostic breakdown.
    """
    triage = record.get("clinical_triage", {})
    bio = record.get("biometrics", {})
    lime_data = record.get("lime_explainability", {})
    timestamp = record.get("timestamp_iso", "")

    tier = triage.get("tier", "RED")
    # Always prioritize the Primary Ensemble Risk (LDA + XGBoost) for the gauge needle
    p_ens = triage.get("primary_ensemble_probability")
    if p_ens is None:
        p_ens = triage.get("malnutrition_probability", 0.60)
    p = float(p_ens)
    pct = round(p * 100.0, 1)

    p_egat = triage.get("backup_egat_probability")
    egat_str = f"{p_egat * 100:.1f}%" if p_egat is not None else "N/A (Disabled)"

    # Rotation angle for SVG needle (-90 deg at 0%, 0 deg at 50%, +90 deg at 100%)
    needle_deg = round((p - 0.5) * 180.0, 2)

    tier_colors = {
        "GREEN": "#00a651",
        "YELLOW": "#f7941d",
        "RED": "#be1e2d"
    }
    tc = tier_colors.get(str(tier).upper(), "#be1e2d")

    tier_titles = {
        "GREEN": "🟢 NORMAL NUTRITIONAL STATUS / సాధారణ పోషణ స్థితి",
        "YELLOW": "🟡 MODERATE RISK (MAM / AT-RISK) / మధ్యస్థ పోషకాహార లోపం",
        "RED": "🔴 CRITICAL SAM (URGENT NRC REFERRAL) / తీవ్రమైన పోషకాహార లోపం"
    }
    tier_title = tier_titles.get(str(tier).upper(), f"{tier} RISK")

    action_guides = {
        "RED": {
            "en": "⚠ Please refer the child to a Primary Health Centre (PHC) or Nutrition Rehabilitation Centre (NRC) for further nutritional assessment and care.",
            "te": "⚠ పిల్లలను మరింత పోషకాహార మూల్యాంకనం మరియు సంరక్షణ కోసం సమీప ప్రాథమిక ఆరోగ్య కేంద్రం (PHC) లేదా పోషక పునరావాస కేంద్రం (NRC)కు తీసుకెళ్లాలని సూచించబడుతోంది.",
        },
        "YELLOW": {
            "en": "Follow-up monitoring in 14 days. Re-check dietary intake and physical MUAC tape measurement.",
            "te": "బిడ్డ బరువును గమనించండి. 14 రోజుల్లో మళ్లీ తనిఖీ చేయండి. టేపుతో MUAC కొలత సరిచూడండి."
        },
        "GREEN": {
            "en": "Child demonstrates healthy arm morphology. Continue age-appropriate nutrition and growth monitoring.",
            "te": "బిడ్డ ఆరోగ్యంగా ఉన్నాడు/ఉన్నది. సమతుల్య ఆహారం మరియు తల్లిపాలు కొనసాగించండి."
        }
    }
    ag = action_guides.get(str(tier).upper(), action_guides["RED"])

    muac_cm = bio.get("vision_estimated_muac_cm", "13.0")
    arm_cm = bio.get("total_arm_length_cm", "28.5")
    primary_wasting = triage.get("primary_wasting_locus", "Proximal Brachium/Bicep (L14-L13)")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Poshan Application</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text-main: #f8fafc;
      --text-sub: #94a3b8;
      --accent: {tc};
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text-main);
      padding: 16px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 640px;
      margin: 0 auto;
    }}
    .header {{
      text-align: center;
      padding: 16px 0;
    }}
    .header h1 {{
      font-size: 20px;
      font-weight: 800;
      letter-spacing: -0.5px;
      color: #38bdf8;
    }}
    .header p {{
      font-size: 12px;
      color: var(--text-sub);
      margin-top: 4px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 20px;
      margin-bottom: 16px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .gauge-wrapper {{
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }}
    .gauge-svg {{
      width: 100%;
      max-width: 360px;
      height: auto;
      overflow: visible;
    }}
    #needle-group {{
      transform-origin: 200px 200px;
      transform: rotate(-90deg);
      transition: transform 1.8s cubic-bezier(0.2, 0.9, 0.3, 1.0);
    }}
    .score-display {{
      text-align: center;
      margin-top: -10px;
    }}
    .score-number {{
      font-size: 38px;
      font-weight: 900;
      color: #ffffff;
      letter-spacing: -1px;
    }}
    .score-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-sub);
      margin-top: -4px;
    }}
    .badge {{
      display: inline-block;
      background: var(--accent);
      color: #ffffff;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 700;
      margin-top: 10px;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .stat-box {{
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px;
      text-align: center;
    }}
    .stat-val {{
      font-size: 20px;
      font-weight: 800;
      color: #38bdf8;
    }}
    .stat-lbl {{
      font-size: 11px;
      color: var(--text-sub);
      margin-top: 2px;
    }}
    .section-title {{
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 6px;
      color: #e2e8f0;
    }}
    .guide-box {{
      background: rgba(190, 30, 45, 0.12);
      border-left: 4px solid var(--accent);
      padding: 12px 14px;
      border-radius: 0 8px 8px 0;
      margin-top: 8px;
    }}
    .guide-box p {{
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .guide-box p.te {{
      color: #fbbf24;
      font-size: 13.5px;
      margin-bottom: 0;
    }}
    .footer {{
      text-align: center;
      font-size: 11px;
      color: var(--text-sub);
      padding: 16px 0 32px 0;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>👶 Poshan Application Result</h1>
       <!-- <p>Session ID: <code>{session_id}</code></p> -->
    </div>

    <!-- Speedometer Gauge Card -->
    <div class="card gauge-wrapper">
      <svg class="gauge-svg" viewBox="0 0 400 240">
        <!-- 5-Zone Gauge Arcs (Matching Reference Speedometer) -->
        <!-- Zone 1: LOW RISK (Green) 180 -> 144 deg -->
        <path d="M 40.00 200.00 A 160 160 0 0 1 70.56 105.95 L 123.14 144.16 A 95 95 0 0 0 105.00 200.00 Z" fill="#00a651" stroke="#ffffff" stroke-width="3"/>
        <!-- Zone 2: MED-LOW RISK (Yellow) 144 -> 108 deg -->
        <path d="M 70.56 105.95 A 160 160 0 0 1 150.56 47.83 L 170.64 109.65 A 95 95 0 0 0 123.14 144.16 Z" fill="#fff200" stroke="#ffffff" stroke-width="3"/>
        <!-- Zone 3: MEDIUM RISK (Orange) 108 -> 72 deg -->
        <path d="M 150.56 47.83 A 160 160 0 0 1 249.44 47.83 L 229.36 109.65 A 95 95 0 0 0 170.64 109.65 Z" fill="#f7941d" stroke="#ffffff" stroke-width="3"/>
        <!-- Zone 4: MED-HIGH RISK (Coral/Red) 72 -> 36 deg -->
        <path d="M 249.44 47.83 A 160 160 0 0 1 329.44 105.95 L 276.86 144.16 A 95 95 0 0 0 229.36 109.65 Z" fill="#ed1c24" stroke="#ffffff" stroke-width="3"/>
        <!-- Zone 5: HIGH RISK (Crimson) 36 -> 0 deg -->
        <path d="M 329.44 105.95 A 160 160 0 0 1 360.00 200.00 L 295.00 200.00 A 95 95 0 0 0 276.86 144.16 Z" fill="#be1e2d" stroke="#ffffff" stroke-width="3"/>

        <!-- Segment Badge Labels -->
        <g font-size="8" font-weight="800" text-anchor="middle" fill="#1e293b">
          <!-- Zone 1 Label -->
          <rect x="52" y="145" width="48" height="20" rx="3" fill="#e8f5e9"/>
          <text x="76" y="158">LOW RISK</text>
          <!-- Zone 2 Label -->
          <rect x="88" y="76" width="56" height="24" rx="3" fill="#fffde7"/>
          <text x="116" y="87">MED-LOW</text>
          <text x="116" y="96">RISK</text>
          <!-- Zone 3 Label -->
          <rect x="175" y="48" width="50" height="24" rx="3" fill="#fff3e0"/>
          <text x="200" y="59">MEDIUM</text>
          <text x="200" y="68">RISK</text>
          <!-- Zone 4 Label -->
          <rect x="256" y="76" width="56" height="24" rx="3" fill="#ffebee"/>
          <text x="284" y="87">MED-HIGH</text>
          <text x="284" y="96">RISK</text>
          <!-- Zone 5 Label -->
          <rect x="300" y="145" width="48" height="20" rx="3" fill="#fce4ec"/>
          <text x="324" y="158">HIGH RISK</text>
        </g>

        <!-- Dynamic Animated Needle -->
        <g id="needle-group">
          <!-- Tapered Needle Blade pointing vertically up at 0 deg -->
          <polygon points="200,42 208,185 192,185" fill="#333333" stroke="#1e293b" stroke-width="1.5"/>
          <!-- Donut Hub matching reference image -->
          <circle cx="200" cy="200" r="30" fill="#333333" stroke="#ffffff" stroke-width="3"/>
          <circle cx="200" cy="200" r="14" fill="#ffffff" stroke="#333333" stroke-width="2"/>
        </g>
      </svg>

      <div class="score-display">
        <div class="score-number" id="risk-counter">0.0%</div>
        <div class="score-label">Primary Predicted Risk Probability</div>
        <div class="badge">{tier_title}</div>
      </div>
    </div>

    <!-- Clinical Action Card -->
    <div class="card">
      <div class="section-title">🩺 Clinical Referral Action</div>
      <div class="guide-box">
        <p><strong>EN:</strong> {ag['en']}</p>
        <p class="te"><strong>TE:</strong> {ag['te']}</p>
      </div>
    </div>

    <!-- Estimated Physical Dimensions -->
    <div class="card">
      <div class="section-title">📏 Vision Estimated Dimensions</div>
      <div class="grid-2">
        <div class="stat-box">
          <div class="stat-val">~{muac_cm} cm</div>
          <div class="stat-lbl">Estimated MUAC</div>
        </div>
        <div class="stat-box">
          <div class="stat-val">{arm_cm} cm</div>
          <div class="stat-lbl">Total Arm Length</div>
        </div>
      </div>
    </div>

    <!-- Diagnostic Breakdown -->
    <div class="card">
      <div class="section-title">🔬 Diagnostic Cross-Verification</div>
      <div style="font-size: 13px; color: var(--text-sub);">
        <p style="margin-bottom: 6px;">• <strong>Risk Prediction:</strong> <code style="color:#38bdf8; font-size:14px;">{pct}%</code> </p>
       <!--  <p style="margin-bottom: 6px;">• <strong>Cloud EGAT Cross-Check:</strong> <code>{egat_str}</code> (17-Node Attention Network)</p>  -->
        <p style="margin-bottom: 6px;">• <strong>Left Arm Wasting Locus:</strong> <span style="color: #fbbf24;">{primary_wasting}</span></p>
        <p>• <strong>Methodology:</strong> Child Left-Arm Morphometrics</p>
      </div>
    </div>

    <div class="footer">
      Poshan Application • Frontline Child Health Assistant<br>
      Automated Clinical Report for Anganwadi / ASHA Health Workers
    </div>
  </div>

  <script>
    // Smooth Needle Sweep & Counter Tick-Up Animation
    window.addEventListener('DOMContentLoaded', () => {{
      const targetDeg = {needle_deg};
      const targetPct = {pct};
      const needle = document.getElementById('needle-group');
      const counter = document.getElementById('risk-counter');

      setTimeout(() => {{
        needle.style.transform = `rotate(${{targetDeg}}deg)`;
      }}, 150);

      const duration = 1600;
      const startTime = performance.now();

      function updateCounter(currentTime) {{
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1.0);
        // Ease-out cubic
        const ease = 1 - Math.pow(1 - progress, 3);
        const currentVal = (targetPct * ease).toFixed(1);
        counter.textContent = `${{currentVal}}%`;

        if (progress < 1.0) {{
          requestAnimationFrame(updateCounter);
        }} else {{
          counter.textContent = `${{targetPct}}%`;
        }}
      }}
      requestAnimationFrame(updateCounter);
    }});
  </script>
</body>
</html>
"""
    return html

