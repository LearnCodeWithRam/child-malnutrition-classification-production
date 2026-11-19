"""
worker_auth.py
==============
Self-Service One-Time PIN Registration & Daily Quota Rate-Limiting Guard for WhatsApp.

Protects against unauthorized users consuming Twilio/Meta API balances or server resources.

Features:
1. Self-Service PIN Activation:
   Unregistered workers must send `JOIN POSHAN2026` (or clinic PIN) to activate their number.
   Once activated, their phone number is permanently recorded in `authorized_workers.json`.
2. Daily Rate Limiting:
   Prevents runaway costs by capping each worker to a configurable daily limit (default 25 scans/day).
3. Bilingual Prompts (English + Telugu):
   Clear registration and quota guidance for frontline ASHA / Anganwadi staff.
"""

import os
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

CURRENT_DIR = Path(__file__).resolve().parent
AUTH_DB_FILE = CURRENT_DIR / "authorized_workers.json"

# Configurable Registration PIN & Daily Quota (Overrides from .env if present)
DEFAULT_REGISTRATION_PIN = os.environ.get("REGISTRATION_PIN", "1234").strip()
DEFAULT_DAILY_QUOTA = int(os.environ.get("DAILY_SCAN_LIMIT", "25"))


def _clean_phone(phone: str) -> str:
    """Normalizes phone numbers across Twilio (whatsapp:+91...) and Meta (91...)."""
    if not phone:
        return ""
    p = phone.strip()
    if p.lower().startswith("whatsapp:"):
        p = p[9:]
    # Remove all non-digits except leading +
    digits = re.sub(r"[^\d]", "", p)
    return digits


def _load_db() -> Dict:
    """Loads authorized workers database from JSON file."""
    if not AUTH_DB_FILE.exists():
        return {}
    try:
        with open(AUTH_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_db(data: Dict):
    """Safely saves authorized workers database to disk."""
    try:
        with open(AUTH_DB_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WorkerAuth Error] Failed to write database: {e}")


def is_worker_authorized(phone: str) -> bool:
    """Checks whether a phone number is registered and active."""
    clean = _clean_phone(phone)
    if not clean:
        return False
    db = _load_db()
    return clean in db and db[clean].get("status") == "active"


def try_register_worker(phone: str, message_text: str) -> Tuple[bool, str]:
    """
    Checks if message_text contains the registration command (e.g. 'JOIN POSHAN2026').
    If valid, registers the phone number permanently.
    
    Returns:
        (is_success: bool, reply_message: str)
    """
    clean = _clean_phone(phone)
    if not clean:
        return False, "⚠️ Invalid phone number."

    text = message_text.strip().upper()
    valid_pin = DEFAULT_REGISTRATION_PIN.upper()

    # Supports both "JOIN POSHAN2026" or directly entering the PIN "POSHAN2026"
    is_match = False
    if text == valid_pin:
        is_match = True
    elif text.startswith("JOIN"):
        parts = text.split()
        if len(parts) >= 2 and parts[1] == valid_pin:
            is_match = True

    if not is_match:
        return False, get_unauthorized_prompt()

    # Register worker in persistent storage
    db = _load_db()
    today_str = str(date.today())
    db[clean] = {
        "registered_at": datetime.now().isoformat(),
        "status": "active",
        "pin_used": valid_pin,
        "scans_today": 0,
        "last_scan_date": today_str,
        "total_scans": 0
    }
    _save_db(db)

    success_msg = (
        "🎉 *నమోదు పూర్తయింది (Registration Successful)*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ *ఫోన్ నంబర్ (Phone):* `{clean}`\n"
        "మీ ఖాతా విజయవంతంగా యాక్టివేట్ చేయబడింది.\n"
        "Your account is now activated for Child Malnutrition Assessment.\n\n"
        "📸 *ఎలా ప్రారంభించాలి (How to Begin):*\n"
        "పిల్లల ఎడమ చేయి నిటారుగా ఉంచి స్పష్టమైన ఫోటో తీసి ఇక్కడ పంపండి.\n"
        "To begin the assessment, click 'Start Assessment' and complete the child's information. \n Next, select 'Left Arm Photo', take a clear photograph of the child's bare left arm, and submit it for analysis.\n\n"
        f"📊 *రోజువారీ కోటా (Daily Quota):* {DEFAULT_DAILY_QUOTA} స్కాన్‌లు / రోజుకు (scans per day)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "_పోషణ్ అప్లికేషన్ • Poshan Frontline Assistant_"
    )
    return True, success_msg


def check_and_increment_daily_quota(phone: str, daily_limit: int = DEFAULT_DAILY_QUOTA) -> Tuple[bool, str, int]:
    """
    Checks if the registered worker has remaining quota for today.
    If quota is available, increments their usage counter by 1.
    
    Returns:
        (can_proceed: bool, message: str, current_scans_today: int)
    """
    clean = _clean_phone(phone)
    db = _load_db()

    if clean not in db:
        return False, get_unauthorized_prompt(), 0

    worker = db[clean]
    today_str = str(date.today())

    # Reset count if it's a new day
    if worker.get("last_scan_date") != today_str:
        worker["scans_today"] = 0
        worker["last_scan_date"] = today_str

    scans_today = worker.get("scans_today", 0)

    if scans_today >= daily_limit:
        return False, get_quota_exceeded_prompt(scans_today, daily_limit), scans_today

    # Increment usage
    worker["scans_today"] = scans_today + 1
    worker["total_scans"] = worker.get("total_scans", 0) + 1
    _save_db(db)

    return True, "OK", worker["scans_today"]


def get_unauthorized_prompt() -> str:
    """Returns bilingual prompt guiding unregistered users to enter registration PIN (Telugu first)."""
    return (
        "🔒 *అనుమతి అవసరం (Access Restricted)*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "ఈ సేవ రిజిస్టర్ అయిన ఆశా / అంగన్‌వాడీ కార్యకర్తలకు మాత్రమే అందుబాటులో ఉంటుంది.\n"
        "This service is reserved for registered frontline ASHA / Anganwadi staff.\n\n"
        "🔑 *ఖాతా ప్రారంభించే విధానం (How to Activate):*\n"
        "మీ క్లినిక్ పిన్ (PIN) నంబర్‌ను క్రింది విధంగా టైప్ చేసి పంపండి:\n"
        f"👉 `JOIN {DEFAULT_REGISTRATION_PIN}`\n\n"
        "Reply with your clinic PIN to activate:\n"
        f"👉 `JOIN {DEFAULT_REGISTRATION_PIN}`\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "_పోషణ్ అప్లికేషన్ • Poshan Frontline Assistant_"
    )


def get_quota_exceeded_prompt(scans_today: int, limit: int) -> str:
    """Returns bilingual message when worker hits their daily scan limit (Telugu first)."""
    return (
        "⏳ *రోజువారీ పరిమితి ముగిసింది (Daily Limit Reached)*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"ఈ రోజుకు కేటాయించిన *{limit} స్కాన్‌ల కోటా* పూర్తయింది ({scans_today}/{limit}).\n"
        f"You have reached your daily quota of *{limit} scans* for today ({scans_today}/{limit}).\n\n"
        "🌅 *తదుపరి సమయం (Next Reset):*\n"
        "మీ కోటా రేపు ఉదయం 00:00 గంటలకు మళ్లీ రీసెట్ అవుతుంది.\n"
        "Your daily quota will reset tomorrow morning at 00:00.\n\n"
        "💡 అత్యవసర స్కాన్‌ల కోసం మీ ప్రాజెక్ట్ సమన్వయకర్తను సంప్రదించండి.\n"
        "For urgent scans, please contact your project coordinator.\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "_పోషణ్ అప్లికేషన్ • Poshan Frontline Assistant_"
    )
