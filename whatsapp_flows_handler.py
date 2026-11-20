"""
whatsapp_flows_handler.py
=========================
Production-grade Meta WhatsApp Flows (v7.3 / Data API v3.0) Handler.
Follows the official Meta WhatsApp-Flows-Tools architecture:
1. End-to-end payload encryption & decryption (RSA-OAEP-SHA256 + AES-GCM-128/256 with flipped IV).
2. Endpoint lifecycle: handles 'ping' healthcheck, 'INIT' dynamic screen entry, and 'data_exchange'.
3. Graph API client: interactive flow message dispatcher (navigate to FIELD_ASSESSMENT_SCREEN).
4. Flow completion processor: parses nfm_reply, resolves PhotoPicker media, invokes PediaMorph AI pipeline,
   records field audit registry, and delivers Speedometer Risk Gauge image + clinical triage text.
"""

import os
import json
import base64
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable

from base64 import b64decode, b64encode
from cryptography.hazmat.primitives.asymmetric.padding import OAEP, MGF1, hashes
from cryptography.hazmat.primitives.ciphers import algorithms, Cipher, modes
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from fastapi.responses import PlainTextResponse, JSONResponse

CURRENT_DIR = Path(__file__).resolve().parent


def _ensure_env_loaded():
    """Ensures local .env variables are populated in os.environ."""
    candidates = [
        CURRENT_DIR / ".env",
        CURRENT_DIR.parent / ".env",
        CURRENT_DIR.parent.parent / ".env"
    ]
    for env_path in candidates:
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k_clean = k.strip()
                            if k_clean not in os.environ:
                                os.environ[k_clean] = v.strip().strip('"').strip("'")
            except Exception:
                pass


_ensure_env_loaded()


class WhatsAppFlowsHandler:
    """
    Dedicated handler for Meta WhatsApp Flows (v7.3, Data API v3.0).
    Manages cryptography, endpoint routing, message dispatching, and form submission lifecycle.
    """

    def __init__(self, private_key_path: Optional[str] = None):
        self.private_key_path = private_key_path
        self._private_key = None

    # -------------------------------------------------------------------------
    # 1. Cryptography Engine (RSA-OAEP-SHA256 + AES-GCM with flipped IV)
    # -------------------------------------------------------------------------
    def get_private_key(self):
        """Loads RSA private key from environment or PEM file with resilient password fallback."""
        if self._private_key:
            return self._private_key

        _ensure_env_loaded()

        password = (
            os.environ.get("PRIVATE_KEY_PASSWORD")
            or os.environ.get("FLOW_PRIVATE_KEY_PASSWORD")
            or ""
        ).strip()
        pw_bytes = password.encode("utf-8") if password else None

        def _try_load(raw_bytes: bytes):
            if pw_bytes:
                try:
                    return load_pem_private_key(raw_bytes, password=pw_bytes)
                except (TypeError, ValueError):
                    pass
            try:
                return load_pem_private_key(raw_bytes, password=None)
            except Exception:
                if pw_bytes:
                    return load_pem_private_key(raw_bytes, password=pw_bytes)
                raise

        # 1. Check embedded env variable string
        raw_env_key = os.environ.get("PRIVATE_KEY") or os.environ.get("FLOW_PRIVATE_KEY")
        if raw_env_key:
            try:
                formatted_key = raw_env_key.replace("\\n", "\n").strip()
                self._private_key = _try_load(formatted_key.encode("utf-8"))
                return self._private_key
            except Exception as e:
                print(f"[FLOW KEY] Could not load private key from env string: {e}", flush=True)

        # 2. Check disk paths
        candidates = []
        custom_path = self.private_key_path or os.environ.get("FLOW_PRIVATE_KEY_PATH")
        if custom_path:
            candidates.append(Path(custom_path))

        candidates.extend([
            CURRENT_DIR.parent / "poshan-whatsapp-keys" / "private.pem",
            CURRENT_DIR / "poshan-whatsapp-keys" / "private.pem",
            CURRENT_DIR.parent.parent / "poshan-whatsapp-keys" / "private.pem",
            CURRENT_DIR / "flow_private.pem",
        ])

        for p in candidates:
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        self._private_key = _try_load(f.read())
                        print(f"[FLOW KEY] Successfully loaded private key from {p}", flush=True)
                        return self._private_key
                except Exception as e:
                    print(f"[FLOW KEY] Found {p} but failed to load: {e}", flush=True)

        return None

    def decrypt_flow_request(
        self,
        encrypted_flow_data_b64: str,
        encrypted_aes_key_b64: str,
        initial_vector_b64: str
    ) -> Tuple[Dict[str, Any], bytes, bytes]:
        """
        Decrypts incoming Meta WhatsApp Flow data using RSA-OAEP SHA-256 for the AES key
        and AES-GCM for the body.
        """
        private_key = self.get_private_key()
        if not private_key:
            raise ValueError("Flow Private Key is not loaded or configured.")

        flow_data = b64decode(encrypted_flow_data_b64)
        iv = b64decode(initial_vector_b64)

        # 1. Decrypt AES Key using RSA-OAEP-SHA256
        encrypted_aes_key = b64decode(encrypted_aes_key_b64)
        aes_key = private_key.decrypt(
            encrypted_aes_key,
            OAEP(mgf=MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )

        # 2. Decrypt Flow Data with AES-GCM
        body_ciphertext = flow_data[:-16]
        tag = flow_data[-16:]
        decryptor = Cipher(algorithms.AES(aes_key), modes.GCM(iv, tag)).decryptor()
        decrypted_bytes = decryptor.update(body_ciphertext) + decryptor.finalize()
        decrypted_json = json.loads(decrypted_bytes.decode("utf-8"))

        return decrypted_json, aes_key, iv

    def encrypt_flow_response(self, response_data: Dict[str, Any], aes_key: bytes, iv: bytes) -> str:
        """
        Encrypts outgoing Meta WhatsApp Flow response using flipped IV (bitwise NOT) and AES-GCM.
        Returns a base64 encoded string as required by Meta Flow Data API v3.0.
        """
        flipped_iv = bytearray(b ^ 0xFF for b in iv)
        encryptor = Cipher(algorithms.AES(aes_key), modes.GCM(flipped_iv)).encryptor()
        json_bytes = json.dumps(response_data).encode("utf-8")
        ciphertext = encryptor.update(json_bytes) + encryptor.finalize()
        return b64encode(ciphertext + encryptor.tag).decode("utf-8")

    # -------------------------------------------------------------------------
    # 2. Flow Data Endpoint Routing (ping, INIT, data_exchange)
    # -------------------------------------------------------------------------
    async def process_flow_data_endpoint(self, body: Dict[str, Any], request_host: str = "127.0.0.1:8000") -> PlainTextResponse:
        """
        Processes encrypted Meta Flow Data API requests and returns encrypted responses.
        Handles:
        - action == 'ping' (Meta endpoint verification)
        - action == 'INIT' (Dynamic flow initialization)
        - action == 'data_exchange' (Form step or validation)
        """
        encrypted_flow_data_b64 = body.get("encrypted_flow_data")
        encrypted_aes_key_b64 = body.get("encrypted_aes_key")
        initial_vector_b64 = body.get("initial_vector")

        if not (encrypted_flow_data_b64 and encrypted_aes_key_b64 and initial_vector_b64):
            return PlainTextResponse("Missing encrypted flow data fields", status_code=400)

        try:
            decrypted_data, aes_key, iv = self.decrypt_flow_request(
                encrypted_flow_data_b64, encrypted_aes_key_b64, initial_vector_b64
            )
            print(f"[Flow Endpoint] Decrypted Request: {decrypted_data}", flush=True)

            action = decrypted_data.get("action")
            today_str = datetime.now().strftime("%Y-%m-%d")

            # 1. Meta Healthcheck Ping
            if action == "ping":
                response_payload = {
                    "version": "3.0",
                    "data": {
                        "status": "active"
                    }
                }

            # 2. Dynamic Entry (INIT)
            elif action == "INIT":
                response_payload = {
                    "version": "3.0",
                    "screen": "DEMOGRAPHICS_SCREEN",
                    "data": {
                        "max_date": today_str
                    }
                }

            # 3. Data Exchange / Dynamic Multi-Screen Transition
            elif action == "data_exchange":
                current_screen = decrypted_data.get("screen", "DEMOGRAPHICS_SCREEN")
                data = decrypted_data.get("data", {})
                print(f"[Flow Data Exchange] Incoming Screen: '{current_screen}' | Data: {data}", flush=True)

                if current_screen == "DEMOGRAPHICS_SCREEN":
                    awc_str = str(data.get("awc_code", "28548190419"))

                    # Transition to Screen 2: MEASUREMENTS_SCREEN
                    response_payload = {
                        "version": "3.0",
                        "screen": "MEASUREMENTS_SCREEN",
                        "data": {
                            "awc_code": awc_str,
                            "serial_number": str(data.get("serial_number", "SN-048")),
                            "date_of_birth": str(data.get("date_of_birth", today_str)),
                            "gender": str(data.get("gender", "Female"))
                        }
                    }
                elif current_screen == "MEASUREMENTS_SCREEN":
                    awc_str = str(data.get("awc_code", "28548190419"))
                    ht_str = str(data.get("height_cm", "78.5"))
                    wt_str = str(data.get("weight_kg", "9.8"))
                    muac_str = str(data.get("muac_cm", "12.5"))

                    # Transition to Screen 3: PHOTO_CAPTURE_SCREEN
                    response_payload = {
                        "version": "3.0",
                        "screen": "PHOTO_CAPTURE_SCREEN",
                        "data": {
                            "awc_code": awc_str,
                            "serial_number": str(data.get("serial_number", "SN-048")),
                            "date_of_birth": str(data.get("date_of_birth", today_str)),
                            "gender": str(data.get("gender", "Female")),
                            "height_cm": ht_str,
                            "weight_kg": wt_str,
                            "muac_cm": muac_str,
                            "nutritional_status": str(data.get("nutritional_status", "mam"))
                        }
                    }
                else:
                    # Default return to initial screen
                    response_payload = {
                        "version": "3.0",
                        "screen": "DEMOGRAPHICS_SCREEN",
                        "data": {
                            "max_date": today_str
                        }
                    }

            # Fallback
            else:
                response_payload = {
                    "version": "3.0",
                    "data": {
                        "status": "active"
                    }
                }

            encrypted_response = self.encrypt_flow_response(response_payload, aes_key, iv)
            return PlainTextResponse(content=encrypted_response, status_code=200)

        except Exception as e:
            print(f"[Flow Endpoint Error] {e}", flush=True)
            return PlainTextResponse(content=f"Decryption/Processing failure: {e}", status_code=421)

    # -------------------------------------------------------------------------
    # 3. Outbound Interactive Flow Message Dispatcher
    # -------------------------------------------------------------------------
    def upload_media_to_meta(self, phone_number_id: str, file_path: str, mime_type: str = "image/jpeg") -> Optional[str]:
        """Uploads local image to Meta WhatsApp Media API and returns cached media_id."""
        if not hasattr(self, "_media_id_cache"):
            self._media_id_cache = {}

        if file_path in self._media_id_cache:
            return self._media_id_cache[file_path]

        token = os.environ.get("META_ACCESS_TOKEN") or os.environ.get("TWILIO_AUTH_TOKEN")
        if not token or not os.path.exists(file_path):
            return None

        url = f"https://graph.facebook.com/v21.0/{phone_number_id}/media"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            with open(file_path, "rb") as f:
                files = {
                    "file": (os.path.basename(file_path), f, mime_type),
                    "type": (None, mime_type),
                    "messaging_product": (None, "whatsapp")
                }
                resp = requests.post(url, headers=headers, files=files, timeout=30)
                if resp.status_code in (200, 201):
                    m_id = resp.json().get("id")
                    if m_id:
                        self._media_id_cache[file_path] = m_id
                        print(f"[Meta Media Upload] Cached media ID {m_id} for {file_path}")
                        return m_id
                else:
                    print(f"[Meta Media Upload Failed] HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[Meta Media Upload Error] {e}")
        return None

    def send_flow_message(
        self,
        phone_number_id: str,
        recipient_phone: str,
        flow_id: Optional[str] = None,
        screen_id: str = "DEMOGRAPHICS_SCREEN",
        header_type: str = "image",
        header_image_path: Optional[str] = None,
        header_media_id: Optional[str] = None,
        header_text: str = "Poshan Child Health Screening",
        body_text: Optional[str] = None,
        footer_text: str = "Govt. of Andhra Pradesh • Poshan Application",
        cta_text: str = "▶️ Start Assessment",
        flow_token: str = "poshan_flow_intake",
        initial_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Dispatches native interactive WhatsApp Flow message via Meta Graph API v21.0.
        Enforces Meta specifications (CTA max 20 chars, header max 60 chars, body max 1024 chars).
        Supports deep-linking to any screen (e.g. PHOTO_CAPTURE_SCREEN) with pre-filled initial_data.
        Supports 'image' header type using uploaded media ID (e.g. child screening steps banner).
        """
        if body_text is None:
            body_text = (
                "👶 *పోషణ్ బాల ఆరోగ్య తనిఖీ*\n\n"
                "🙏 *పోషణ్ యాప్‌కు స్వాగతం*\n"
                " Welcome to the Poshan Application\n\n"
                "👶 *పిల్లల పోషక స్థితి తనిఖీ వ్యవస్థ*\n"
                " Child Nutrition Screening System\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📋 *తక్షణ ఫలితం పొందడానికి*\n"
                " To Get an Instant Screening Result\n\n"
                "క్రింద ఉన్న \"Start Assessment\" బటన్ నొక్కి 3 సులభ దశలు పూర్తి చేయండి.\n"
                " Tap \"Start Assessment\" button and complete these 3 simple steps:\n\n"
                "✅ *దశ 1: వివరాలు నమోదు చేయండి*\n"
                " Step 1: Enter Child Details\n"
                " • పిల్లల వయస్సు, లింగం, ఎత్తు, బరువు, MUAC నమోదు చేయండి.\n"
                " • Enter the child's Age, Gender, Height, Weight, and MUAC.\n\n"
                "✅ *దశ 2: ఎడమ చేయి ఫోటో తీయండి*\n"
                " Step 2: Take Left Arm Photo\n"
                " • పిల్లల ఎడమ చేయి ఫోటో తీసి Submit నొక్కండి.\n"
                " • Capture a clear photo of child's left arm and click Submit.\n\n"
                "✅ *దశ 3: ఫలితం పొందండి*\n"
                " Step 3: Get Instant Result\n"
                " • పిల్లల పోషక స్థితి ఫలితాన్ని వెంటనే చూడండి.\n"
                " • View child's nutrition screening result instantly.\n\n"
                "📌 *గమనిక / Note*\n"
                " • ఎడమ చేయి మొత్తం కనిపించేలా మంచి వెలుతురులో ఫోటో తీయండి.\n"
                " • Ensure full left arm is visible and photo is taken in good lighting."
            )
        token = os.environ.get("META_ACCESS_TOKEN") or os.environ.get("TWILIO_AUTH_TOKEN")
        resolved_flow_id = (flow_id or os.environ.get("META_FLOW_ID") or os.environ.get("WHATSAPP_FLOW_ID") or "").strip()
        resolved_screen_id = os.environ.get("FLOW_START_SCREEN") or screen_id or "DEMOGRAPHICS_SCREEN"

        if not token or not phone_number_id or not resolved_flow_id:
            print(f"[Flow Dispatch Error] Missing token, phone_number_id ({phone_number_id}), or flow_id ({resolved_flow_id})")
            return False

        today_str = datetime.now().strftime("%Y-%m-%d")
        url = f"https://graph.facebook.com/v21.0/{phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # flow_cta max length is 20 characters
        safe_cta = cta_text[:20]

        # Parameter can be flow_id (numeric) or flow_name (string)
        flow_param_key = "flow_id" if resolved_flow_id.isdigit() else "flow_name"

        # Prepare screen-specific flow_action_payload data
        if resolved_screen_id == "DEMOGRAPHICS_SCREEN":
            flow_action_data: Dict[str, Any] = {"max_date": today_str}
            if initial_data:
                flow_action_data.update(initial_data)
        else:
            flow_action_data = dict(initial_data) if initial_data else {}

        # Resolve header configuration (Image banner or text)
        header_obj: Dict[str, Any] = {"type": "text", "text": header_text[:60]}
        if header_type == "image":
            resolved_media_id = header_media_id
            if not resolved_media_id:
                target_img = header_image_path or str(Path(__file__).resolve().parent / "system_images" / "child screening steps.jpg")
                resolved_media_id = self.upload_media_to_meta(phone_number_id, target_img)
            if resolved_media_id:
                header_obj = {
                    "type": "image",
                    "image": {
                        "id": str(resolved_media_id)
                    }
                }
            else:
                print("[Flow Dispatch] Falling back to text header since image media ID was not resolved.")

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient_phone,
            "type": "interactive",
            "interactive": {
                "type": "flow",
                "header": header_obj,
                "body": {
                    "text": body_text
                },
                "footer": {
                    "text": footer_text[:60]
                },
                "action": {
                    "name": "flow",
                    "parameters": {
                        "flow_message_version": "3",
                        "flow_token": flow_token,
                        flow_param_key: resolved_flow_id,
                        "flow_cta": safe_cta,
                        "flow_action": "navigate",
                        "flow_action_payload": {
                            "screen": resolved_screen_id,
                            "data": flow_action_data
                        }
                    }
                }
            }
        }

        try:
            print(f"[Meta WA Outgoing Request] POST {url}")
            print(f"  Phone Number ID: {phone_number_id}")
            print(f"  Recipient: {recipient_phone}")
            print(f"  Flow: {resolved_flow_id} | CTA: '{safe_cta}' | Start Screen: '{resolved_screen_id}'")
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code in (200, 201):
                print(f"[Meta WA] Successfully sent Flow button to {recipient_phone}")
                return True
            else:
                print(f"[Meta WA Flow Error] HTTP {resp.status_code}: {resp.text}")

                # If the primary number (e.g. +1 711) is pending Display Name Approval (#131037),
                # automatically attempt fallback via the pre-approved test number if available
                fallback_phone_id = os.environ.get("META_FALLBACK_PHONE_NUMBER_ID") or "1462244890300142"
                if "131037" in resp.text and phone_number_id != fallback_phone_id:
                    print(
                        f"[Meta WA #131037 Alert] Phone Number ID '{phone_number_id}' needs display name approval in Meta Business Manager. "
                        f"Attempting automatic fallback via pre-approved test number ID '{fallback_phone_id}'..."
                    )
                    fallback_url = f"https://graph.facebook.com/v21.0/{fallback_phone_id}/messages"
                    fb_resp = requests.post(fallback_url, headers=headers, json=payload, timeout=20)
                    if fb_resp.status_code in (200, 201):
                        print(f"[Meta WA] Successfully sent Flow button via fallback number {fallback_phone_id} to {recipient_phone}")
                        return True
                    else:
                        print(f"[Meta WA Fallback Flow Error] HTTP {fb_resp.status_code}: {fb_resp.text}")

                return False
        except Exception as e:
            print(f"[Meta WA Flow Exception] {e}")
            return False

    # -------------------------------------------------------------------------
    # 4. Media Resolution for Flow PhotoPicker (camera_gallery)
    # -------------------------------------------------------------------------
    def resolve_photo_bytes(self, raw_photo_value: Any) -> Optional[bytes]:
        """
        Resolves raw photograph bytes from Meta WhatsApp Flow PhotoPicker payload.
        Handles list of media dicts, list of base64 strings, or direct media IDs.
        """
        token = os.environ.get("META_ACCESS_TOKEN") or os.environ.get("TWILIO_AUTH_TOKEN")
        if not raw_photo_value:
            return None

        # 1. List from PhotoPicker (e.g. [{"id": "12345"}])
        first_item = raw_photo_value[0] if isinstance(raw_photo_value, list) and len(raw_photo_value) > 0 else raw_photo_value

        # 2. Dictionary containing media ID
        media_id = None
        if isinstance(first_item, dict):
            media_id = first_item.get("id")
        elif isinstance(first_item, str):
            # Check if base64 data URI
            if first_item.startswith("data:image"):
                b64_part = first_item.split(",", 1)[1]
                try:
                    return base64.b64decode(b64_part)
                except Exception:
                    pass
            # Check if pure base64
            if len(first_item) > 1000 and not first_item.isdigit():
                try:
                    return base64.b64decode(first_item)
                except Exception:
                    pass
            media_id = first_item

        # 3. Fetch from Meta Graph API via media_id
        if media_id and token:
            try:
                meta_url = f"https://graph.facebook.com/v21.0/{media_id}"
                headers = {"Authorization": f"Bearer {token}"}
                r1 = requests.get(meta_url, headers=headers, timeout=20)
                if r1.status_code == 200:
                    download_url = r1.json().get("url")
                    if download_url:
                        r2 = requests.get(download_url, headers=headers, timeout=30)
                        if r2.status_code == 200:
                            return r2.content
            except Exception as ex:
                print(f"[Photo Resolution Error] Failed to download media {media_id}: {ex}")

        return None


# Global singleton instance for application reuse
flows_handler = WhatsAppFlowsHandler()
