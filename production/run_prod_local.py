"""
Local Runner for Poshan Production Stack.
Runs the production FastAPI server locally without Docker or external services,
using SQLite fallback and local NVMe/SSD directory for testing.
"""
import os
import sys
from pathlib import Path

# Setup paths
PROD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROD_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local environment configuration for zero-dependency testing
os.environ.setdefault("DATABASE_URL", f"sqlite:///{PROD_DIR / 'poshan_local_prod.db'}")
os.environ.setdefault("RAW_PHOTOS_DIR", str(PROD_DIR / "local_test_photos"))
os.environ.setdefault("ADMIN_API_KEY", "poshan_admin_secret_2026")
os.environ.setdefault("META_VERIFY_TOKEN", "poshan_secret_webhook_token_2026")

import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print("🚀 Starting Poshan Production Stack Locally (Zero Docker Dependencies)")
    print("=" * 65)
    print(f"📁 Local Database: {os.environ['DATABASE_URL']}")
    print(f"📁 Local Photos Directory: {os.environ['RAW_PHOTOS_DIR']}")
    print(f"🔑 Admin Secret Key: {os.environ['ADMIN_API_KEY']}")
    print("🌐 Swagger API Documentation: http://127.0.0.1:8000/docs")
    print("🌐 Health Check Endpoint:     http://127.0.0.1:8000/health")
    print("=" * 65)

    uvicorn.run(
        "production.app_prod:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )

