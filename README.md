# 🏥 Poshan AI - Production Child Malnutrition Classification System

[![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Celery](https://img.shields.io/badge/Celery-37814A?logo=celery&logoColor=white)](https://docs.celeryq.dev/)
[![Redis](https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Nginx](https://img.shields.io/badge/Nginx-009639?logo=nginx&logoColor=white)](https://nginx.org/)

Production-grade, asynchronous, containerized screening suite for the **Poshan Frontline Child Malnutrition Screening Platform (PediaMorph AI)**. Designed and battle-tested for high-concurrency WhatsApp Cloud API / Flows workloads with zero data loss on Linux VPS instances (Hostinger KVM 4 / Ubuntu 22.04 or 24.04 LTS).

---

## 🏛️ System Architecture

```text
                               ┌────────────────────────────────────────────────────────┐
                               │                    HOSTINGER LINUX VPS                 │
                               │             (4 vCPUs, 16 GB RAM, 200 GB NVMe)          │
                               └────────────────────────────────────────────────────────┘
                                                           │
                                 ┌─────────────────────────┼─────────────────────────┐
                                 │ Port 80/443             │ Port 8888               │ Port 5555
                                 ▼                         ▼                         ▼
                    ┌─────────────────────────┐ ┌────────────────────┐    ┌────────────────────┐
                    │       Nginx Proxy       │ │  Dozzle Dashboard  │    │  Flower Dashboard  │
                    │ (SSL, Rate Limits, 35M) │ │ (Logs + CPU/RAM)   │    │(Celery Task Monitor│
                    └────────────┬────────────┘ └────────────────────┘    └──────────┬─────────┘
                                 │                                                   │
                       HTTP Proxy (Port 8000)                                        │
                                 ▼                                                   │
┌─────────────────────┐ Webhook (<50ms) ┌─────────────────────┐                      │
│    Meta WhatsApp    │ ──────────────> │ Gunicorn + Uvicorn  │                      │
│ Cloud API/Flows v3  │ <────────────── │ FastAPI Web Gateway │                      │
└─────────────────────┘ HTTP 200 OK     └──────────┬──────────┘                      │
                                                   │                                 │
                                    1. Direct binary write to NVMe                   │
                                    2. Computes SHA-256 integrity hash               │
                                    3. Dispatches task to Redis (<10ms)              │
                                                   │                                 │
                        ┌──────────────────────────┴──────────────────────────┐      │
                        ▼                                                     ▼      ▼
           ┌─────────────────────────┐                           ┌─────────────────────────┐
           │      Redis Broker       │ <─────────────────────────┤      PostgreSQL 15      │
           │   (Persistent AOF)      │                           │   (Metadata & Audits)   │
           └────────────┬────────────┘                           └────────────▲────────────┘
                        │                                                     │
                 Celery Task Queue                                 Updates Status & Metrics
                        │                                                     │
                        ▼                                                     │
           ┌─────────────────────────┐                                        │
           │   Celery Worker Pool    │ ───────────────────────────────────────┘
           │   (Concurrency = 4)     │
           │  MediaPipe + XGBoost/   │ ──> Uploads Lossless Speedometer Visual Report
           │  Cloud EGAT Inference   │
           └────────────┬────────────┘
                        │
                        ▼
    Persistent Host Mount: `${POSHAN_DATA_DIR:-./poshan_data}`
    ├── `photos/`       (Zero-loss raw JPEG/PNG binaries on NVMe)
    ├── `postgres/`     (Relational DB files)
    ├── `redis/`        (Queue broker state)
    └── `audit_logs/`   (Generated speedometer risk gauges)
```

---

## 📂 Repository Directory Structure

```text
├── production/                         # Containerized production runtime
│   ├── Dockerfile                      # Production container image (PyTorch CPU, MediaPipe, OpenCV)
│   ├── docker-compose.yml              # 7-service orchestration stack
│   ├── requirements_prod.txt           # Production Python dependencies
│   ├── app_prod.py                     # High-performance FastAPI asynchronous webhook gateway
│   ├── celery_app.py                   # Celery broker & result backend initialization
│   ├── tasks.py                        # Asynchronous worker pipeline & WhatsApp reply dispatcher
│   ├── database.py                     # PostgreSQL connection pool & SQLAlchemy engine
│   ├── db_models.py                    # Schema definitions (ChildAssessment, ChildPhoto, FieldAudit)
│   ├── storage_manager.py              # Zero-loss NVMe photo writer & SHA-256 validator
│   ├── data_management.py              # Retention, pruning, and streaming bulk data exports
│   ├── deploy_vps.sh                   # Automated one-click VPS deployment script
│   ├── setup_swap.sh                   # 4 GB swap configuration helper for VPS
│   ├── nginx/
│   │   └── nginx.conf                  # Production reverse proxy config (35MB upload, gzip, security)
│   ├── .env.production.example         # Production environment template
│   └── README.md                       # Deep-dive production engineering manual
├── poselandmark_model/                 # MediaPipe Pose & Selfie segmentation models
│   ├── pose_landmarker_heavy.task
│   └── selfie_multiclass_256x256.tflite
├── saved_models/                       # Trained ML weights and allometric regression models
│   ├── cloud_egat_model.pt
│   ├── xgboost_multimodal_model.joblib
│   ├── xgboost_shape_model.joblib
│   ├── allometric_regression_model.joblib
│   └── ...
├── system_images/                      # Step-by-step guidance visuals sent to workers
├── sample_images/                      # Verification and test image suite
├── static/                             # Web interface assets
├── app.py                              # Core algorithmic pipeline
├── classifier_engine.py                # Multi-modal feature extraction and model inference
├── image_quality_guard.py              # Lighting, blur, and elbow visibility quality guard
├── risk_gauge_visualizer.py            # Clinical speedometer chart renderer
├── whatsapp_flows_handler.py           # Meta WhatsApp Flows v3 encryption & payload handler
└── .env.production.example             # Root copy of environment configuration template
```

---

## 🚀 Quick Start Deployment (Hostinger VPS / Linux)

### 1. Clone Repository & Setup Directory
```bash
git clone https://github.com/LearnCodeWithRam/child-malnutrition-classification-production.git poshan-prod
cd poshan-prod/production
```

### 2. Configure Environment Variables
Copy the example environment file and edit it with your production secrets:
```bash
cp .env.production.example .env.production
nano .env.production
```

Configure your domain/IP, database password, Redis password, and Meta WhatsApp Cloud API credentials:
```env
PUBLIC_APP_URL=https://your-domain-or-vps-ip
POSTGRES_PASSWORD=your_secure_postgres_password
REDIS_PASSWORD=your_secure_redis_password
DATABASE_URL=postgresql://poshan_user:your_secure_postgres_password@postgres:5432/poshan_db
CELERY_BROKER_URL=redis://:your_secure_redis_password@redis:6379/0
CELERY_RESULT_BACKEND=redis://:your_secure_redis_password@redis:6379/1

# Meta WhatsApp Cloud API
META_PHONE_NUMBER_ID=your_phone_number_id
META_ACCESS_TOKEN=your_permanent_access_token
META_APP_SECRET=your_app_secret
META_VERIFY_TOKEN=poshan_secret_webhook_token_2026
META_FLOW_ID=your_flow_id
META_FLOW_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----"
```

### 3. Launch Container Stack
```bash
docker compose up -d --build
```

### 4. Verify Services
Check running containers:
```bash
docker compose ps
```

You should see 7 running services:
- `poshan_nginx` (Port 80 / 443)
- `poshan_web` (FastAPI Gateway)
- `poshan_celery_worker` (ML Task Queue)
- `poshan_postgres` (PostgreSQL 15 Database)
- `poshan_redis` (Redis 7 Message Broker)
- `poshan_dozzle` (Log Monitoring on `:8888`)
- `poshan_flower` (Celery Monitoring on `:5555`)

---

## 📊 Live Monitoring Dashboards

| Dashboard | URL | Credentials / Notes |
| :--- | :--- | :--- |
| **Dozzle (Log Viewer)** | `http://<YOUR_VPS_IP>:8888` | Live real-time container logs, CPU/Memory telemetry |
| **Flower (Task Monitor)** | `http://<YOUR_VPS_IP>:5555` | Configured via `FLOWER_BASIC_AUTH` in `.env.production` |
| **Health Check API** | `http://<YOUR_VPS_IP>/health` | JSON status of Web, Database, and Redis connections |

---

## 🔒 Data Persistence & Storage Guarantees

All child screening photos, relational database records, and Redis snapshots are bound to host NVMe storage (`production/poshan_data` or `/var/lib/poshan_data`).

- **Container Reboots**: Rebuilding, restarting, or upgrading containers (`docker compose down && docker compose up -d --build`) **does not** delete or affect database records or images.
- **Approach B Zero-Loss Photos**: Raw incoming image bytes are written directly to NVMe disk via binary stream without re-compression, and validated via SHA-256 integrity checksums before ML inference.

---

## 🛠️ Operational Commands

```bash
# View aggregated real-time logs
docker compose logs -f

# Inspect Celery worker logs
docker compose logs -f celery_worker

# Restart the stack
docker compose restart

# Graceful shutdown
docker compose down
```

---

## 📄 License
Government of Andhra Pradesh • Poshan Malnutrition Screening Initiative. All rights reserved.
