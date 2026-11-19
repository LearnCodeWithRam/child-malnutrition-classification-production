# 🏥 Poshan AI - Production Deployment & Operations Guide

Production-grade, containerized deployment suite for the **Poshan Child Malnutrition Screening System (PediaMorph AI)**, optimized for deployment on a **Hostinger KVM 4 VPS (4 vCPUs, 16 GB RAM, 200 GB NVMe SSD)** running Ubuntu 22.04 / 24.04 LTS.

---

## 📑 Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Approach B: Lossless Storage & Data Persistence](#2-approach-b-lossless-storage--data-persistence)
3. [Server Directory, Mounts & Port Structure](#3-server-directory-mounts--port-structure)
4. [Prerequisites & Initial VPS Setup](#4-prerequisites--initial-vps-setup)
5. [Environment Variables Configuration](#5-environment-variables-configuration)
6. [Deployment Instructions](#6-deployment-instructions)
7. [API Reference & Operational Endpoints](#7-api-reference--operational-endpoints)
8. [Real-Time Web Dashboards (Dozzle & Flower)](#8-real-time-web-dashboards-dozzle--flower)
9. [Maintenance, Monitoring & Disaster Recovery](#9-maintenance-monitoring--disaster-recovery)
10. [Automated Verification & Testing](#10-automated-verification--testing)

---

## 1. Architecture Overview

The production architecture is fully decoupled to handle high-concurrency WhatsApp traffic from frontline Anganwadi workers across Andhra Pradesh:

```text
                               ┌────────────────────────────────────────────────────────┐
                               │                  HOSTINGER KVM 4 VPS                   │
                               │           (4 vCPUs, 16 GB RAM, 200 GB NVMe)            │
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
└─────────────────────┘ HTTP 200 OK +   └──────────┬──────────┘                      │
                        Receipt                    │                                 │
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
           │ OMP/OpenBLAS Isolated   │ ──> Dispatches Speedometer Visual Report
           └────────────┬────────────┘
                        │
                        ▼
    Persistent Host Mount: `${POSHAN_DATA_DIR:-./poshan_data}` (or `/var/lib/poshan_data/`)
    ├── `photos/`       (Zero-loss raw JPEG/PNG binaries on NVMe)
    ├── `postgres/`     (Relational DB files)
    ├── `redis/`        (Queue broker state)
    └── `audit_logs/`   (Generated speedometer risk gauges)
```

---

## 2. Approach B: Lossless Storage & Data Persistence

### Why Approach B?
Standard image storage libraries (like Pillow or OpenCV) automatically re-encode photos, causing compression artifacts and altering landmark pixel coordinates.
- **Direct Binary Write (`"wb"` + `os.fsync`)**: Incoming raw bytes from the camera sensor are saved directly to NVMe SSD without touching image codecs.
- **SHA-256 Integrity Verification**: An exact SHA-256 hash is computed and recorded in PostgreSQL. Every Celery worker validates this hash before running inference to guarantee zero data corruption.
- **Zero S3 Cloud Costs**: Saves thousands of dollars in cloud object storage bills by utilizing your 200 GB NVMe SSD.

### Will Data Disappear During Container Restarts?
**NO.** All data is stored in host bind mounts outside of the Docker containers. Rebuilding, updating, or restarting containers (`docker compose down && docker compose up -d --build`) will **never erase** database rows or child photos.

---

## 3. Server Directory, Mounts & Port Structure

| Service | Container Name | Port Mapping | Host Persistence Path | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Web Gateway** | `poshan_web` | `8000:8000` | N/A | FastAPI + Gunicorn (4 Uvicorn workers) |
| **Reverse Proxy** | `poshan_nginx` | `80:80`, `443:443` | N/A | Nginx SSL reverse proxy |
| **Celery Workers**| `poshan_celery_worker`| Internal | N/A | 4 Parallel ML inference workers |
| **PostgreSQL 15** | `poshan_postgres` | `127.0.0.1:5432` | `${POSHAN_DATA_DIR}/postgres` | Relational clinical metadata |
| **Redis 7** | `poshan_redis` | `127.0.0.1:6379` | `${POSHAN_DATA_DIR}/redis` | Queue broker with persistent AOF |
| **Dozzle** | `poshan_dozzle` | `8888:8080` | `/var/run/docker.sock` | Real-time container logs & CPU/RAM |
| **Flower** | `poshan_flower` | `5555:5555` | N/A | Celery task & Redis queue monitor |
| **Raw Photos** | N/A | N/A | `${POSHAN_DATA_DIR}/photos` | Uncompressed camera photos on NVMe |
| **Audit Logs** | N/A | N/A | `${POSHAN_DATA_DIR}/audit_logs` | Speedometer risk gauge graphics |

---

## 4. Prerequisites & Initial VPS Setup

### Step 1: Connect to your Hostinger VPS via SSH
```bash
ssh root@<YOUR_VPS_IP>
```

### Step 2: Configure 16 GB Linux Swap Space
On a 16 GB RAM server, a 16 GB swap file prevents the Linux Out-Of-Memory (OOM) killer from terminating Python worker processes during heavy burst screening:
```bash
cd /root/poshan_project/production
bash setup_swap.sh
```

### Step 3: Create Persistent Storage Directories
```bash
sudo mkdir -p /var/lib/poshan_data/postgres
sudo mkdir -p /var/lib/poshan_data/redis
sudo mkdir -p /var/lib/poshan_data/photos
sudo mkdir -p /var/lib/poshan_data/audit_logs

sudo chmod -R 777 /var/lib/poshan_data/photos
sudo chmod -R 777 /var/lib/poshan_data/audit_logs
sudo chmod -R 700 /var/lib/poshan_data/postgres
sudo chmod -R 700 /var/lib/poshan_data/redis
```

---

## 5. Environment Variables Configuration

Copy the template and configure your production environment:
```bash
cp .env.production.example .env.production
nano .env.production
```

### Required Configuration Keys:
```ini
# Public URL (your domain or VPS IP)
PUBLIC_APP_URL=https://poshan.ap.gov.in

# PostgreSQL Credentials
POSTGRES_DB=poshan_db
POSTGRES_USER=poshan_user
POSTGRES_PASSWORD=SetAStrongRandomPassword2026!
DATABASE_URL=postgresql://poshan_user:SetAStrongRandomPassword2026!@postgres:5432/poshan_db

# Redis Password
REDIS_PASSWORD=SetAStrongRedisPassword2026!
CELERY_BROKER_URL=redis://:SetAStrongRedisPassword2026!@redis:6379/0
CELERY_RESULT_BACKEND=redis://:SetAStrongRedisPassword2026!@redis:6379/1

# Secret Admin Key for Data Export & Deletion APIs
ADMIN_API_KEY=SetASecretAdminKey2026!

# Real-Time Monitoring Dashboards Configuration
DOZZLE_PORT=8888
FLOWER_PORT=5555
FLOWER_BASIC_AUTH=admin:SetAStrongFlowerPassword2026!

# Meta WhatsApp Cloud API Credentials
META_PHONE_NUMBER_ID=your_meta_phone_number_id
META_ACCESS_TOKEN=your_permanent_system_user_access_token
META_APP_SECRET=your_meta_app_secret
META_VERIFY_TOKEN=poshan_secret_webhook_token_2026
META_FLOW_ID=your_registered_flow_id
```

---

## 6. Deployment Instructions

### Method A: Automated 1-Click Deployment (Recommended)
```bash
bash deploy_vps.sh
```
This script checks swap, verifies permissions, builds container images, sets up database tables, and starts all services in detached mode.

### Method B: Manual Step-by-Step Deployment
```bash
# 1. Build and launch all 7 containers
docker compose up -d --build

# 2. Verify all services are healthy and running
docker compose ps

# 3. Inspect real-time Celery worker logs
docker compose logs -f celery_worker
```

---

## 7. API Reference & Operational Endpoints

### 1. Health & Resource Monitoring
- **Endpoint**: `GET /health`
- **Description**: Returns live status of PostgreSQL, Redis broker, Celery worker pool, and free NVMe SSD gigabytes.
```bash
curl http://localhost:8000/health
```

### 2. High-Capacity Bulk Data Export (8GB+ Capacity)
- **Endpoint**: `GET /api/v1/admin/export/all-data`
- **Headers**: `X-Admin-Key: <ADMIN_API_KEY>`
- **Description**: Generates an on-the-fly streaming ZIP directly from NVMe storage with standard ZIP64. Delivers multi-gigabyte exports while maintaining a constant server RAM footprint of **<20 MB**.
- **Query Parameters**:
  - `awc_code` *(optional)*: Filter by Anganwadi Center code.
  - `start_date` *(optional)*: Filter by start date (`YYYY-MM-DD`).
  - `end_date` *(optional)*: Filter by end date (`YYYY-MM-DD`).
  - `include_photos` *(optional, default `true`)*: Include raw uncompressed photos.
  - `include_audit_visuals` *(optional, default `true`)*: Include generated speedometer gauges.

```bash
# Export entire state dataset
curl -H "X-Admin-Key: your_admin_key" \
     "http://localhost:8000/api/v1/admin/export/all-data" \
     --output state_poshan_export.zip

# Export a single AWC center's records
curl -H "X-Admin-Key: your_admin_key" \
     "http://localhost:8000/api/v1/admin/export/all-data?awc_code=28548190419" \
     --output awc_28548190419.zip
```

### 3. Individual Child Data Deletion API
- **Endpoint**: `DELETE /api/v1/admin/child/data` (or `POST /api/v1/admin/child/delete`)
- **Headers**: `X-Admin-Key: <ADMIN_API_KEY>`
- **Description**: Permanently purges a child's raw photos from NVMe disk, removes diagnostic audit logs, and cascades record deletion in PostgreSQL.

```bash
# Via DELETE request
curl -X DELETE -H "X-Admin-Key: your_admin_key" \
     "http://localhost:8000/api/v1/admin/child/data?awc_code=28548190419&serial_number=SN-001"

# Via POST request (JSON payload)
curl -X POST -H "X-Admin-Key: your_admin_key" \
     -H "Content-Type: application/json" \
     -d '{"awc_code": "28548190419", "serial_number": "SN-001"}' \
     "http://localhost:8000/api/v1/admin/child/delete"
```

---

## 8. Real-Time Web Dashboards (Dozzle & Flower)

### 📊 1. Dozzle Dashboard (Live Container Logs & CPU/RAM Metrics)
- **URL**: 👉 **`http://localhost:8888`**
- **Purpose**:
  - Live color-coded streaming log viewer for all 7 containers (`poshan_web`, `poshan_celery_worker`, `poshan_postgres`, `poshan_redis`, `poshan_nginx`, etc.).
  - Real-time CPU % and Memory (MB) utilization meters.
  - Multi-container split-screen log monitoring.
  - RAM overhead: < 15 MB.

### 🌸 2. Flower Dashboard (Celery AI Pipeline & Redis Task Queue Monitor)
- **URL**: 👉 **`http://localhost:5555`**
- **Default Login**: `admin` / `poshan_admin_2026` (configured via `FLOWER_BASIC_AUTH` in `.env`)
- **Purpose**:
  - Real-time tracking of every child screening task (`process_child_screening_task`).
  - Execution speed metrics (e.g. `0.68s` per screening).
  - Queue depth and worker concurrency gauges.
  - Detailed stack traces for any failed tasks.
  - RAM overhead: ~30 MB.

---

## 9. Maintenance, Monitoring & Disaster Recovery

### Common Management Commands
```bash
# View live status of containers
docker compose ps

# View live Celery worker logs
docker compose logs -f celery_worker

# Restart a specific service without affecting others
docker compose restart web
docker compose restart celery_worker

# Stop all services (data remains completely safe on host)
docker compose down

# Upgrade codebase and rebuild containers
git pull
docker compose up -d --build
```

### Database Backup & Restore
```bash
# Backup PostgreSQL database to host NVMe
docker exec -t poshan_postgres pg_dump -U poshan_user poshan_db > /root/backup_$(date +%Y%m%d).sql

# Restore from backup
cat /root/backup_20261118.sql | docker exec -i poshan_postgres psql -U poshan_user -d poshan_db
```

---

## 9. Automated Verification & Testing

To test the database schema, lossless storage manager, SHA-256 integrity checker, 8GB+ streaming export, and child data deletion locally or on the VPS:

```bash
python test_prod_stack.py
```

Expected Output:
```text
============================================================
🔍 STEP 1: Verifying Database Schema & Engine Initialization...
✅ Tables created: workers, child_assessments, child_photos, audit_logs

🔍 STEP 2: Testing Approach B (Lossless Raw Photo Saving + SHA-256)...
📄 Using real clinical sample: 03f74f_image1_without_ruler.jpg (226,038 bytes)
🔒 Computed SHA-256 on incoming bytes: cf4eadc4a3bd17c58a1a4ef7cc299cec86c38d5f2c3ecbb0d618f4ab428ba059
💾 Written to NVMe path: .../20261118_120000_test1234.jpg
📊 PostgreSQL Recorded Photo ID: 1
📊 PostgreSQL Recorded Size: 226,038 bytes
📊 PostgreSQL Recorded Hash: cf4eadc4a3bd17c58a1a4ef7cc299cec86c38d5f2c3ecbb0d618f4ab428ba059
✅ VERIFIED: Bit-for-bit identical! Zero compression, zero artifact, exact SHA-256 match.

🔍 STEP 3: Verifying Relationships & Audit Logging...
✅ ORM Relationship query successful: Assessment -> 1 Photo(s), 1 Audit Log(s)

🔍 STEP 4: Verifying Multi-Gigabyte Streaming Export Generation...
📦 Archive generated successfully! Files inside: ['metadata_assessments.csv', 'metadata_assessments.json', 'raw_photos/20261118_120000_test1234.jpg']
✅ VERIFIED: Streaming export ZIP is valid and contains exact lossless photo & metadata!

🔍 STEP 5: Verifying Individual Child Data Deletion by AWC & Serial Number...
🗑️ Deletion result: {'status': 'success', 'deleted_assessments_count': 1, 'deleted_photos_count': 1, ...}
✅ VERIFIED: Physical photo file removed from NVMe disk.
✅ VERIFIED: Database records cascaded and removed from PostgreSQL.
✅ VERIFIED: Repeat deletion gracefully returns 'not_found'.

============================================================
🎉 ALL PRODUCTION CHECKS PASSED SUCCESSFULLY!
============================================================
```

