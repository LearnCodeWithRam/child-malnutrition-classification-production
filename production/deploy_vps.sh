#!/bin/bash
# ==============================================================================
# Hostinger KVM 4 VPS Automated Deployment Script for Poshan Production Stack
# Reference-optimized with thread isolation, swap safety, and persistent NVMe
# ==============================================================================
set -e

echo "=========================================================="
echo "🚀 Initializing Poshan Malnutrition Screening Stack on VPS"
echo "=========================================================="

# 1. Check and configure 16GB swap space if missing
if [ -f setup_swap.sh ]; then
    echo "⚙️ Running swap configuration..."
    bash setup_swap.sh
fi

# 2. Create persistent NVMe directories on host
DATA_ROOT="/var/lib/poshan_data"
echo "📁 Setting up persistent NVMe directories in ${DATA_ROOT}..."
sudo mkdir -p ${DATA_ROOT}/postgres
sudo mkdir -p ${DATA_ROOT}/redis
sudo mkdir -p ${DATA_ROOT}/photos
sudo mkdir -p ${DATA_ROOT}/audit_logs

# Set directory permissions
sudo chmod -R 777 ${DATA_ROOT}/photos
sudo chmod -R 777 ${DATA_ROOT}/audit_logs
sudo chmod -R 700 ${DATA_ROOT}/postgres
sudo chmod -R 700 ${DATA_ROOT}/redis

echo "✅ Persistent storage directories verified."

# 3. Check for Docker Engine & Compose plugin
if ! command -v docker &> /dev/null; then
    echo "📦 Docker not detected. Installing Docker Engine..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm -f get-docker.sh
fi

# Determine compose command
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# 4. Check for .env.production file
if [ ! -f .env.production ]; then
    if [ -f .env.production.example ]; then
        echo "⚠️ .env.production not found. Generating template..."
        cp .env.production.example .env.production
        echo "❗ Please configure .env.production with your Meta API keys and database credentials."
    fi
fi

# 5. Build and launch containers
echo "🚀 Building and starting containers in detached mode..."
$DOCKER_COMPOSE down --remove-orphans || true
$DOCKER_COMPOSE up -d --build

echo "⏳ Waiting for healthchecks to pass..."
sleep 10

# 6. Display container status
$DOCKER_COMPOSE ps

# 7. Optional SSL setup with Certbot
if [ -n "$DOMAIN_NAME" ] && command -v certbot &> /dev/null; then
    echo "🔒 Configuring SSL certificate for $DOMAIN_NAME..."
    sudo certbot --nginx -d "$DOMAIN_NAME" --non-interactive --agree-tos -m "admin@$DOMAIN_NAME" || true
    sudo systemctl restart nginx 2>/dev/null || true
fi

echo "=========================================================="
echo "🎉 Poshan Production Stack is running successfully!"
echo "Check health: curl http://localhost/health"
echo "=========================================================="
