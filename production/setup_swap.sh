#!/bin/bash
# setup_swap.sh - Create 16GB Swap space on Hostinger Linux VPS
# Protects Celery worker processes from OOM (Out-of-Memory) crashes during heavy peak workloads.
set -e

echo "🔍 Checking current swap space..."
sudo swapon --show

if [ $(sudo swapon --show | wc -l) -gt 1 ]; then
    echo "✅ Swap is already configured:"
    sudo swapon --show
    exit 0
fi

echo "📦 Creating 16GB swap file at /swapfile on NVMe SSD..."
sudo fallocate -l 16G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=16384

echo "🔒 Setting secure permissions..."
sudo chmod 600 /swapfile

echo "⚙️ Initializing swap filesystem..."
sudo mkswap /swapfile

echo "🚀 Activating swap..."
sudo swapon /swapfile

echo "📝 Persisting swap across system reboots in /etc/fstab..."
if ! grep -q "/swapfile" /etc/fstab; then
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# Set optimal swappiness for database & Python workloads (low swappiness to favor RAM)
sudo sysctl vm.swappiness=15
if ! grep -q "vm.swappiness" /etc/sysctl.conf; then
    echo 'vm.swappiness=15' | sudo tee -a /etc/sysctl.conf
fi

echo "=========================================================="
echo "🎉 16GB Swap configuration completed successfully!"
free -h
echo "=========================================================="

