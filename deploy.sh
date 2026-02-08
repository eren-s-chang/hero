#!/usr/bin/env bash
# ===========================================================================
# FormPerfect – Vultr VPS deploy script
# ===========================================================================
# Usage:
#   ./deploy.sh <VULTR_IP> [SSH_KEY_PATH]
#
# Prerequisites:
#   - A Vultr VPS running Ubuntu 22.04 / 24.04
#   - SSH access (root or a sudo user)
#   - Your GEMINI_API_KEY ready
#
# What this does:
#   1. Installs Docker + Docker Compose on the VPS
#   2. Copies the backend to the VPS
#   3. Starts all services via docker compose
#
# The frontend is deployed separately on Vercel.
# ===========================================================================

set -euo pipefail

REMOTE_IP="${1:?Usage: ./deploy.sh <VULTR_IP> [SSH_KEY_PATH]}"
SSH_KEY="${2:-$HOME/.ssh/id_rsa}"
REMOTE_USER="root"
REMOTE_DIR="/opt/form-perfect"

SSH_CMD="ssh -o StrictHostKeyChecking=no -i $SSH_KEY $REMOTE_USER@$REMOTE_IP"
SCP_CMD="scp -o StrictHostKeyChecking=no -i $SSH_KEY"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  FormPerfect → Deploying to $REMOTE_IP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ------------------------------------------------------------------
# Step 1: Install Docker on the VPS (idempotent)
# ------------------------------------------------------------------
echo "▸ Installing Docker on VPS..."
$SSH_CMD << 'ENDSSH'
  if ! command -v docker &>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
  fi
  echo "Docker $(docker --version)"
ENDSSH

# ------------------------------------------------------------------
# Step 2: Sync backend to VPS
# ------------------------------------------------------------------
echo "▸ Syncing backend files to VPS..."
rsync -avz --delete \
  -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY" \
  --include='backend/***' \
  --include='docker-compose.yml' \
  --include='.env.example' \
  --exclude='*' \
  ./ "$REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/"

# ------------------------------------------------------------------
# Step 4: Prompt for .env if it doesn't exist on VPS
# ------------------------------------------------------------------
$SSH_CMD << ENDSSH
  if [ ! -f $REMOTE_DIR/.env ]; then
    echo ""
    echo "⚠  No .env found on VPS. Creating one..."
    read -p "Enter your GEMINI_API_KEY: " GEMINI_KEY
    echo "GEMINI_API_KEY=\$GEMINI_KEY" > $REMOTE_DIR/.env
    echo ".env created."
  fi
ENDSSH

# ------------------------------------------------------------------
# Step 5: Start services
# ------------------------------------------------------------------
echo "▸ Starting services..."
$SSH_CMD << ENDSSH
  cd $REMOTE_DIR
  docker compose down --remove-orphans 2>/dev/null || true
  docker compose up -d --build
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  ✅ FormPerfect backend is live!"
  echo "  🌐 API: http://$REMOTE_IP"
  echo "  📋 Health: http://$REMOTE_IP/health"
  echo "  📊 Logs: docker compose logs -f"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ENDSSH
