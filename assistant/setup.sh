#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Multimodal Assistant Setup ==="
echo "Repo: $REPO_DIR"
echo ""

# Copy .env from template if it doesn't exist
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo "Created .env from .env.example — edit it to set HUGGINGFACE_TOKEN"
fi

# Source .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Step 1: Build base containers via jetson-containers
echo "── Step 1: Building base containers ──"
CONTAINERS=(open-webui speaches "kokoro-tts:fastapi" vila gemma_vlm llava)
for pkg in "${CONTAINERS[@]}"; do
    echo "Building $pkg ..."
    jetson-containers build "$pkg" || echo "WARNING: $pkg build failed, continuing..."
done

# Step 2: Build VLM bridge images
echo ""
echo "── Step 2: Building VLM bridge images ──"
cd "$SCRIPT_DIR/vlm-bridge"

for vlm in vila gemma llava; do
    case $vlm in
        vila)  BASE=$(autotag vila 2>/dev/null || echo "") ;;
        gemma) BASE=$(autotag gemma_vlm 2>/dev/null || echo "") ;;
        llava) BASE=$(autotag llava 2>/dev/null || echo "") ;;
    esac
    if [ -n "$BASE" ]; then
        echo "Building vlm-bridge:$vlm from $BASE"
        docker build --build-arg BASE_IMAGE="$BASE" --build-arg VLM_MODEL="$vlm" \
            -t "vlm-bridge:$vlm" .
    else
        echo "WARNING: No image found for $vlm, skipping bridge build"
    fi
done

# Step 3: Resolve image tags for docker-compose
echo ""
echo "── Step 3: Setup complete ──"
echo ""
echo "Usage:"
echo "  cd $SCRIPT_DIR"
echo ""
echo "  # Edit .env to select your model:"
echo "  #   COMPOSE_PROFILES=vila   (or gemma, llava)"
echo "  #   HUGGINGFACE_TOKEN=hf_... (needed for gated models)"
echo ""
echo "  # Start the assistant:"
echo "  docker compose up -d"
echo ""
echo "  # Open the web UI:"
echo "  echo http://\$(hostname -I | awk '{print \$1}'):8080"
echo ""
echo "  # Switch models:"
echo "  docker compose down"
echo "  # Edit .env: COMPOSE_PROFILES=llava"
echo "  docker compose up -d"
