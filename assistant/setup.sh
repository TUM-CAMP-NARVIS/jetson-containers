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

# Step 3: Resolve image tags and write to .env
echo ""
echo "── Step 3: Resolving image tags ──"
cd "$SCRIPT_DIR"

resolve_tag() {
    local pkg="$1"
    # autotag returns the best matching image (local > registry > build)
    # Use --quiet and pipe to avoid interactive prompts
    local tag
    tag=$(autotag "$pkg" --quiet 2>/dev/null || docker images --format '{{.Repository}}:{{.Tag}}' | grep "$pkg" | head -1 || echo "")
    echo "$tag"
}

WEBUI_TAG=$(resolve_tag open-webui)
STT_TAG=$(resolve_tag speaches)
TTS_TAG=$(resolve_tag kokoro-tts:fastapi)

# Append resolved tags to .env (avoiding duplicates)
for var_line in \
    "OPEN_WEBUI_IMAGE=$WEBUI_TAG" \
    "SPEACHES_IMAGE=$STT_TAG" \
    "KOKORO_TTS_IMAGE=$TTS_TAG"; do
    var_name="${var_line%%=*}"
    # Remove old entry if present, then append
    sed -i "/^${var_name}=/d" "$SCRIPT_DIR/.env" 2>/dev/null || true
    if [ -n "${var_line#*=}" ]; then
        echo "$var_line" >> "$SCRIPT_DIR/.env"
    fi
done

echo "Resolved tags written to .env"
echo ""
echo "── Step 4: Setup complete ──"
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
