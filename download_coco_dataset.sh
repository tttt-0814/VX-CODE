#!/usr/bin/env bash
set -euo pipefail

# Download MS COCO 2017 train/val images and annotations.
# Usage: ./download_coco_dataset.sh [output_dir]

OUT_DIR="${1:-datasets/coco}"
IMAGES_DIR="$OUT_DIR/val2017"
ANN_DIR="$OUT_DIR/annotations"
mkdir -p "$IMAGES_DIR" "$ANN_DIR"

# URLs from the official COCO site (val images + annotations).
VAL_IMAGES_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

fetch() {
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then
    echo "[skip] $dest already exists"
    return
  fi
  echo "[dl ] $url -> $dest"
  curl -L "$url" -o "$dest"
}

fetch "$VAL_IMAGES_URL" "$OUT_DIR/val2017.zip"
fetch "$ANN_URL" "$ANN_DIR/annotations_trainval2017.zip"

echo "Extracting val images..."
unzip -n "$OUT_DIR/val2017.zip" -d "$OUT_DIR" >/dev/null

echo "Extracting annotations..."
unzip -n "$ANN_DIR/annotations_trainval2017.zip" -d "$OUT_DIR" >/dev/null

echo "Done. Images in $OUT_DIR/val2017, annotations in $ANN_DIR."
