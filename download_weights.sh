#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-weights}"

FRCNN_URL="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl"
FRCNN_DIR="${OUT_DIR}/faster_rcnn"
FRCNN_OUT="${FRCNN_DIR}/model_final_b275ba.pkl"

DETR_URL="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
DETR_DIR="${OUT_DIR}/detr"
DETR_OUT="${DETR_DIR}/detr-r50-e632da11.pth"

mkdir -p "$FRCNN_DIR" "$DETR_DIR"

echo "Downloading Faster R-CNN R50-FPN 1x weights to $FRCNN_OUT"
curl -L "$FRCNN_URL" -o "$FRCNN_OUT"

echo "Downloading DETR R50 weights to $DETR_OUT"
curl -L "$DETR_URL" -o "$DETR_OUT"

echo "All downloads complete."
