#!/bin/bash

set -e

VERSION="2.0.0"
PACKAGE_NAME="chair_ai_training_v${VERSION}"

echo "========================================="
echo "Package Chair AI Training System"
echo "Version: ${VERSION}"
echo "========================================="

cd /home/harleyw/ai_practice/chair_ai_training

mkdir -p "releases"

if [ -d "${PACKAGE_NAME}" ]; then
    rm -rf "${PACKAGE_NAME}"
fi

mkdir -p "${PACKAGE_NAME}"

echo "[1/4] Copying core source files..."

cp *.py "${PACKAGE_NAME}/" 2>/dev/null || true
cp *.md "${PACKAGE_NAME}/" 2>/dev/null || true
cp *.txt "${PACKAGE_NAME}/" 2>/dev/null || true
cp *.sh "${PACKAGE_NAME}/" 2>/dev/null || true

echo "[2/4] Copying module directories..."

cp -r env "${PACKAGE_NAME}/env" 2>/dev/null || true
cp -r training "${PACKAGE_NAME}/training" 2>/dev/null || true
cp -r utils "${PACKAGE_NAME}/utils" 2>/dev/null || true
cp -r api "${PACKAGE_NAME}/api" 2>/dev/null || true
cp -r export "${PACKAGE_NAME}/export" 2>/dev/null || true

echo "[3/4] Copying test and documentation files..."

cp test_*.py "${PACKAGE_NAME}/" 2>/dev/null || true
cp -r .trae "${PACKAGE_NAME}/.trae" 2>/dev/null || true

find "${PACKAGE_NAME}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${PACKAGE_NAME}" -type d -name ".git" -exec rm -rf {} + 2>/dev/null || true
find "${PACKAGE_NAME}" -name "*.pyc" -delete 2>/dev/null || true
find "${PACKAGE_NAME}" -name "*.zip" -delete 2>/dev/null || true
find "${PACKAGE_NAME}" -name "*.onnx" -delete 2>/dev/null || true

echo "[4/4] Creating archive..."

tar -czf "releases/${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}"

ARCHIVE_SIZE=$(du -h "releases/${PACKAGE_NAME}.tar.gz" | cut -f1)
SOURCE_SIZE=$(du -sh "${PACKAGE_NAME}" | cut -f1)

FILE_COUNT=$(find "${PACKAGE_NAME}" -type f | wc -l)
DIR_COUNT=$(find "${PACKAGE_NAME}" -type d | wc -l)

rm -rf "${PACKAGE_NAME}"

echo ""
echo "========================================="
echo "  ✅ Packaging completed!"
echo "========================================="
echo "  Package:    ${PACKAGE_NAME}.tar.gz"
echo "  Version:    ${VERSION}"
echo "  Files:      ${FILE_COUNT} files in ${DIR_COUNT} directories"
echo "  Source:     ${SOURCE_SIZE}"
echo "  Archive:    ${ARCHIVE_SIZE}"
echo "  Location:   releases/"
echo ""
echo "  Contents:"
echo "    ├─ Core:         train.py, evaluate.py, version.py, etc."
echo "    ├─ Environment:  env/ (PyBullet simulation)"
echo "    ├─ Training:     training/ (PPO training scripts)"
echo "    ├─ API Service:  api/ (FastAPI REST + WebSocket)"
echo "    ├─ Export:       export/ (ONNX export tools)"
echo "    ├─ Tests:        test_onnx_export.py, test_websocket.py"
echo "    └─ Docs:         README.md, USAGE.md, CHANGELOG.md"
echo ""
echo "  Quick Start:"
echo "    cd ${PACKAGE_NAME}"
echo "    source env/bin/activate"
echo "    python -m api.main          # Start API server"
echo "    python train.py             # Train model"
echo "    python export_onnx.py ...   # Export to ONNX"
echo "========================================="
echo ""

ls -lh releases/
