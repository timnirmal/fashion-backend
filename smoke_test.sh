#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-fashion-recsys:latest}
CONTAINER_NAME=${CONTAINER_NAME:-fashion-recsys-smoke}
PORT=${PORT:-8000}
ORIGINS=${ALLOW_ORIGINS:-http://localhost:3000}

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "[1/5] Building image ${IMAGE}..."
docker build -t "${IMAGE}" "$ROOT_DIR"

echo "[2/5] Running container ${CONTAINER_NAME}..."
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run --rm -d --name "${CONTAINER_NAME}" -p ${PORT}:8000 -e ALLOW_ORIGINS="${ORIGINS}" "${IMAGE}"

echo "[3/5] Waiting for health..."
for i in {1..30}; do
  if curl -fsS "http://localhost:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS "http://localhost:${PORT}/health" | jq . || true

echo "[4/5] Training..."
curl -fsS -X POST "http://localhost:${PORT}/train" | jq . || true

# Extract an existing user id from the dataset
EXISTING_USER=$(awk -F, 'NR==2{print $2}' "$ROOT_DIR/generated_data/product_interactions.csv")
NEW_USER=ffffffff-ffff-ffff-ffff-ffffffffffff

echo "[5/5] Recommend for existing user: ${EXISTING_USER}"
curl -fsS -X POST "http://localhost:${PORT}/recommend" \
  -H 'Content-Type: application/json' \
  -d "{\"user_id\":\"${EXISTING_USER}\",\"method\":\"blend_bpr_tfidf\",\"top_k\":10}" | jq . || true

echo "Recommend for new user (cold-start): ${NEW_USER}"
curl -fsS -X POST "http://localhost:${PORT}/recommend" \
  -H 'Content-Type: application/json' \
  -d "{\"user_id\":\"${NEW_USER}\",\"method\":\"blend_bpr_tfidf\",\"top_k\":10}" | jq . || true

echo "Stopping container..."
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
echo "Smoke test completed."


