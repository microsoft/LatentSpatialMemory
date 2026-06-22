#!/bin/bash
#
# Build sharded LMDB from Spatia data folders
#
# This script:
# 1. Packs .pt and .txt files into sharded LMDB
# 2. Caches LMDB keys for fast dataset initialization
#
# Configuration is read from data_process/_0_0_0_root_assign.py
# Only override DATA_ROOT if you need to process a different path.
#
# Usage:
#   ./build_lmdb.sh                                    # Use config from _0_0_0_root_assign.py
#   ./build_lmdb.sh data/Spatia/frame33_fps16_5000    # Override data root
#   DATA_ROOT=data/Spatia/xxx ./build_lmdb.sh         # Via environment variable
#
# Output: {DATA_ROOT}_lmdb/
#   e.g., data/Spatia/frame33_fps16_2000 -> data/Spatia/frame33_fps16_2000_lmdb
#

set -e

# === Get DATA_ROOT from config if not specified ===
if [ -n "$1" ]; then
    DATA_ROOT="$1"
elif [ -n "$DATA_ROOT" ]; then
    DATA_ROOT="$DATA_ROOT"
else
    # Read from _0_0_0_root_assign.py CONFIG.output_root
    DATA_ROOT=$(python -c "from data_process._0_0_0_root_assign import CONFIG; print(CONFIG.output_root)")
fi

TARGET_SHARD_SIZE_GB="${TARGET_SHARD_SIZE_GB:-10.0}"

# Derive LMDB output path
LMDB_ROOT="${DATA_ROOT}_lmdb"

echo "============================================================"
echo "Spatia LMDB Builder"
echo "============================================================"
echo "Data root:        $DATA_ROOT"
echo "LMDB output:      $LMDB_ROOT"
echo "Target shard GB:  $TARGET_SHARD_SIZE_GB"
echo "============================================================"
echo ""

# Check data root exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root not found: $DATA_ROOT"
    exit 1
fi

# Step 1: Pack to LMDB
echo ">>> Step 1: Packing data to sharded LMDB..."
echo ""
python -m data_process.pack_to_lmdb \
    --data-root "$DATA_ROOT" \
    --lmdb-root "$LMDB_ROOT" \
    --target-shard-size-gb "$TARGET_SHARD_SIZE_GB"

echo ""
echo "============================================================"

# Step 2: Cache keys
echo ">>> Step 2: Caching LMDB keys..."
echo ""
python -m data_process.cache_lmdb_keys \
    --lmdb-root "$LMDB_ROOT" \
    --force

echo ""
echo "============================================================"
echo "LMDB build complete!"
echo "Output: $LMDB_ROOT"
echo "============================================================"
