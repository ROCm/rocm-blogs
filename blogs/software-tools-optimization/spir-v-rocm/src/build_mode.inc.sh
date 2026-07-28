# build_mode.inc.sh — map BUILD_MODE to compiler flags (source from benchmark scripts).
#
# BUILD_MODE:
#   default     — compiler default (no explicit -O3, no --offload-compress)
#   O3          — release: -O3
#   O3compress  — release + --offload-compress (PyTorch-style shipping flags)
#
BUILD_MODE="${BUILD_MODE:-O3}"
OPT_FLAGS=""
COMPRESS_FLAG=""
case "$BUILD_MODE" in
    default)
        BUILD_MODE_LABEL="default"
        ;;
    O3)
        OPT_FLAGS="-O3"
        BUILD_MODE_LABEL="O3"
        ;;
    O3compress)
        OPT_FLAGS="-O3"
        COMPRESS_FLAG="--offload-compress"
        BUILD_MODE_LABEL="O3 + --offload-compress"
        ;;
    *)
        echo "Unknown BUILD_MODE='$BUILD_MODE' (use default, O3, or O3compress)" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
