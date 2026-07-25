#!/usr/bin/env bash
# =============================================================================
# build.sh — JetsonTracker JetPack 5 build helper
# Usage:
#   ./build.sh base                    Build base image (~90 min)
#   ./build.sh dev                     Build dev image
#   ./build.sh compile <project_path>  Compile both tracker binaries
#   ./build.sh runtime <project_path>  Assemble runtime image
#   ./build.sh save                    Save runtime image
#   ./build.sh load                    Load runtime image
# =============================================================================

set -euo pipefail

OPENCV_VERSION="4.5.4"
CUDA_ARCH="7.2;8.7"
MAKE_JOBS=2
L4T_BASE="nvcr.io/nvidia/l4t-jetpack:r35.4.1"

BASE_IMAGE="jvp/base:${OPENCV_VERSION}-jp5-orin-xavier"
DEV_IMAGE="jvp/dev:jp5-orin-xavier"
RUNTIME_IMAGE="jvp/jetsontracker:v2.0-jp5-orin-xavier"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

log()  { echo ""; echo "▶ $*"; }
ok()   { echo "✔ $*"; }
fail() { echo "✘ $*" >&2; exit 1; }

check_docker() {
    docker info &>/dev/null || fail "Docker not running or permission denied."
}

build_base() {
    log "Building base image (OpenCV ${OPENCV_VERSION} — ~90 min)"
    docker build \
        --target base \
        --build-arg L4T_BASE="${L4T_BASE}" \
        --build-arg OPENCV_VERSION="${OPENCV_VERSION}" \
        --build-arg OPENCV_CUDA_ARCH="${CUDA_ARCH}" \
        --build-arg MAKE_JOBS="${MAKE_JOBS}" \
        -t "${BASE_IMAGE}" \
        -f "${DOCKERFILE}" \
        "${SCRIPT_DIR}"
    ok "Base image: ${BASE_IMAGE}"
}

build_dev() {
    log "Building dev image"
    docker build \
        --target dev \
        --build-arg L4T_BASE="${L4T_BASE}" \
        --build-arg OPENCV_VERSION="${OPENCV_VERSION}" \
        --build-arg OPENCV_CUDA_ARCH="${CUDA_ARCH}" \
        --build-arg MAKE_JOBS="${MAKE_JOBS}" \
        -t "${DEV_IMAGE}" \
        -f "${DOCKERFILE}" \
        "${SCRIPT_DIR}"
    ok "Dev image: ${DEV_IMAGE}"
}

compile() {
    PROJECT_DIR="${1:-}"
    [ -n "${PROJECT_DIR}" ] || fail "Usage: ./build.sh compile <path_to_project>"
    [ -d "${PROJECT_DIR}" ] || fail "Project not found: ${PROJECT_DIR}"
    [ -f "${PROJECT_DIR}/CMakeLists.txt" ] || fail "CMakeLists.txt not found"

    log "Compiling both tracker binaries from ${PROJECT_DIR}"
    docker run --rm \
        --runtime nvidia \
        -v "${PROJECT_DIR}":/workspace \
        -w /workspace \
        "${DEV_IMAGE}" \
        bash -c "
            set -e
            export PATH=/usr/local/cuda/bin:\$PATH
            export CUDACXX=/usr/local/cuda/bin/nvcc

            echo '--- Building constant_robotics_lib ---'
            rm -rf build/CMakeFiles build/CMakeCache.txt build/cvtracker build/src
            cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
                -DTRACKER_VERSION:STRING=constant_robotics_lib \
                -DCMAKE_CXX_FLAGS='-Wno-error -fpermissive'
            ninja -C build -j${MAKE_JOBS}

            echo '--- Building cuda_library ---'
            rm -rf build/CMakeFiles build/CMakeCache.txt build/cvtracker build/src
            cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
                -DTRACKER_VERSION:STRING=cuda_library \
                -DCMAKE_CXX_FLAGS='-Wno-error -fpermissive'
            ninja -C build -j${MAKE_JOBS}

            echo '[OK] Both binaries compiled'
        "
    ok "Compile done — binaries at ${PROJECT_DIR}/build/bin/"
    ls "${PROJECT_DIR}/build/bin/"
}

build_runtime() {
    PROJECT_DIR="${1:-}"
    [ -n "${PROJECT_DIR}" ] || fail "Usage: ./build.sh runtime <path_to_project>"
    [ -f "${PROJECT_DIR}/build/bin/JetsonTracker_constant_robotics_lib" ] || \
        fail "JetsonTracker_constant_robotics_lib not found — run compile first"
    [ -f "${PROJECT_DIR}/build/bin/JetsonTracker_cuda_library" ] || \
        fail "JetsonTracker_cuda_library not found — run compile first"
    [ -f "${PROJECT_DIR}/run.sh" ] || \
        fail "run.sh not found in project directory"

    log "Assembling runtime image"
    docker build \
        --target runtime \
        --build-arg L4T_BASE="${L4T_BASE}" \
        --build-arg OPENCV_VERSION="${OPENCV_VERSION}" \
        --build-arg OPENCV_CUDA_ARCH="${CUDA_ARCH}" \
        --build-arg MAKE_JOBS="${MAKE_JOBS}" \
        -t "${RUNTIME_IMAGE}" \
        -f "${DOCKERFILE}" \
        "${SCRIPT_DIR}"

    CID=$(docker create "${RUNTIME_IMAGE}")
    trap "docker rm -f ${CID} 2>/dev/null || true" EXIT

    docker cp "${PROJECT_DIR}/build/bin/JetsonTracker_constant_robotics_lib" \
              "${CID}":/opt/jvp/bin/
    docker cp "${PROJECT_DIR}/build/bin/JetsonTracker_cuda_library" \
              "${CID}":/opt/jvp/bin/
    chmod +x "${PROJECT_DIR}/run_docker.sh"
    docker cp "${PROJECT_DIR}/run_docker.sh" \
              "${CID}":/opt/jvp/run_docker.sh

    docker commit \
        --message "JetsonTracker v2 jp5 runtime $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${CID}" "${RUNTIME_IMAGE}"

    trap - EXIT
    docker rm -f "${CID}" 2>/dev/null || true
    ok "Runtime image: ${RUNTIME_IMAGE}"
}

save_image() {
    log "Saving runtime image"
    docker save "${RUNTIME_IMAGE}" | gzip > "${SCRIPT_DIR}/jetsontracker-v2-jp5.tar.gz"
    ok "Saved: jetsontracker-v2-jp5.tar.gz"
}

load_image() {
    [ -f "${SCRIPT_DIR}/jetsontracker-v2-jp5.tar.gz" ] || fail "jetsontracker-v2-jp5.tar.gz not found"
    docker load < "${SCRIPT_DIR}/jetsontracker-v2-jp5.tar.gz"
    ok "Loaded: ${RUNTIME_IMAGE}"
}

check_docker

case "${1:-}" in
    base)     build_base ;;
    dev)      build_dev ;;
    compile)  compile "${2:-}" ;;
    runtime)  build_runtime "${2:-}" ;;
    save)     save_image ;;
    load)     load_image ;;
    *)
        echo "Usage: $0 {base|dev|compile <path>|runtime <path>|save|load}"
        exit 1
        ;;
esac