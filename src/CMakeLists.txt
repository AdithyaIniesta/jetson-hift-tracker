cmake_minimum_required(VERSION 3.13)

# ─────────────────────────────────────────────────────────────────────────────
# Tracker version setup (Passed from build script / build system)
# ─────────────────────────────────────────────────────────────────────────────
set(TRACKER_VERSION "cuda_library" CACHE STRING "Tracker algorithm version")
set(JETSON_BOARD "orin" CACHE STRING "Jetson board: orin or xavier")
set(EXE_NAME "JetsonTracker_${TRACKER_VERSION}_${JETSON_BOARD}")

project(${EXE_NAME} LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Collect Sources ──────────────────────────────────────────
file(GLOB_RECURSE SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.S
    ${CMAKE_CURRENT_SOURCE_DIR}/../filters/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../filters/*.h
)

# The lockon engine ships its own cr::vtracker::Ekf and uses it internally,
# so its libcvtracker.a pulls in that Ekf.cpp. The pipeline's
# filters/ekf/Ekf.cpp defines the same fully-qualified symbol, causing a
# "multiple definition" link error. The lockon build doesn't use the
# pipeline's external EKF (guarded out in tracker.cpp), so drop that TU
# from this build only. Baseline/v1 keep it.
if(TRACKER_ENGINE_DIR STREQUAL "cvtracker-lockon")
    list(FILTER SOURCES EXCLUDE REGEX "filters/ekf/Ekf\\.cpp$")
endif()

add_executable(${EXE_NAME} ${SOURCES})
# Force the output directory for this specific target
set_target_properties(${EXE_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ─────────────────────────────────────────────────────────────────────────────
# Build reproducibility — inject git commit + build time + engine name so
# meta.json in every recording captures exactly which binary produced it.
# Values are consumed via JT_GIT_COMMIT / JT_BUILD_TIME / JT_TRACKER_ENGINE
# in src/main.cpp. Missing git falls back to "unknown".
# ─────────────────────────────────────────────────────────────────────────────
execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE JT_GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT JT_GIT_COMMIT_HASH)
    set(JT_GIT_COMMIT_HASH "unknown")
endif()
string(TIMESTAMP JT_BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%SZ" UTC)
target_compile_definitions(${EXE_NAME} PRIVATE
    JT_GIT_COMMIT="${JT_GIT_COMMIT_HASH}"
    JT_BUILD_TIME="${JT_BUILD_TIMESTAMP}"
    JT_TRACKER_ENGINE="${TRACKER_ENGINE_DIR}"
)
# ─────────────────────────────────────────────────────────────────────────────
# Tracker Integration (Native CUDA v2 vs Legacy Constant Robotics)
# ─────────────────────────────────────────────────────────────────────────────
option(USE_NATIVE_CUDA_TRACKER "ON for Native CUDA Tracker, OFF for Constant Robotics" ON)

if(USE_NATIVE_CUDA_TRACKER)
    message(STATUS "-> Building with NATIVE CUDA TRACKER (v2 Engine)")
    # We add the subdirectory here. If the root also adds it, use the logic below:
    if(NOT TARGET cvtracker)
        add_subdirectory(${CMAKE_SOURCE_DIR}/${TRACKER_ENGINE_DIR} ${CMAKE_BINARY_DIR}/cvtracker_submodule)
    endif()

    target_link_libraries(${EXE_NAME} PUBLIC cvtracker::cvtracker)
    target_compile_definitions(${EXE_NAME} PUBLIC TRACKER_V2)
    # Lockon engine extends VTrackerParam with ENABLE_GMC (24),
    # ENABLE_DNN_ARBITRATION (25), ENABLE_DNN_REACQUISITION (26) —
    # widen the app's param range only for that build.
    if(TRACKER_ENGINE_DIR STREQUAL "cvtracker-lockon")
        target_compile_definitions(${EXE_NAME} PUBLIC TRACKER_LOCKON)
    endif()
else()
    message(STATUS "-> Building with legacy CONSTANT ROBOTICS tracker")
    # Legacy fallback: point to the static library path
    target_include_directories(${EXE_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../constant_robotics/include")
    target_link_libraries(${EXE_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../constant_robotics/lib/libCvTracker.a")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Includes and Dependencies
# ─────────────────────────────────────────────────────────────────────────────
target_include_directories(${EXE_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../filters"
    "${CMAKE_SOURCE_DIR}/${TRACKER_ENGINE_DIR}/src"   # <--- THIS IS THE KEY
    "${CMAKE_SOURCE_DIR}/${TRACKER_ENGINE_DIR}/include"
)

find_package(OpenMP)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)

# FIX: Add these two lines to ensure GST headers are passed to the compiler
target_include_directories(${EXE_NAME} PRIVATE ${GST_INCLUDE_DIRS})
target_compile_options(${EXE_NAME} PRIVATE ${GST_CFLAGS_OTHER})

target_include_directories(${EXE_NAME} PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../filters"
)

target_link_libraries(${EXE_NAME} PRIVATE 
    ${OpenCV_LIBS} 
    ${GST_LIBRARIES} 
    pthread
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${EXE_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()