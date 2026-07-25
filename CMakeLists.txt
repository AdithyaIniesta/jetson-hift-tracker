cmake_minimum_required(VERSION 3.14)
project(JetsonTracker LANGUAGES CXX)

# 1. Turn on the options needed for your engine build (adjust switches as needed)
set(CVTRACKER_WITH_CUDA ON CACHE BOOL "" FORCE)
set(CVTRACKER_WITH_TENSORRT ON CACHE BOOL "" FORCE)

# 2. Add the tracker engine subdirectory so its target exists.
# TRACKER_ENGINE_DIR selects which vendored engine builds the
# `cvtracker` target: "cvtracker" (baseline) or "cvtracker-lockon".
set(TRACKER_ENGINE_DIR "cvtracker" CACHE STRING "Tracker engine source dir: cvtracker or cvtracker-lockon")
add_subdirectory(${TRACKER_ENGINE_DIR})

# 3. Add your main application source directory
add_subdirectory(src)