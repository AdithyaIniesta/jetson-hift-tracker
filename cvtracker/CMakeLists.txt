cmake_minimum_required(VERSION 3.14)
project(cvtracker VERSION 1.0.0 LANGUAGES CXX)

option(CVTRACKER_WITH_CUDA "Build with CUDA (cuFFT) acceleration" OFF)
option(CVTRACKER_WITH_TENSORRT "Build the TensorRT DNN feature extractor (Jetson)" OFF)
option(CVTRACKER_BUILD_TESTS "Build tests" OFF)
option(CVTRACKER_BUILD_EXAMPLES "Build examples (requires OpenCV for capture/display)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CVTRACKER_SOURCES
    src/VTracker.cpp
    src/Fft.cpp
    src/CorrelationCore.cpp
    src/Ekf.cpp
    src/TemplateBank.cpp
    src/trt/TrtFeatureExtractor.cpp
    src/CvTracker.cpp)

if(CVTRACKER_WITH_CUDA)
    enable_language(CUDA)
    list(APPEND CVTRACKER_SOURCES src/cuda/CudaFft.cu)
    # Old CMake (3.16 on JetPack / Ubuntu 20.04) has no CUDA_ARCHITECTURES:
    # pass the arch via flags instead. Override CMAKE_CUDA_FLAGS if needed.
    if(CMAKE_VERSION VERSION_LESS 3.18 AND NOT CMAKE_CUDA_FLAGS MATCHES "arch=")
        string(APPEND CMAKE_CUDA_FLAGS
            " -gencode arch=compute_72,code=sm_72")
    endif()
endif()

add_library(cvtracker ${CVTRACKER_SOURCES})
add_library(cvtracker::cvtracker ALIAS cvtracker)

target_include_directories(cvtracker
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>)

find_package(Threads REQUIRED)
target_link_libraries(cvtracker PRIVATE Threads::Threads)

if(CVTRACKER_WITH_CUDA)
    target_compile_definitions(cvtracker PRIVATE CVTRACKER_WITH_CUDA)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18
       AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Cover the whole Jetson series:
        # 5.3 - Nano/TX1, 6.2 - TX2, 7.2 - Xavier, 8.7 - Orin.
        set_target_properties(cvtracker PROPERTIES
            CUDA_ARCHITECTURES "53;62;72;87")
    endif()
    # Link cudart/cufft by absolute path: works with any CMake version and
    # bakes cleanly into the exported cvtrackerConfig.cmake.
    find_library(CVTRACKER_CUDART_LIB cudart
        HINTS /usr/local/cuda/lib64 ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CVTRACKER_CUFFT_LIB cufft
        HINTS /usr/local/cuda/lib64 ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    if(NOT CVTRACKER_CUDART_LIB OR NOT CVTRACKER_CUFFT_LIB)
        message(FATAL_ERROR "cudart / cufft not found - check CUDA install")
    endif()
    target_link_libraries(cvtracker PUBLIC
        ${CVTRACKER_CUDART_LIB} ${CVTRACKER_CUFFT_LIB})
endif()

if(CVTRACKER_WITH_TENSORRT)
    target_compile_definitions(cvtracker PRIVATE CVTRACKER_WITH_TENSORRT)
    # TensorRT + CUDA runtime. Standard locations on Jetson (JetPack) and
    # x86 TensorRT installs; adjust the HINTS for non-standard setups.
    find_path(CVTRACKER_TRT_INCLUDE NvInfer.h
        HINTS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu
              /usr/include)
    find_path(CVTRACKER_CUDA_INCLUDE cuda_runtime_api.h
        HINTS /usr/local/cuda/include)
    find_library(CVTRACKER_NVINFER_LIB nvinfer
        HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
    find_library(CVTRACKER_NVONNXPARSER_LIB nvonnxparser
        HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
    find_library(CVTRACKER_TRT_CUDART_LIB cudart
        HINTS /usr/local/cuda/lib64)
    if(NOT CVTRACKER_TRT_INCLUDE OR NOT CVTRACKER_CUDA_INCLUDE
       OR NOT CVTRACKER_NVINFER_LIB OR NOT CVTRACKER_NVONNXPARSER_LIB
       OR NOT CVTRACKER_TRT_CUDART_LIB)
        message(FATAL_ERROR
            "TensorRT / CUDA runtime not found - install TensorRT "
            "(part of JetPack on Jetson) or disable CVTRACKER_WITH_TENSORRT")
    endif()
    target_include_directories(cvtracker PRIVATE
        ${CVTRACKER_TRT_INCLUDE} ${CVTRACKER_CUDA_INCLUDE})
    target_link_libraries(cvtracker PUBLIC
        ${CVTRACKER_NVINFER_LIB} ${CVTRACKER_NVONNXPARSER_LIB}
        ${CVTRACKER_TRT_CUDART_LIB})
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cvtracker PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>)
endif()

# Install rules.
include(GNUInstallDirs)
install(TARGETS cvtracker
    EXPORT cvtrackerTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/cvtracker DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT cvtrackerTargets
    FILE cvtrackerTargets.cmake
    NAMESPACE cvtracker::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cvtracker)
# Config file resolves the Threads dependency for consumers.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cvtrackerConfig.cmake
"include(CMakeFindDependencyMacro)\nfind_dependency(Threads)\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/cvtrackerTargets.cmake\")\n")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cvtrackerConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cvtracker)

if(CVTRACKER_BUILD_EXAMPLES)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui)
    add_executable(camera_demo examples/camera_demo.cpp)
    target_link_libraries(camera_demo PRIVATE cvtracker ${OpenCV_LIBS}
        Threads::Threads)
    target_include_directories(camera_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

if(CVTRACKER_BUILD_TESTS)
    enable_testing()
    add_executable(test_tracker test/test_tracker.cpp)
    target_link_libraries(test_tracker PRIVATE cvtracker Threads::Threads)
    add_test(NAME test_tracker COMMAND test_tracker)
endif()
