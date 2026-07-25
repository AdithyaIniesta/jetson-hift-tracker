#pragma once
#include <memory>
#include <string>
#include "FeatureExtractor.h"

namespace cr {
namespace vtracker {

/// Configuration for the TensorRT-backed feature extractor (Jetson).
struct TrtExtractorConfig
{
    /// Path to the ONNX embedding model. The model must have a static
    /// input shape 1 x channels x inputSize x inputSize and produce a
    /// single output tensor (the embedding, any length).
    std::string onnxPath;

    /// Serialized engine cache path. Empty = onnxPath + ".engine".
    /// The first run parses the ONNX and builds the engine (can take
    /// minutes on Jetson); subsequent runs deserialize the cache instantly.
    /// Engine files are specific to the GPU and TensorRT version - delete
    /// the cache after a JetPack upgrade.
    std::string engineCachePath;

    /// Model input height = width, pixels.
    int inputSize{128};

    /// Model input channels: 1 = grayscale, 3 = grayscale replicated to
    /// three channels (for ImageNet-pretrained backbones).
    int channels{3};

    /// Per-channel normalization applied to [0..1]-scaled pixels:
    /// out = (pixel/255 - mean) / std. Defaults are ImageNet statistics.
    float mean[3]{0.485f, 0.456f, 0.406f};
    float std[3]{0.229f, 0.224f, 0.225f};

    /// Build the engine with FP16 when the platform supports it
    /// (recommended on all Jetson modules).
    bool fp16{true};

    /// Builder workspace limit, MB.
    int workspaceMb{256};
};

/// Create a TensorRT-backed FeatureExtractor.
///
/// Returns nullptr when:
/// - the library was built without -DCVTRACKER_WITH_TENSORRT=ON, or
/// - the ONNX file is missing / cannot be parsed, or
/// - engine creation fails (diagnostics are printed to stderr).
///
/// Usage:
///   TrtExtractorConfig cfg;
///   cfg.onnxPath = "/home/user/models/embedder.onnx";
///   auto fe = createTrtFeatureExtractor(cfg);
///   if (fe) tracker.setFeatureExtractor(fe);
std::shared_ptr<FeatureExtractor>
createTrtFeatureExtractor(const TrtExtractorConfig& config);

} // namespace vtracker
} // namespace cr
