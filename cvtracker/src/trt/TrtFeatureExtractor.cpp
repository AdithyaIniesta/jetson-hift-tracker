#include "cvtracker/TrtFeatureExtractor.h"
#include <cstdio>

#ifdef CVTRACKER_WITH_TENSORRT

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <mutex>
#include <vector>

namespace cr {
namespace vtracker {

namespace {

class TrtLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::fprintf(stderr, "[cvtracker/trt] %s\n", msg);
    }
};

TrtLogger& trtLogger()
{
    // Intentionally leaked: TensorRT stores this reference inside every
    // IRuntime/ICudaEngine/IExecutionContext it creates and may log from
    // their destructors. Those objects are typically owned transitively by
    // a global (e.g. the tracker singleton), which is destroyed after
    // main() returns. A function-local static here would be destroyed
    // before that global (reverse construction order), leaving TensorRT
    // holding a dangling reference at process-exit teardown time.
    static TrtLogger* logger = new TrtLogger();
    return *logger;
}

// TRT >= 8 supports plain delete on interface objects.
template <typename T>
struct TrtDelete
{
    void operator()(T* p) const { delete p; }
};
template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDelete<T>>;

} // namespace

/// TensorRT-backed embedding extractor. Loads (or builds and caches) an
/// engine from an ONNX model with static input 1 x C x S x S and a single
/// output tensor. Inference is synchronous; extract() is called under the
/// tracker's lock, so no extra synchronization is needed beyond m_mutex
/// (which guards direct multi-tracker sharing of one extractor).
class TrtFeatureExtractor : public FeatureExtractor
{
public:
    ~TrtFeatureExtractor() override
    {
        if (m_dIn)
            cudaFree(m_dIn);
        if (m_dOut)
            cudaFree(m_dOut);
        if (m_stream)
            cudaStreamDestroy(m_stream);
    }

    bool init(const TrtExtractorConfig& cfg)
    {
        m_cfg = cfg;
        if (m_cfg.channels != 1 && m_cfg.channels != 3)
        {
            std::fprintf(stderr, "[cvtracker/trt] channels must be 1 or 3\n");
            return false;
        }
        if (!buildOrLoadEngine())
            return false;

        m_context.reset(m_engine->createExecutionContext());
        if (!m_context)
            return false;
        if (!resolveIo())
            return false;

        m_inCount = (size_t)m_cfg.channels * m_cfg.inputSize * m_cfg.inputSize;
        m_hostIn.resize(m_inCount);
        m_hostOut.resize(m_outCount);
        if (cudaMalloc(&m_dIn, m_inCount * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&m_dOut, m_outCount * sizeof(float)) != cudaSuccess ||
            cudaStreamCreate(&m_stream) != cudaSuccess)
        {
            std::fprintf(stderr, "[cvtracker/trt] CUDA allocation failed\n");
            return false;
        }
        std::fprintf(stderr,
                     "[cvtracker/trt] engine ready: input %dx%dx%d, "
                     "embedding dim %zu\n",
                     m_cfg.channels, m_cfg.inputSize, m_cfg.inputSize,
                     m_outCount);
        return true;
    }

    int inputSize() const override { return m_cfg.inputSize; }

    bool extract(const float* patch, std::vector<float>& embedding) override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        // Normalize [0..255] gray patch into 1 x C x S x S host buffer,
        // replicating the gray channel when the model expects 3 channels.
        const int n = m_cfg.inputSize * m_cfg.inputSize;
        for (int c = 0; c < m_cfg.channels; ++c)
        {
            const float mean = m_cfg.mean[c];
            const float inv = 1.0f / m_cfg.std[c];
            float* dst = m_hostIn.data() + (size_t)c * n;
            for (int i = 0; i < n; ++i)
                dst[i] = (patch[i] * (1.0f / 255.0f) - mean) * inv;
        }

        if (cudaMemcpyAsync(m_dIn, m_hostIn.data(),
                            m_inCount * sizeof(float),
                            cudaMemcpyHostToDevice, m_stream) != cudaSuccess)
            return false;

#if NV_TENSORRT_MAJOR >= 10
        if (!m_context->setTensorAddress(m_inName.c_str(), m_dIn) ||
            !m_context->setTensorAddress(m_outName.c_str(), m_dOut))
            return false;
        if (!m_context->enqueueV3(m_stream))
            return false;
#else
        void* bindings[16] = {nullptr};
        bindings[m_inIndex] = m_dIn;
        bindings[m_outIndex] = m_dOut;
        if (!m_context->enqueueV2(bindings, m_stream, nullptr))
            return false;
#endif
        if (cudaMemcpyAsync(m_hostOut.data(), m_dOut,
                            m_outCount * sizeof(float),
                            cudaMemcpyDeviceToHost, m_stream) != cudaSuccess)
            return false;
        if (cudaStreamSynchronize(m_stream) != cudaSuccess)
            return false;

        embedding.assign(m_hostOut.begin(), m_hostOut.end());
        return true;
    }

private:
    bool buildOrLoadEngine()
    {
        const std::string cachePath = m_cfg.engineCachePath.empty()
                                          ? m_cfg.onnxPath + ".engine"
                                          : m_cfg.engineCachePath;
        m_runtime.reset(nvinfer1::createInferRuntime(trtLogger()));
        if (!m_runtime)
            return false;

        // Fast path: deserialize cached engine.
        {
            std::ifstream in(cachePath, std::ios::binary);
            if (in)
            {
                std::vector<char> blob((std::istreambuf_iterator<char>(in)),
                                       std::istreambuf_iterator<char>());
                if (!blob.empty())
                    m_engine.reset(m_runtime->deserializeCudaEngine(
                        blob.data(), blob.size()));
                if (m_engine)
                {
                    std::fprintf(stderr,
                                 "[cvtracker/trt] loaded engine cache: %s\n",
                                 cachePath.c_str());
                    return true;
                }
                std::fprintf(stderr,
                             "[cvtracker/trt] stale engine cache ignored "
                             "(GPU/TRT version changed?): %s\n",
                             cachePath.c_str());
            }
        }

        // Slow path: parse ONNX and build the engine.
        TrtPtr<nvinfer1::IBuilder> builder(
            nvinfer1::createInferBuilder(trtLogger()));
        if (!builder)
            return false;
#if NV_TENSORRT_MAJOR >= 10
        TrtPtr<nvinfer1::INetworkDefinition> network(
            builder->createNetworkV2(0));
#else
        const auto flags =
            1U << (uint32_t)
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH;
        TrtPtr<nvinfer1::INetworkDefinition> network(
            builder->createNetworkV2(flags));
#endif
        if (!network)
            return false;
        TrtPtr<nvonnxparser::IParser> parser(
            nvonnxparser::createParser(*network, trtLogger()));
        if (!parser ||
            !parser->parseFromFile(
                m_cfg.onnxPath.c_str(),
                (int)nvinfer1::ILogger::Severity::kWARNING))
        {
            std::fprintf(stderr, "[cvtracker/trt] failed to parse ONNX: %s\n",
                         m_cfg.onnxPath.c_str());
            return false;
        }

        TrtPtr<nvinfer1::IBuilderConfig> config(
            builder->createBuilderConfig());
        if (!config)
            return false;
#if NV_TENSORRT_MAJOR > 8 ||                                                   \
    (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                   (size_t)m_cfg.workspaceMb << 20);
#else
        config->setMaxWorkspaceSize((size_t)m_cfg.workspaceMb << 20);
#endif
        if (m_cfg.fp16 && builder->platformHasFastFp16())
            config->setFlag(nvinfer1::BuilderFlag::kFP16);

        std::fprintf(stderr,
                     "[cvtracker/trt] building engine from %s (this can "
                     "take minutes on Jetson)...\n",
                     m_cfg.onnxPath.c_str());
        TrtPtr<nvinfer1::IHostMemory> plan(
            builder->buildSerializedNetwork(*network, *config));
        if (!plan)
        {
            std::fprintf(stderr, "[cvtracker/trt] engine build failed\n");
            return false;
        }
        m_engine.reset(
            m_runtime->deserializeCudaEngine(plan->data(), plan->size()));
        if (!m_engine)
            return false;

        std::ofstream out(cachePath, std::ios::binary);
        if (out)
        {
            out.write((const char*)plan->data(), (std::streamsize)plan->size());
            std::fprintf(stderr, "[cvtracker/trt] engine cached: %s\n",
                         cachePath.c_str());
        }
        return true;
    }

    /// Locate the input / output tensors and the output element count.
    bool resolveIo()
    {
#if NV_TENSORRT_MAJOR >= 10
        const int n = m_engine->getNbIOTensors();
        for (int i = 0; i < n; ++i)
        {
            const char* name = m_engine->getIOTensorName(i);
            if (m_engine->getTensorIOMode(name) ==
                nvinfer1::TensorIOMode::kINPUT)
                m_inName = name;
            else if (m_outName.empty())
                m_outName = name;
        }
        if (m_inName.empty() || m_outName.empty())
            return false;
        const nvinfer1::Dims d = m_engine->getTensorShape(m_outName.c_str());
#else
        const int n = m_engine->getNbBindings();
        m_inIndex = -1;
        m_outIndex = -1;
        for (int i = 0; i < n && i < 16; ++i)
        {
            if (m_engine->bindingIsInput(i))
                m_inIndex = i;
            else if (m_outIndex < 0)
                m_outIndex = i;
        }
        if (m_inIndex < 0 || m_outIndex < 0)
            return false;
        const nvinfer1::Dims d = m_engine->getBindingDimensions(m_outIndex);
#endif
        size_t count = 1;
        for (int i = 0; i < d.nbDims; ++i)
        {
            if (d.d[i] < 0)
            {
                std::fprintf(stderr,
                             "[cvtracker/trt] dynamic shapes are not "
                             "supported - export the ONNX with a static "
                             "input shape\n");
                return false;
            }
            count *= (size_t)d.d[i];
        }
        if (count == 0 || count > (1u << 20))
            return false;
        m_outCount = count;
        return true;
    }

    TrtExtractorConfig m_cfg;
    std::mutex m_mutex;
    TrtPtr<nvinfer1::IRuntime> m_runtime;
    TrtPtr<nvinfer1::ICudaEngine> m_engine;
    TrtPtr<nvinfer1::IExecutionContext> m_context;
#if NV_TENSORRT_MAJOR >= 10
    std::string m_inName;
    std::string m_outName;
#else
    int m_inIndex{-1};
    int m_outIndex{-1};
#endif
    size_t m_inCount{0};
    size_t m_outCount{0};
    std::vector<float> m_hostIn;
    std::vector<float> m_hostOut;
    void* m_dIn{nullptr};
    void* m_dOut{nullptr};
    cudaStream_t m_stream{nullptr};
};

std::shared_ptr<FeatureExtractor>
createTrtFeatureExtractor(const TrtExtractorConfig& config)
{
    auto extractor = std::make_shared<TrtFeatureExtractor>();
    if (!extractor->init(config))
        return nullptr;
    return extractor;
}

} // namespace vtracker
} // namespace cr

#else // !CVTRACKER_WITH_TENSORRT

namespace cr {
namespace vtracker {

std::shared_ptr<FeatureExtractor>
createTrtFeatureExtractor(const TrtExtractorConfig&)
{
    std::fprintf(stderr,
                 "[cvtracker] TensorRT support not compiled in - rebuild "
                 "with -DCVTRACKER_WITH_TENSORRT=ON\n");
    return nullptr;
}

} // namespace vtracker
} // namespace cr

#endif // CVTRACKER_WITH_TENSORRT
