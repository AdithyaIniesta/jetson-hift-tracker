#include "hift/TrtHiFT.h"

#include <cstdio>

#ifdef CVTRACKER_WITH_TENSORRT

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <map>
#include <vector>

namespace cr {
namespace hift {

namespace {

class TrtLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::fprintf(stderr, "[hift/trt] %s\n", msg);
    }
};

TrtLogger& trtLogger()
{
    // Intentionally leaked (see TrtFeatureExtractor.cpp): TensorRT logs from
    // destructors of globally-owned objects that outlive a function-local static.
    static TrtLogger* logger = new TrtLogger();
    return *logger;
}

template <typename T>
struct TrtDelete
{
    void operator()(T* p) const { delete p; }
};
template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDelete<T>>;

size_t volume(const nvinfer1::Dims& d)
{
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i)
    {
        if (d.d[i] < 0)
            return 0;  // dynamic — unsupported
        v *= (size_t)d.d[i];
    }
    return v;
}

// One loaded engine + its execution context, with named-tensor bookkeeping.
struct Engine
{
    TrtPtr<nvinfer1::ICudaEngine> engine;
    TrtPtr<nvinfer1::IExecutionContext> context;

    bool build(nvinfer1::IRuntime* runtime, const std::string& onnxPath,
               const std::string& enginePath, bool fp16, int workspaceMb)
    {
        const std::string cachePath =
            enginePath.empty() ? onnxPath + ".engine" : enginePath;

        // Fast path: cached engine.
        {
            std::ifstream in(cachePath, std::ios::binary);
            if (in)
            {
                std::vector<char> blob((std::istreambuf_iterator<char>(in)),
                                       std::istreambuf_iterator<char>());
                if (!blob.empty())
                    engine.reset(runtime->deserializeCudaEngine(blob.data(),
                                                                blob.size()));
                if (engine)
                {
                    std::fprintf(stderr, "[hift/trt] loaded engine cache: %s\n",
                                 cachePath.c_str());
                    context.reset(engine->createExecutionContext());
                    return context != nullptr;
                }
                std::fprintf(stderr,
                             "[hift/trt] stale engine cache ignored: %s\n",
                             cachePath.c_str());
            }
        }

        // Slow path: parse ONNX, build, cache.
        TrtPtr<nvinfer1::IBuilder> builder(
            nvinfer1::createInferBuilder(trtLogger()));
        if (!builder)
            return false;
        const auto flags =
            1U << (uint32_t)
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH;
        TrtPtr<nvinfer1::INetworkDefinition> network(
            builder->createNetworkV2(flags));
        if (!network)
            return false;
        TrtPtr<nvonnxparser::IParser> parser(
            nvonnxparser::createParser(*network, trtLogger()));
        if (!parser ||
            !parser->parseFromFile(
                onnxPath.c_str(),
                (int)nvinfer1::ILogger::Severity::kWARNING))
        {
            std::fprintf(stderr, "[hift/trt] failed to parse ONNX: %s\n",
                         onnxPath.c_str());
            return false;
        }
        TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (!config)
            return false;
#if NV_TENSORRT_MAJOR > 8 ||                                                   \
    (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                   (size_t)workspaceMb << 20);
#else
        config->setMaxWorkspaceSize((size_t)workspaceMb << 20);
#endif
        if (fp16 && builder->platformHasFastFp16())
            config->setFlag(nvinfer1::BuilderFlag::kFP16);

        std::fprintf(stderr,
                     "[hift/trt] building engine from %s (minutes on Jetson)...\n",
                     onnxPath.c_str());
        TrtPtr<nvinfer1::IHostMemory> plan(
            builder->buildSerializedNetwork(*network, *config));
        if (!plan)
        {
            std::fprintf(stderr, "[hift/trt] engine build failed: %s\n",
                         onnxPath.c_str());
            return false;
        }
        engine.reset(
            runtime->deserializeCudaEngine(plan->data(), plan->size()));
        if (!engine)
            return false;
        std::ofstream out(cachePath, std::ios::binary);
        if (out)
        {
            out.write((const char*)plan->data(), (std::streamsize)plan->size());
            std::fprintf(stderr, "[hift/trt] engine cached: %s\n",
                         cachePath.c_str());
        }
        context.reset(engine->createExecutionContext());
        return context != nullptr;
    }
};

}  // namespace

struct TrtHiFT::Impl
{
    TrtPtr<nvinfer1::IRuntime> runtime;
    Engine tmpl;
    Engine trk;
    cudaStream_t stream{nullptr};

    // Device buffers.
    void* dZ{nullptr};                 // template input z
    void* dX{nullptr};                 // track input x
    std::vector<void*> dZf;            // shared: tmpl outputs == trk inputs
    std::vector<size_t> zfCount;       // element count per zf
    void* dLoc{nullptr};
    void* dCls1{nullptr};
    void* dCls2{nullptr};

    size_t zCount{0};
    size_t xCount{0};
    size_t locCount{0}, cls1Count{0}, cls2Count{0};
    bool hasTemplate{false};
    bool ok{false};

    ~Impl()
    {
        if (dZ) cudaFree(dZ);
        if (dX) cudaFree(dX);
        for (void* p : dZf)
            if (p) cudaFree(p);
        if (dLoc) cudaFree(dLoc);
        if (dCls1) cudaFree(dCls1);
        if (dCls2) cudaFree(dCls2);
        if (stream) cudaStreamDestroy(stream);
    }

    // Binding index for a named tensor in an engine (TRT < 10).
    static int bindIndex(nvinfer1::ICudaEngine* e, const char* name)
    {
        return e->getBindingIndex(name);
    }
};

TrtHiFT::TrtHiFT() : d_(new Impl) {}
TrtHiFT::~TrtHiFT() = default;

bool TrtHiFT::ready() const { return d_->ok; }

bool TrtHiFT::initialise(const TrtHiFTConfig& cfg)
{
    d_->runtime.reset(nvinfer1::createInferRuntime(trtLogger()));
    if (!d_->runtime)
        return false;
    if (!d_->tmpl.build(d_->runtime.get(), cfg.templateOnnx, cfg.templateEngine,
                        cfg.fp16, cfg.workspaceMb))
        return false;
    if (!d_->trk.build(d_->runtime.get(), cfg.trackOnnx, cfg.trackEngine,
                       cfg.fp16, cfg.workspaceMb))
        return false;
    if (cudaStreamCreate(&d_->stream) != cudaSuccess)
        return false;

    auto* te = d_->tmpl.engine.get();
    auto* ke = d_->trk.engine.get();

    // z input.
    const int zIdx = Impl::bindIndex(te, "z");
    if (zIdx < 0)
    {
        std::fprintf(stderr, "[hift/trt] template engine has no input 'z'\n");
        return false;
    }
    d_->zCount = volume(te->getBindingDimensions(zIdx));

    // zf tensors: template outputs, matched by name into track inputs. The
    // exporter names them zf0, zf1, ... in order; we allocate ONE device buffer
    // per zf and bind it both as a template output and a track input.
    for (int i = 0;; ++i)
    {
        char name[16];
        std::snprintf(name, sizeof(name), "zf%d", i);
        const int tOut = Impl::bindIndex(te, name);
        if (tOut < 0)
            break;  // no more zf tensors
        const int tIn = Impl::bindIndex(ke, name);
        if (tIn < 0)
        {
            std::fprintf(stderr, "[hift/trt] track engine missing input %s\n",
                         name);
            return false;
        }
        const size_t n = volume(te->getBindingDimensions(tOut));
        if (n == 0)
            return false;
        d_->zfCount.push_back(n);
    }
    if (d_->zfCount.empty())
    {
        std::fprintf(stderr, "[hift/trt] no zf tensors found\n");
        return false;
    }

    // x input + loc/cls1/cls2 outputs.
    const int xIdx = Impl::bindIndex(ke, "x");
    const int locIdx = Impl::bindIndex(ke, "loc");
    const int cls1Idx = Impl::bindIndex(ke, "cls1");
    const int cls2Idx = Impl::bindIndex(ke, "cls2");
    if (xIdx < 0 || locIdx < 0 || cls1Idx < 0 || cls2Idx < 0)
    {
        std::fprintf(stderr,
                     "[hift/trt] track engine missing x/loc/cls1/cls2\n");
        return false;
    }
    d_->xCount = volume(ke->getBindingDimensions(xIdx));
    d_->locCount = volume(ke->getBindingDimensions(locIdx));
    d_->cls1Count = volume(ke->getBindingDimensions(cls1Idx));
    d_->cls2Count = volume(ke->getBindingDimensions(cls2Idx));

    // Allocate device buffers.
    auto alloc = [](void** p, size_t count) {
        return cudaMalloc(p, count * sizeof(float)) == cudaSuccess;
    };
    d_->dZf.assign(d_->zfCount.size(), nullptr);
    bool okAlloc = alloc(&d_->dZ, d_->zCount) && alloc(&d_->dX, d_->xCount) &&
                   alloc(&d_->dLoc, d_->locCount) &&
                   alloc(&d_->dCls1, d_->cls1Count) &&
                   alloc(&d_->dCls2, d_->cls2Count);
    for (size_t i = 0; i < d_->zfCount.size() && okAlloc; ++i)
        okAlloc = alloc(&d_->dZf[i], d_->zfCount[i]);
    if (!okAlloc)
    {
        std::fprintf(stderr, "[hift/trt] CUDA allocation failed\n");
        return false;
    }

    std::fprintf(stderr,
                 "[hift/trt] ready: z=%zu x=%zu zf=%zu loc=%zu cls1=%zu "
                 "cls2=%zu\n",
                 d_->zCount, d_->xCount, d_->zfCount.size(), d_->locCount,
                 d_->cls1Count, d_->cls2Count);
    d_->ok = true;
    return true;
}

bool TrtHiFT::setTemplate(const float* z_chw)
{
    if (!d_->ok)
        return false;
    auto* te = d_->tmpl.engine.get();
    if (cudaMemcpyAsync(d_->dZ, z_chw, d_->zCount * sizeof(float),
                        cudaMemcpyHostToDevice, d_->stream) != cudaSuccess)
        return false;

    // Bindings: z at its index, each zf output at its index.
    std::vector<void*> bindings(te->getNbBindings(), nullptr);
    bindings[Impl::bindIndex(te, "z")] = d_->dZ;
    for (size_t i = 0; i < d_->dZf.size(); ++i)
    {
        char name[16];
        std::snprintf(name, sizeof(name), "zf%zu", i);
        bindings[Impl::bindIndex(te, name)] = d_->dZf[i];
    }
    if (!d_->tmpl.context->enqueueV2(bindings.data(), d_->stream, nullptr))
        return false;
    if (cudaStreamSynchronize(d_->stream) != cudaSuccess)
        return false;
    d_->hasTemplate = true;
    return true;
}

bool TrtHiFT::track(const float* x_chw, HiFTTrackOut& out)
{
    if (!d_->ok || !d_->hasTemplate)
        return false;
    auto* ke = d_->trk.engine.get();
    if (cudaMemcpyAsync(d_->dX, x_chw, d_->xCount * sizeof(float),
                        cudaMemcpyHostToDevice, d_->stream) != cudaSuccess)
        return false;

    std::vector<void*> bindings(ke->getNbBindings(), nullptr);
    bindings[Impl::bindIndex(ke, "x")] = d_->dX;
    for (size_t i = 0; i < d_->dZf.size(); ++i)
    {
        char name[16];
        std::snprintf(name, sizeof(name), "zf%zu", i);
        bindings[Impl::bindIndex(ke, name)] = d_->dZf[i];  // shared with template
    }
    bindings[Impl::bindIndex(ke, "loc")] = d_->dLoc;
    bindings[Impl::bindIndex(ke, "cls1")] = d_->dCls1;
    bindings[Impl::bindIndex(ke, "cls2")] = d_->dCls2;

    if (!d_->trk.context->enqueueV2(bindings.data(), d_->stream, nullptr))
        return false;

    out.loc.resize(d_->locCount);
    out.cls1.resize(d_->cls1Count);
    out.cls2.resize(d_->cls2Count);
    if (cudaMemcpyAsync(out.loc.data(), d_->dLoc, d_->locCount * sizeof(float),
                        cudaMemcpyDeviceToHost, d_->stream) != cudaSuccess ||
        cudaMemcpyAsync(out.cls1.data(), d_->dCls1, d_->cls1Count * sizeof(float),
                        cudaMemcpyDeviceToHost, d_->stream) != cudaSuccess ||
        cudaMemcpyAsync(out.cls2.data(), d_->dCls2, d_->cls2Count * sizeof(float),
                        cudaMemcpyDeviceToHost, d_->stream) != cudaSuccess)
        return false;
    if (cudaStreamSynchronize(d_->stream) != cudaSuccess)
        return false;
    return true;
}

}  // namespace hift
}  // namespace cr

#else  // !CVTRACKER_WITH_TENSORRT

namespace cr {
namespace hift {

struct TrtHiFT::Impl {};
TrtHiFT::TrtHiFT() : d_(nullptr) {}
TrtHiFT::~TrtHiFT() = default;
bool TrtHiFT::ready() const { return false; }
bool TrtHiFT::initialise(const TrtHiFTConfig&)
{
    std::fprintf(stderr,
                 "[hift] TensorRT not compiled in - rebuild with "
                 "-DCVTRACKER_WITH_TENSORRT=ON\n");
    return false;
}
bool TrtHiFT::setTemplate(const float*) { return false; }
bool TrtHiFT::track(const float*, HiFTTrackOut&) { return false; }

}  // namespace hift
}  // namespace cr

#endif  // CVTRACKER_WITH_TENSORRT
