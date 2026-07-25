#include "cvtracker/VTracker.h"
#include "cvtracker/CvTrackerVersion.h"
#include <cstring>

namespace cr {
namespace vtracker {

namespace {

constexpr uint8_t HEADER_COMMAND = 0x00;
constexpr uint8_t HEADER_SET_PARAM = 0x01;
constexpr uint8_t HEADER_PARAMS = 0x02;

inline void writeI32(uint8_t* p, int32_t v) { std::memcpy(p, &v, 4); }
inline void writeF32(uint8_t* p, float v) { std::memcpy(p, &v, 4); }
inline int32_t readI32(const uint8_t* p) { int32_t v; std::memcpy(&v, p, 4); return v; }
inline float readF32(const uint8_t* p) { float v; std::memcpy(&v, p, 4); return v; }

} // namespace

void VTracker::encodeSetParamCommand(uint8_t* data, int& size,
                                     VTrackerParam id, float value)
{
    data[0] = HEADER_SET_PARAM;
    data[1] = CVTRACKER_VERSION_MAJOR;
    data[2] = CVTRACKER_VERSION_MINOR;
    writeI32(data + 3, (int32_t)id);
    writeF32(data + 7, value);
    size = 11;
}

void VTracker::encodeCommand(uint8_t* data, int& size, VTrackerCommand id,
                             float arg1, float arg2, float arg3)
{
    data[0] = HEADER_COMMAND;
    data[1] = CVTRACKER_VERSION_MAJOR;
    data[2] = CVTRACKER_VERSION_MINOR;
    writeI32(data + 3, (int32_t)id);
    writeF32(data + 7, arg1);
    writeF32(data + 11, arg2);
    writeF32(data + 15, arg3);
    size = 19;
}

int VTracker::decodeCommand(uint8_t* data, int size,
                            VTrackerParam& paramId,
                            VTrackerCommand& commandId,
                            float& value1, float& value2, float& value3)
{
    if (data == nullptr)
        return -1;
    if (size == 11 && data[0] == HEADER_SET_PARAM)
    {
        if (data[1] != CVTRACKER_VERSION_MAJOR)
            return -1;
        paramId = (VTrackerParam)readI32(data + 3);
        value1 = readF32(data + 7);
        return 1;
    }
    if (size == 19 && data[0] == HEADER_COMMAND)
    {
        if (data[1] != CVTRACKER_VERSION_MAJOR)
            return -1;
        commandId = (VTrackerCommand)readI32(data + 3);
        value1 = readF32(data + 7);
        value2 = readF32(data + 11);
        value3 = readF32(data + 15);
        return 0;
    }
    return -1;
}

bool VTracker::decodeAndExecuteCommand(uint8_t* data, int size)
{
    VTrackerParam paramId{};
    VTrackerCommand commandId{};
    float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    switch (decodeCommand(data, size, paramId, commandId, v1, v2, v3))
    {
    case 0: return executeCommand(commandId, v1, v2, v3);
    case 1: return setParam(paramId, v1);
    default: return false;
    }
}

// Serialization helper: walks fields in declaration order.
namespace {

struct FieldWriter
{
    uint8_t* buf;
    int capacity;
    int pos{0};
    bool ok{true};

    void put(const void* src, int n)
    {
        if (!ok || pos + n > capacity) { ok = false; return; }
        std::memcpy(buf + pos, src, n);
        pos += n;
    }
    void putI(int32_t v, bool masked) { if (masked) put(&v, 4); }
    void putF(float v, bool masked) { if (masked) put(&v, 4); }
    void putB(bool v, bool masked) { uint8_t b = v ? 1 : 0; if (masked) put(&b, 1); }
};

struct FieldReader
{
    const uint8_t* buf;
    int sizeLeft;
    int pos{0};
    bool ok{true};

    bool get(void* dst, int n)
    {
        if (!ok || pos + n > sizeLeft) { ok = false; return false; }
        std::memcpy(dst, buf + pos, n);
        pos += n;
        return true;
    }
    void getI(int& v, bool masked) { if (masked) { int32_t t; if (get(&t, 4)) v = t; } }
    void getF(float& v, bool masked) { if (masked) { float t; if (get(&t, 4)) v = t; } }
    void getB(bool& v, bool masked) { if (masked) { uint8_t t; if (get(&t, 1)) v = t != 0; } }
};

constexpr int NUM_FIELDS = 40;

void maskToBits(const VTrackerParamsMask& m, bool bits[NUM_FIELDS])
{
    const bool src[NUM_FIELDS] = {
        m.mode, m.rectX, m.rectY, m.rectWidth, m.rectHeight,
        m.objectX, m.objectY, m.objectWidth, m.objectHeight,
        m.lostModeFrameCounter, m.frameCounter, m.frameWidth, m.frameHeight,
        m.searchWindowWidth, m.searchWindowHeight, m.searchWindowX,
        m.searchWindowY, m.lostModeOption, m.frameBufferSize,
        m.maxFramesInLostMode, m.processedFrameId, m.frameId, m.velX, m.velY,
        m.detectionProbability, m.rectAutoSize, m.rectAutoPosition,
        m.multipleThreads, m.numChannels, m.type, m.processingTimeMks,
        m.custom1, m.custom2, m.custom3, m.ekfEnabled,
        m.dnnVerifierEnabled, m.dnnVerifyInterval, m.dnnVetoThreshold,
        m.dnnAcceptThreshold, m.dnnSimilarity };
    std::memcpy(bits, src, sizeof(src));
}

} // namespace

bool VTrackerParams::encode(uint8_t* data, int bufferSize, int& size,
                            VTrackerParamsMask* mask)
{
    if (data == nullptr || bufferSize < 8)
        return false;

    VTrackerParamsMask defaultMask;
    if (mask == nullptr)
        mask = &defaultMask;
    bool bits[NUM_FIELDS];
    maskToBits(*mask, bits);

    data[0] = HEADER_PARAMS;
    data[1] = CVTRACKER_VERSION_MAJOR;
    data[2] = CVTRACKER_VERSION_MINOR;
    const int maskBytes = (NUM_FIELDS + 7) / 8;
    if (3 + maskBytes > bufferSize)
        return false;
    std::memset(data + 3, 0, maskBytes);
    for (int i = 0; i < NUM_FIELDS; ++i)
        if (bits[i])
            data[3 + i / 8] |= (uint8_t)(1 << (7 - (i % 8)));

    FieldWriter w{data, bufferSize, 3 + maskBytes, true};
    int i = 0;
    w.putI(mode, bits[i++]);
    w.putI(rectX, bits[i++]);
    w.putI(rectY, bits[i++]);
    w.putI(rectWidth, bits[i++]);
    w.putI(rectHeight, bits[i++]);
    w.putI(objectX, bits[i++]);
    w.putI(objectY, bits[i++]);
    w.putI(objectWidth, bits[i++]);
    w.putI(objectHeight, bits[i++]);
    w.putI(lostModeFrameCounter, bits[i++]);
    w.putI(frameCounter, bits[i++]);
    w.putI(frameWidth, bits[i++]);
    w.putI(frameHeight, bits[i++]);
    w.putI(searchWindowWidth, bits[i++]);
    w.putI(searchWindowHeight, bits[i++]);
    w.putI(searchWindowX, bits[i++]);
    w.putI(searchWindowY, bits[i++]);
    w.putI(lostModeOption, bits[i++]);
    w.putI(frameBufferSize, bits[i++]);
    w.putI(maxFramesInLostMode, bits[i++]);
    w.putI(processedFrameId, bits[i++]);
    w.putI(frameId, bits[i++]);
    w.putF(velX, bits[i++]);
    w.putF(velY, bits[i++]);
    w.putF(detectionProbability, bits[i++]);
    w.putB(rectAutoSize, bits[i++]);
    w.putB(rectAutoPosition, bits[i++]);
    w.putB(multipleThreads, bits[i++]);
    w.putI(numChannels, bits[i++]);
    w.putI(type, bits[i++]);
    w.putI(processingTimeMks, bits[i++]);
    w.putF(custom1, bits[i++]);
    w.putF(custom2, bits[i++]);
    w.putF(custom3, bits[i++]);
    w.putB(ekfEnabled, bits[i++]);
    w.putB(dnnVerifierEnabled, bits[i++]);
    w.putI(dnnVerifyInterval, bits[i++]);
    w.putF(dnnVetoThreshold, bits[i++]);
    w.putF(dnnAcceptThreshold, bits[i++]);
    w.putF(dnnSimilarity, bits[i++]);
    if (!w.ok)
        return false;
    size = w.pos;
    return true;
}

bool VTrackerParams::decode(uint8_t* data, int dataSize)
{
    const int maskBytes = (NUM_FIELDS + 7) / 8;
    if (data == nullptr || dataSize < 3 + maskBytes)
        return false;
    if (data[0] != HEADER_PARAMS || data[1] != CVTRACKER_VERSION_MAJOR)
        return false;

    bool bits[NUM_FIELDS];
    for (int i = 0; i < NUM_FIELDS; ++i)
        bits[i] = (data[3 + i / 8] & (1 << (7 - (i % 8)))) != 0;

    FieldReader r{data + 3 + maskBytes, dataSize - 3 - maskBytes, 0, true};
    int i = 0;
    r.getI(mode, bits[i++]);
    r.getI(rectX, bits[i++]);
    r.getI(rectY, bits[i++]);
    r.getI(rectWidth, bits[i++]);
    r.getI(rectHeight, bits[i++]);
    r.getI(objectX, bits[i++]);
    r.getI(objectY, bits[i++]);
    r.getI(objectWidth, bits[i++]);
    r.getI(objectHeight, bits[i++]);
    r.getI(lostModeFrameCounter, bits[i++]);
    r.getI(frameCounter, bits[i++]);
    r.getI(frameWidth, bits[i++]);
    r.getI(frameHeight, bits[i++]);
    r.getI(searchWindowWidth, bits[i++]);
    r.getI(searchWindowHeight, bits[i++]);
    r.getI(searchWindowX, bits[i++]);
    r.getI(searchWindowY, bits[i++]);
    r.getI(lostModeOption, bits[i++]);
    r.getI(frameBufferSize, bits[i++]);
    r.getI(maxFramesInLostMode, bits[i++]);
    r.getI(processedFrameId, bits[i++]);
    r.getI(frameId, bits[i++]);
    r.getF(velX, bits[i++]);
    r.getF(velY, bits[i++]);
    r.getF(detectionProbability, bits[i++]);
    r.getB(rectAutoSize, bits[i++]);
    r.getB(rectAutoPosition, bits[i++]);
    r.getB(multipleThreads, bits[i++]);
    r.getI(numChannels, bits[i++]);
    r.getI(type, bits[i++]);
    r.getI(processingTimeMks, bits[i++]);
    r.getF(custom1, bits[i++]);
    r.getF(custom2, bits[i++]);
    r.getF(custom3, bits[i++]);
    r.getB(ekfEnabled, bits[i++]);
    r.getB(dnnVerifierEnabled, bits[i++]);
    r.getI(dnnVerifyInterval, bits[i++]);
    r.getF(dnnVetoThreshold, bits[i++]);
    r.getF(dnnAcceptThreshold, bits[i++]);
    r.getF(dnnSimilarity, bits[i++]);
    return r.ok;
}

} // namespace vtracker
} // namespace cr
