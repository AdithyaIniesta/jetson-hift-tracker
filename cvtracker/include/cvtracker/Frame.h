#pragma once
#include <cstdint>
#include <cstring>

namespace cr {
namespace video {

/// Make FOURCC code from four chars.
constexpr uint32_t makeFourcc(char a, char b, char c, char d)
{
    return (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
           ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}

/// Supported pixel formats (8 bit color depth RAW formats).
enum class Fourcc : uint32_t
{
    /// 8 bit monochrome.
    GRAY = makeFourcc('G', 'R', 'A', 'Y'),
    /// 24 bit YUV (Y, U, V per pixel).
    YUV24 = makeFourcc('Y', 'U', 'V', '3'),
    /// Packed YUV 4:2:2 (Y0 U0 Y1 V0).
    YUYV = makeFourcc('Y', 'U', 'Y', 'V'),
    /// Packed YUV 4:2:2 (U0 Y0 V0 Y1).
    UYVY = makeFourcc('U', 'Y', 'V', 'Y'),
    /// Planar Y + interleaved UV 4:2:0.
    NV12 = makeFourcc('N', 'V', '1', '2'),
    /// Planar Y + interleaved VU 4:2:0.
    NV21 = makeFourcc('N', 'V', '2', '1'),
    /// Planar YUV 4:2:0 (I420).
    YU12 = makeFourcc('Y', 'U', '1', '2'),
    /// Planar YVU 4:2:0.
    YV12 = makeFourcc('Y', 'V', '1', '2')
};

/// Video frame class. Owns its data buffer.
class Frame
{
public:
    /// Frame width, pixels.
    int width{0};
    /// Frame height, pixels.
    int height{0};
    /// Pixel format.
    Fourcc fourcc{Fourcc::GRAY};
    /// Frame data size, bytes.
    uint32_t size{0};
    /// Frame ID (set by user / data source).
    int32_t frameId{-1};
    /// Data source ID.
    int32_t sourceId{-1};
    /// Pointer to frame data.
    uint8_t* data{nullptr};

    Frame() = default;

    Frame(int w, int h, Fourcc f) { init(w, h, f); }

    Frame(const Frame& other) { *this = other; }

    Frame& operator=(const Frame& other)
    {
        if (this == &other)
            return *this;
        release();
        width = other.width;
        height = other.height;
        fourcc = other.fourcc;
        frameId = other.frameId;
        sourceId = other.sourceId;
        size = other.size;
        if (other.data != nullptr && other.size > 0)
        {
            data = new uint8_t[other.size];
            std::memcpy(data, other.data, other.size);
        }
        return *this;
    }

    ~Frame() { release(); }

    /// Allocate buffer for given size and format. Returns false on invalid args.
    bool init(int w, int h, Fourcc f)
    {
        release();
        uint32_t s = dataSize(w, h, f);
        if (s == 0)
            return false;
        width = w;
        height = h;
        fourcc = f;
        size = s;
        data = new uint8_t[s];
        std::memset(data, 0, s);
        return true;
    }

    /// Free buffer.
    void release()
    {
        delete[] data;
        data = nullptr;
        size = 0;
        width = 0;
        height = 0;
    }

    /// Data size in bytes for given dimensions and format. 0 if unsupported.
    static uint32_t dataSize(int w, int h, Fourcc f)
    {
        if (w <= 0 || h <= 0)
            return 0;
        const uint32_t n = (uint32_t)w * (uint32_t)h;
        switch (f)
        {
        case Fourcc::GRAY: return n;
        case Fourcc::YUV24: return n * 3u;
        case Fourcc::YUYV:
        case Fourcc::UYVY: return n * 2u;
        case Fourcc::NV12:
        case Fourcc::NV21:
        case Fourcc::YU12:
        case Fourcc::YV12: return n + n / 2u;
        default: return 0;
        }
    }
};

} // namespace video
} // namespace cr
