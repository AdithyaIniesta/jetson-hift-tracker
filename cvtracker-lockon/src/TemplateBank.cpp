#include "TemplateBank.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace cr {
namespace vtracker {

namespace {

constexpr uint32_t BANK_MAGIC   = 0x314B4243u;  // 'CBK1' little-endian
constexpr uint32_t BANK_VERSION = 1u;

// Little-endian encode/decode helpers — portable across ARM/x86.
inline void put_u32(std::vector<unsigned char>& b, uint32_t v)
{
    b.push_back((unsigned char)(v & 0xFFu));
    b.push_back((unsigned char)((v >> 8) & 0xFFu));
    b.push_back((unsigned char)((v >> 16) & 0xFFu));
    b.push_back((unsigned char)((v >> 24) & 0xFFu));
}
inline void put_i32(std::vector<unsigned char>& b, int32_t v)
{
    put_u32(b, (uint32_t)v);
}
inline void put_f32(std::vector<unsigned char>& b, float v)
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    put_u32(b, u);
}
inline bool get_u32(const unsigned char* p, size_t sz, size_t& off, uint32_t& v)
{
    if (off + 4 > sz) return false;
    v = (uint32_t)p[off] | ((uint32_t)p[off + 1] << 8) |
        ((uint32_t)p[off + 2] << 16) | ((uint32_t)p[off + 3] << 24);
    off += 4;
    return true;
}
inline bool get_i32(const unsigned char* p, size_t sz, size_t& off, int32_t& v)
{
    uint32_t u;
    if (!get_u32(p, sz, off, u)) return false;
    v = (int32_t)u;
    return true;
}
inline bool get_f32(const unsigned char* p, size_t sz, size_t& off, float& v)
{
    uint32_t u;
    if (!get_u32(p, sz, off, u)) return false;
    std::memcpy(&v, &u, sizeof(v));
    return true;
}

}  // namespace

void TemplateBank::configure(int capacity, int minSpacingFrames)
{
    m_capacity = capacity < 2 ? 2 : capacity;
    m_minSpacing = minSpacingFrames < 1 ? 1 : minSpacingFrames;
}

void TemplateBank::clear()
{
    m_items.clear();
    m_lastAddFrame = -1000000;
    m_dim = 0;
}

bool TemplateBank::normalize(std::vector<float>& v)
{
    double s = 0.0;
    for (float x : v)
        s += (double)x * x;
    if (s < 1e-12)
        return false;
    const float inv = 1.0f / (float)std::sqrt(s);
    for (float& x : v)
        x *= inv;
    return true;
}

void TemplateBank::seed(const std::vector<float>& emb, int frameCounter)
{
    Item it;
    it.e = emb;
    it.frame = frameCounter;
    if (!normalize(it.e))
        return;
    clear();
    m_dim = it.e.size();
    m_lastAddFrame = frameCounter;
    m_items.push_back(std::move(it));
}

void TemplateBank::maybeAdd(const std::vector<float>& emb, int frameCounter)
{
    if (m_items.empty())
    {
        seed(emb, frameCounter);
        return;
    }
    if (emb.size() != m_dim)
        return;
    if (frameCounter - m_lastAddFrame < m_minSpacing)
        return;
    Item it;
    it.e = emb;
    it.frame = frameCounter;
    if (!normalize(it.e))
        return;
    m_lastAddFrame = frameCounter;
    m_items.push_back(std::move(it));
    // Evict the oldest non-seed template (index 1) when over capacity;
    // index 0 is the capture seed and is kept forever.
    if ((int)m_items.size() > m_capacity)
        m_items.erase(m_items.begin() + 1);
}

float TemplateBank::similarity(const std::vector<float>& emb) const
{
    if (m_items.empty() || emb.size() != m_dim)
        return -2.0f;
    // Normalize the query once (locally, without modifying the argument).
    double s = 0.0;
    for (float x : emb)
        s += (double)x * x;
    if (s < 1e-12)
        return -2.0f;
    const float inv = 1.0f / (float)std::sqrt(s);

    float best = -2.0f;
    for (const Item& it : m_items)
    {
        double dot = 0.0;
        for (size_t i = 0; i < m_dim; ++i)
            dot += (double)it.e[i] * emb[i];
        const float sim = (float)dot * inv;
        if (sim > best)
            best = sim;
    }
    return best;
}

bool TemplateBank::serialize(std::vector<unsigned char>& out) const
{
    out.clear();
    if (m_items.empty() || m_dim == 0)
        return false;

    // Header
    put_u32(out, BANK_MAGIC);
    put_u32(out, BANK_VERSION);
    put_u32(out, (uint32_t)m_dim);
    put_u32(out, (uint32_t)m_items.size());

    // Body
    for (const Item& it : m_items)
    {
        if (it.e.size() != m_dim)
        {
            out.clear();
            return false;
        }
        put_i32(out, it.frame);
        for (size_t i = 0; i < m_dim; ++i)
            put_f32(out, it.e[i]);
    }
    return true;
}

bool TemplateBank::deserialize(const std::vector<unsigned char>& in)
{
    const unsigned char* p = in.data();
    const size_t sz = in.size();
    size_t off = 0;

    uint32_t magic = 0, version = 0, dim = 0, n = 0;
    if (!get_u32(p, sz, off, magic)   || magic   != BANK_MAGIC)   return false;
    if (!get_u32(p, sz, off, version) || version != BANK_VERSION) return false;
    if (!get_u32(p, sz, off, dim)     || dim     == 0)            return false;
    if (!get_u32(p, sz, off, n)       || n       == 0)            return false;

    // Cheap sanity — reject buffers that clearly can't hold n items.
    const size_t need = (size_t)n * (4u + (size_t)dim * 4u);
    if (off + need > sz) return false;

    std::vector<Item> parsed;
    parsed.reserve(n);
    int32_t lastFrame = -1000000;
    for (uint32_t i = 0; i < n; ++i)
    {
        Item it;
        if (!get_i32(p, sz, off, it.frame)) return false;
        it.e.resize(dim);
        for (uint32_t k = 0; k < dim; ++k)
            if (!get_f32(p, sz, off, it.e[k])) return false;
        if (it.frame > lastFrame) lastFrame = it.frame;
        parsed.push_back(std::move(it));
    }

    // Commit atomically — only after all reads succeeded so a bad
    // buffer never leaves the bank half-loaded.
    m_items      = std::move(parsed);
    m_dim        = dim;
    m_lastAddFrame = lastFrame;
    return true;
}

} // namespace vtracker
} // namespace cr
