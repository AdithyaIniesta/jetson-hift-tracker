#pragma once
#include <cstddef>
#include <vector>

namespace cr {
namespace vtracker {

/// Gallery of L2-normalized appearance embeddings of the tracked object.
///
/// The bank is seeded at capture and grown with high-confidence, temporally
/// spaced samples during tracking, giving the verifier a memory of the
/// object across scales / orientations / lighting instead of a single
/// decaying template. Similarity of a query embedding is the maximum cosine
/// similarity over all stored templates, so the object only has to match
/// one remembered appearance.
class TemplateBank
{
public:
    /// capacity - max stored templates; minSpacingFrames - minimum frame
    /// distance between consecutive adds (diversity over redundancy).
    void configure(int capacity, int minSpacingFrames);

    void clear();
    bool empty() const { return m_items.empty(); }
    int size() const { return (int)m_items.size(); }

    /// Add unconditionally (capture seed). The seed is never evicted.
    void seed(const std::vector<float>& emb, int frameCounter);

    /// Add if at least minSpacingFrames after the previous add; evicts the
    /// oldest non-seed template when over capacity.
    void maybeAdd(const std::vector<float>& emb, int frameCounter);

    /// Max cosine similarity of emb against the bank, [-1..1].
    /// Returns -2.0f when the bank is empty or dimensions mismatch.
    float similarity(const std::vector<float>& emb) const;

    /// Serialise the bank to a versioned byte buffer for cross-tracker
    /// template transfer (dual-camera handoff). Layout:
    ///   [magic u32 = 'CBK1'][version u32 = 1][dim u32][n u32]
    ///   then n entries: [frame i32][dim * float]
    /// Little-endian, embedded floats already L2-normalised.
    /// Returns false if the bank is empty.
    bool serialize(std::vector<unsigned char>& out) const;

    /// Replace this bank's contents from a byte buffer previously
    /// produced by serialize(). Returns false on magic/version
    /// mismatch, size inconsistency, or zero items — the bank is
    /// left untouched in those cases.
    bool deserialize(const std::vector<unsigned char>& in);

private:
    struct Item
    {
        std::vector<float> e; // L2-normalized
        int frame{0};
    };

    /// L2-normalize v in place; false if the norm is ~0.
    static bool normalize(std::vector<float>& v);

    std::vector<Item> m_items;
    int m_capacity{12};
    int m_minSpacing{30};
    int m_lastAddFrame{-1000000};
    size_t m_dim{0};
};

} // namespace vtracker
} // namespace cr
