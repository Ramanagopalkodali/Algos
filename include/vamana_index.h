#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include <string>

// Result of a single query search.
struct SearchResult {
    std::vector<uint32_t> ids;  // nearest neighbor IDs (sorted by distance)
    uint32_t dist_cmps;         // number of distance computations
    double latency_us;          // search latency in microseconds
};

// Vamana graph-based approximate nearest neighbor index.
//
// Key concepts:
//   - The graph is built incrementally: each point is inserted by searching
//     the current graph, pruning candidates with the alpha-RNG rule, and
//     adding forward + backward edges.
//   - Greedy search starts from a fixed start node and follows edges to
//     find nearest neighbors, maintaining a candidate list of size L.
//   - The alpha parameter controls edge diversity (alpha > 1 favors long-range
//     edges for better navigability).
//   - R is the max out-degree; gamma*R is the threshold that triggers pruning
//     on neighbor nodes when backward edges are added.
class VamanaIndex {
  public:
    VamanaIndex() = default;
    ~VamanaIndex();

    // ---- Build ----
    // Loads data from an fbin file and builds the Vamana graph.
    //   R:     max out-degree per node
    //   L:     search list size during construction (L >= R)
    //   alpha: RNG pruning parameter (typically 1.0 - 1.5)
    //   gamma: max-degree multiplier for triggering neighbor pruning (e.g. 1.5)
    void build(const std::string& data_path, uint32_t R, uint32_t L,
               float alpha, float gamma);

    // ---- Search ----
    // Search for K nearest neighbors of a query vector.
    //   query: pointer to query vector (must have dim_ floats)
    //   K:     number of nearest neighbors to return
    //   L:     search list size (L >= K)
    SearchResult search(const float* query, uint32_t K, uint32_t L) const;

    // ---- Persistence ----
    // Save index (graph + metadata) to a binary file.
    void save(const std::string& path) const;

    // Load index from a binary file. Data file must also be loaded separately.
    void load(const std::string& index_path, const std::string& data_path);

    uint32_t get_npts() const { return npts_; }
    uint32_t get_dim()  const { return dim_; }

  private:
    // A candidate = (distance, node_id). Ordered by distance.
    using Candidate = std::pair<float, uint32_t>;

    // ---- Core algorithms ----
    // Add this under "Core algorithms" or "Data"
    uint32_t find_medoid();
    // Greedy search starting from start_node_.
    // Returns (sorted candidate list, number of distance computations).
    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search(const float* query, uint32_t L) const;

    // Alpha-RNG pruning: selects a diverse subset of candidates as neighbors.
    // Modifies graph_[node] in place. Candidates should NOT include node itself.
    void robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                      float alpha, uint32_t R);

    // ---- Data ----
    float*   data_    = nullptr;  // contiguous row-major [npts x dim], aligned
    uint32_t npts_    = 0;
    uint32_t dim_     = 0;
    bool     owns_data_ = false;  // whether we allocated data_

    // ---- Graph ----
    std::vector<std::vector<uint32_t>> graph_;  // adjacency lists
    uint32_t start_node_ = 0;

    // ---- Concurrency ----
    // Per-node locks for parallel build (mutable so search can be const).
    mutable std::vector<std::mutex> locks_;

    // ---- Helpers ----
    const float* get_vector(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }
};
