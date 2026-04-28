#include "vamana_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <cstdlib>

// ============================================================================
// Destructor
// ============================================================================

VamanaIndex::~VamanaIndex() {
    if (owns_data_ && data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

// ============================================================================
// Greedy Search
// ============================================================================
// Beam search starting from start_node_. Maintains a candidate set of at most
// L nodes, always expanding the closest unvisited node. Returns when no
// unvisited candidates remain.
//
// Uses std::set<Candidate> as an ordered container — simple, correct, and
// easy for students to understand and modify.

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L) const {
    // Candidate set: ordered by (distance, id). Bounded at size L.
    std::set<Candidate> candidate_set;
    // Track which nodes we've already expanded (visited).
    std::vector<bool> visited(npts_, false);

    uint32_t dist_cmps = 0;

    // Seed with start node
    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    candidate_set.insert({start_dist, start_node_});
    visited[start_node_] = true;

    // Track which candidates have been expanded (their neighbors explored).
    // We iterate through candidate_set; entries before our "frontier" pointer
    // have been expanded. We use a simple approach: keep scanning from the
    // beginning of the set for the first un-expanded entry.
    std::set<uint32_t> expanded;

    while (true) {
        // Find closest candidate that hasn't been expanded yet
        uint32_t best_node = UINT32_MAX;
        for (const auto& [dist, id] : candidate_set) {
            if (expanded.find(id) == expanded.end()) {
                best_node = id;
                break;
            }
        }
        if (best_node == UINT32_MAX)
            break;  // all candidates expanded

        expanded.insert(best_node);

        // Expand: evaluate all neighbors of best_node
        // Copy neighbor list under lock to avoid data race with parallel build
        // (another thread might push_back / reallocate graph_[best_node]).
        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }
        for (uint32_t nbr : neighbors) {
            if (visited[nbr])
                continue;
            visited[nbr] = true;

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

            // Insert if candidate set isn't full or this is closer than worst
            if (candidate_set.size() < L) {
                candidate_set.insert({d, nbr});
            } else {
                auto worst = std::prev(candidate_set.end());
                if (d < worst->first) {
                    candidate_set.erase(worst);
                    candidate_set.insert({d, nbr});
                }
            }
        }
    }

    // Convert to sorted vector
    std::vector<Candidate> results(candidate_set.begin(), candidate_set.end());
    return {results, dist_cmps};
}

// ============================================================================
// Robust Prune (Alpha-RNG Rule)
// ============================================================================
// Given a node and a set of candidates, greedily select neighbors that are
// "diverse" — a candidate c is added only if it's not too close to any
// already-selected neighbor (within a factor of alpha).
//
// Formally: add c if for ALL already-chosen neighbors n:
//     dist(node, c) <= alpha * dist(c, n)
//
// This ensures good graph navigability by keeping some long-range edges
// (alpha > 1 makes it easier for a candidate to survive pruning).

void VamanaIndex::robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                               float alpha, uint32_t R) {
    // Remove self from candidates if present
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c) { return c.second == node; }),
        candidates.end());

    // Sort by distance to node (ascending)
    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    for (const auto& [dist_to_node, cand_id] : candidates) {
        if (new_neighbors.size() >= R)
            break;

        // Check alpha-RNG condition against all already-selected neighbors
        bool keep = true;
        for (uint32_t selected : new_neighbors) {
            float dist_cand_to_selected =
                compute_l2sq(get_vector(cand_id), get_vector(selected), dim_);
            if (dist_to_node > alpha * dist_cand_to_selected) {
                keep = false;
                break;
            }
        }

        if (keep)
            new_neighbors.push_back(cand_id);
    }

    graph_[node] = std::move(new_neighbors);
}

// ============================================================================
// Build
// ============================================================================

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    // --- Load data ---
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    std::cout << "  Points: " << npts_ << ", Dimensions: " << dim_ << std::endl;

    if (L < R) {
        std::cerr << "Warning: L (" << L << ") < R (" << R
                  << "). Setting L = R." << std::endl;
        L = R;
    }

    // --- Initialize empty graph and per-node locks ---
    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    // --- Pick mediod start node ---
    start_node_ = find_medoid(); 
    std::cout << "  Medoid start node: " << start_node_ << std::endl;

    // --- Create random insertion order ---
    std::mt19937 rng(42); 
    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    auto it = std::find(perm.begin(), perm.end(), start_node_);
    if (it != perm.end()) {
        std::swap(perm[0], *it);
    }

    // --- Build graph: parallel insertion with per-node locking ---
    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);
    std::cout << "Building index (R=" << R << ", L=" << L
              << ", alpha=" << alpha << ", gamma=" << gamma
              << ", gammaR=" << gamma_R << ")..." << std::endl;

    Timer build_timer;

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];

        // Step 1: Search for this point in the current graph to find candidates
        auto [candidates, _dist_cmps] = greedy_search(get_vector(point), L);

        // Step 2: Prune candidates to get this point's neighbors
        // We don't need to lock graph_[point] here because each point appears
        // exactly once in the permutation — only this thread writes to it now.
        robust_prune(point, candidates, alpha, R);

        // Step 3: Add backward edges from each new neighbor back to this point
        for (uint32_t nbr : graph_[point]) {
            std::lock_guard<std::mutex> lock(locks_[nbr]);

            // Add backward edge
            graph_[nbr].push_back(point);

            // Step 4: If neighbor's degree exceeds gamma*R, prune its neighborhood
            if (graph_[nbr].size() > gamma_R) {
                // Build candidate list from current neighbors of nbr
                std::vector<Candidate> nbr_candidates;
                nbr_candidates.reserve(graph_[nbr].size());
                for (uint32_t nn : graph_[nbr]) {
                    float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                    nbr_candidates.push_back({d, nn});
                }
                robust_prune(nbr, nbr_candidates, alpha, R);
            }
        }

        // Progress reporting (from one thread only)
        if (idx % 10000 == 0) {
            #pragma omp critical
            {
                std::cout << "\r  Inserted " << idx << " / " << npts_
                          << " points" << std::flush;
            }
        }
    }

    double build_time = build_timer.elapsed_seconds();

    // Compute average degree
    size_t total_edges = 0;
    for (uint32_t i = 0; i < npts_; i++)
        total_edges += graph_[i].size();
    double avg_degree = (double)total_edges / npts_;

    std::cout << "\n  Build complete in " << build_time << " seconds."
              << std::endl;
    std::cout << "  Average out-degree: " << avg_degree << std::endl;
}

// ============================================================================
// Search
// ============================================================================

SearchResult VamanaIndex::search(const float* query, uint32_t K, uint32_t L) const {
    if (L < K) L = K;

    Timer t;
    auto [candidates, dist_cmps] = greedy_search(query, L);
    double latency = t.elapsed_us();

    // Return top-K results
    SearchResult result;
    result.dist_cmps = dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++) {
        result.ids.push_back(candidates[i].second);
    }
    return result;
}

// ============================================================================
// Save / Load
// ============================================================================
// Binary format:
//   [uint32] npts
//   [uint32] dim
//   [uint32] start_node
//   For each node i in [0, npts):
//     [uint32] degree
//     [uint32 * degree] neighbor IDs

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open file for writing: " + path);

    out.write(reinterpret_cast<const char*>(&npts_), 4);
    out.write(reinterpret_cast<const char*>(&dim_), 4);
    out.write(reinterpret_cast<const char*>(&start_node_), 4);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg = graph_[i].size();
        out.write(reinterpret_cast<const char*>(&deg), 4);
        if (deg > 0) {
            out.write(reinterpret_cast<const char*>(graph_[i].data()),
                      deg * sizeof(uint32_t));
        }
    }

    std::cout << "Index saved to " << path << std::endl;
}

void VamanaIndex::load(const std::string& index_path,
                       const std::string& data_path) {
    // Load data vectors
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    // Load graph
    std::ifstream in(index_path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open index file: " + index_path);

    uint32_t file_npts, file_dim;
    in.read(reinterpret_cast<char*>(&file_npts), 4);
    in.read(reinterpret_cast<char*>(&file_dim), 4);
    in.read(reinterpret_cast<char*>(&start_node_), 4);

    if (file_npts != npts_ || file_dim != dim_)
        throw std::runtime_error(
            "Index/data mismatch: index has " + std::to_string(file_npts) +
            "x" + std::to_string(file_dim) + ", data has " +
            std::to_string(npts_) + "x" + std::to_string(dim_));

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg;
        in.read(reinterpret_cast<char*>(&deg), 4);
        graph_[i].resize(deg);
        if (deg > 0) {
            in.read(reinterpret_cast<char*>(graph_[i].data()),
                    deg * sizeof(uint32_t));
        }
    }

    std::cout << "Index loaded: " << npts_ << " points, " << dim_
              << " dims, start=" << start_node_ << std::endl;
}

uint32_t VamanaIndex::find_medoid() {
    std::cout << "Calculating medoid for initialization..." << std::endl;
    
    // 1. Calculate the Centroid (average vector)
    std::vector<float> centroid(dim_, 0.0f);
    for (uint32_t i = 0; i < npts_; i++) {
        const float* v = get_vector(i);
        for (uint32_t d = 0; d < dim_; d++) {
            centroid[d] += v[d];
        }
    }
    for (uint32_t d = 0; d < dim_; d++) {
        centroid[d] /= npts_;
    }

    // 2. Find the point closest to the Centroid
    uint32_t medoid_id = 0;
    float min_dist = std::numeric_limits<float>::max();

    // Note: For massive datasets, you can sample 10,000 random points 
    // here instead of checking all npts_ to save time.
    for (uint32_t i = 0; i < npts_; i++) {
        float d = compute_l2sq(centroid.data(), get_vector(i), dim_);
        if (d < min_dist) {
            min_dist = d;
            medoid_id = i;
        }
    }

    std::cout << "Medoid found at index: " << medoid_id << std::endl;
    return medoid_id;
}