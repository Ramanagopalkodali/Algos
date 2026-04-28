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
#include <limits>

// FIX: Correct OpenMP detection. The macro _OPENMP is defined by the compiler
// when -fopenmp is passed. The stubs below only compile when OpenMP is absent,
// preventing duplicate-symbol errors if omp.h is also included.
#if defined(_OPENMP)
  #include <omp.h>
#else
  inline int omp_get_max_threads() { return 1; }
  inline int omp_get_thread_num()  { return 0; }
#endif

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

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L, float epsilon) const {

    auto& scratch = get_scratchpad();
    scratch.current_token++;

    // Handle token wraparound: reset both arrays
    if (scratch.current_token == 0) {
        std::fill(scratch.visited.begin(),  scratch.visited.end(),  0);
        std::fill(scratch.expanded.begin(), scratch.expanded.end(), 0);
        scratch.current_token = 1;
    }

    uint32_t token = scratch.current_token;

    std::set<Candidate> candidate_set;
    uint32_t dist_cmps = 0;

    // Seed with start node
    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    candidate_set.insert({start_dist, start_node_});
    scratch.visited[start_node_] = token;

    while (true) {
        // Find best unexpanded candidate
        uint32_t best_node = UINT32_MAX;
        float    best_unexpanded_dist = 0.0f;

        for (const auto& [dist, id] : candidate_set) {
            // FIX: use token-stamped expanded array instead of std::set<uint32_t>
            // This avoids O(log n) set lookups on every iteration.
            if (scratch.expanded[id] != token) {
                best_node             = id;
                best_unexpanded_dist  = dist;
                break;
            }
        }

        if (best_node == UINT32_MAX) break;

        // Adaptive exit: stop if best unexpanded is too far from global best
        if (epsilon > 0.0f) {
            float best_found_dist = candidate_set.begin()->first;
            if (best_unexpanded_dist > best_found_dist * (1.0f + epsilon)) break;
        }

        // FIX: mark expanded via token stamp (O(1), no allocation)
        scratch.expanded[best_node] = token;

        // Read neighbors under lock (safe for concurrent build)
        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }

        for (uint32_t nbr : neighbors) {
            if (scratch.visited[nbr] == token) continue;
            scratch.visited[nbr] = token;

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

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

    return {std::vector<Candidate>(candidate_set.begin(), candidate_set.end()), dist_cmps};
}

// ============================================================================
// Robust Prune (Alpha-RNG Rule)
// ============================================================================

void VamanaIndex::robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                               float alpha, uint32_t R) {
    // Remove self from candidates
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c) { return c.second == node; }),
        candidates.end());

    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    for (const auto& [dist_to_node, cand_id] : candidates) {
        if (new_neighbors.size() >= R) break;

        bool keep = true;
        for (uint32_t selected : new_neighbors) {
            float dist_cand_to_selected =
                compute_l2sq(get_vector(cand_id), get_vector(selected), dim_);
            if (dist_to_node > alpha * dist_cand_to_selected) {
                keep = false;
                break;
            }
        }
        if (keep) new_neighbors.push_back(cand_id);
    }

    // FIX: caller must hold locks_[node] before calling robust_prune when
    // operating on neighbor nodes during parallel build. See build() below.
    graph_[node] = std::move(new_neighbors);
}

// ============================================================================
// Build
// ============================================================================

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_      = mat.npts;
    dim_       = mat.dims;
    data_      = mat.data.release();
    owns_data_ = true;

    init_scratchpads();

    if (L < R) L = R;

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    start_node_ = find_medoid();
    std::cout << "  Medoid start node: " << start_node_ << std::endl;

    std::mt19937 rng(42);
    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    // Ensure medoid is processed first
    auto it = std::find(perm.begin(), perm.end(), start_node_);
    if (it != perm.end()) std::swap(perm[0], *it);

    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);
    Timer build_timer;

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];

        // Disable adaptive exit during build for graph quality
        auto [candidates, _dist_cmps] = greedy_search(get_vector(point), L, 0.0f);

        {
            // FIX: hold point's lock while writing graph_[point] via robust_prune
            std::lock_guard<std::mutex> lk(locks_[point]);
            robust_prune(point, candidates, alpha, R);
        }

        // Add backward edges
        // Take a snapshot of point's neighbors (under lock) before releasing
        std::vector<uint32_t> fwd_neighbors;
        {
            std::lock_guard<std::mutex> lk(locks_[point]);
            fwd_neighbors = graph_[point];
        }

        for (uint32_t nbr : fwd_neighbors) {
            std::lock_guard<std::mutex> lock(locks_[nbr]);
            graph_[nbr].push_back(point);

            if (graph_[nbr].size() > gamma_R) {
                // FIX: lock is already held for nbr here; robust_prune writes
                // graph_[nbr] directly — this is now safe because the lock is held.
                std::vector<Candidate> nbr_candidates;
                nbr_candidates.reserve(graph_[nbr].size());
                for (uint32_t nn : graph_[nbr]) {
                    float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                    nbr_candidates.push_back({d, nn});
                }
                robust_prune(nbr, nbr_candidates, alpha, R);
            }
        }

        if (idx % 10000 == 0) {
            #pragma omp critical
            { std::cout << "\r  Inserted " << idx << " / " << npts_ << std::flush; }
        }
    }
    std::cout << "\n  Build complete in " << build_timer.elapsed_seconds() << "s." << std::endl;
}

// ============================================================================
// Search
// ============================================================================

// FIX: signature now matches header (3 params + defaulted epsilon).
SearchResult VamanaIndex::search(const float* query, uint32_t K, uint32_t L,
                                 float epsilon) const {
    if (L < K) L = K;
    Timer t;
    auto [candidates, dist_cmps] = greedy_search(query, L, epsilon);
    double latency = t.elapsed_us();

    SearchResult result;
    result.dist_cmps  = dist_cmps;
    result.latency_us = latency;
    for (uint32_t i = 0; i < K && i < candidates.size(); i++) {
        result.ids.push_back(candidates[i].second);
    }
    return result;
}

// ============================================================================
// Helpers & IO
// ============================================================================

void VamanaIndex::init_scratchpads() {
    uint32_t max_threads = static_cast<uint32_t>(omp_get_max_threads());
    thread_scratchpads.clear();
    for (uint32_t i = 0; i < max_threads; i++) {
        thread_scratchpads.push_back(std::make_unique<Scratchpad>(npts_));
    }
}

VamanaIndex::Scratchpad& VamanaIndex::get_scratchpad() const {
    int tid = omp_get_thread_num();
    if (tid < 0 || tid >= static_cast<int>(thread_scratchpads.size()))
        return *thread_scratchpads[0];
    return *thread_scratchpads[tid];
}

uint32_t VamanaIndex::find_medoid() {
    std::vector<float> centroid(dim_, 0.0f);
    for (uint32_t i = 0; i < npts_; i++) {
        const float* v = get_vector(i);
        for (uint32_t d = 0; d < dim_; d++) centroid[d] += v[d];
    }
    for (uint32_t d = 0; d < dim_; d++) centroid[d] /= static_cast<float>(npts_);

    uint32_t medoid_id = 0;
    float min_dist = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i < npts_; i++) {
        float d = compute_l2sq(centroid.data(), get_vector(i), dim_);
        if (d < min_dist) { min_dist = d; medoid_id = i; }
    }
    return medoid_id;
}

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for writing: " + path);
    out.write(reinterpret_cast<const char*>(&npts_),       sizeof(npts_));
    out.write(reinterpret_cast<const char*>(&dim_),        sizeof(dim_));
    out.write(reinterpret_cast<const char*>(&start_node_), sizeof(start_node_));
    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg = static_cast<uint32_t>(graph_[i].size());
        out.write(reinterpret_cast<const char*>(&deg),           sizeof(deg));
        out.write(reinterpret_cast<const char*>(graph_[i].data()), deg * sizeof(uint32_t));
    }
}

void VamanaIndex::load(const std::string& index_path, const std::string& data_path) {
    FloatMatrix mat = load_fbin(data_path);
    npts_      = mat.npts;
    dim_       = mat.dims;
    data_      = mat.data.release();
    owns_data_ = true;
    init_scratchpads();

    std::ifstream in(index_path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open index file: " + index_path);

    uint32_t f_npts, f_dim;
    in.read(reinterpret_cast<char*>(&f_npts),       sizeof(f_npts));
    in.read(reinterpret_cast<char*>(&f_dim),         sizeof(f_dim));
    in.read(reinterpret_cast<char*>(&start_node_),   sizeof(start_node_));

    if (f_npts != npts_ || f_dim != dim_)
        throw std::runtime_error("Index/data file dimension mismatch");

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);
    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg;
        in.read(reinterpret_cast<char*>(&deg), sizeof(deg));
        graph_[i].resize(deg);
        in.read(reinterpret_cast<char*>(graph_[i].data()), deg * sizeof(uint32_t));
    }
}