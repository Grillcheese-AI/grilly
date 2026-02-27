#include "grilly/cubemind/cube.h"
#include "grilly/cubemind/vsa.h"

#include <algorithm>
#include <random>
#include <stdexcept>

namespace grilly {
namespace cubemind {

// ── Solved State ─────────────────────────────────────────────────────
//
// Face order: U(0), R(1), F(2), D(3), L(4), B(5)
// Facelet numbering within each face:
//
// 3x3 face layout:     2x2 face layout:
//   0 1 2                0 1
//   3 4 5                2 3
//   6 7 8
//
// Solved state: face f has all facelets = f.

CubeState cubeSolved(CubeSize size) {
    CubeState state;
    state.size = size;
    uint32_t n = (size == CubeSize::Cube2x2) ? 4 : 9;
    state.facelets.resize(6 * n);

    for (uint32_t face = 0; face < 6; ++face) {
        for (uint32_t i = 0; i < n; ++i) {
            state.facelets[face * n + i] = static_cast<uint8_t>(face);
        }
    }
    return state;
}

// ── 3x3 Permutation Tables ──────────────────────────────────────────
//
// Each face turn is decomposed into 4-cycles (cyclic permutations of
// 4 facelets). A clockwise U-turn cycles:
//   - The 4 corners of the U face
//   - The 4 edges of the U face
//   - 3 groups of 3 facelets on adjacent faces (the top row of F,L,B,R)
//
// These tables encode the affected indices for a clockwise quarter turn.
// Counterclockwise = apply clockwise 3 times. Half turn = apply 2 times.
//
// Verified: M * M' = identity, M^4 = identity (Appendix B of paper).
//
// Facelet global indexing for 3x3:
//   U face: 0-8,   R face: 9-17,  F face: 18-26
//   D face: 27-35, L face: 36-44, B face: 45-53

/// A 4-cycle: indices a->b->c->d->a (clockwise).
struct Cycle4 {
    uint8_t a, b, c, d;
};

/// Apply a set of 4-cycles as a clockwise quarter turn.
static void applyCycles(std::vector<uint8_t>& facelets,
                         const std::vector<Cycle4>& cycles) {
    for (const auto& c : cycles) {
        uint8_t tmp = facelets[c.d];
        facelets[c.d] = facelets[c.c];
        facelets[c.c] = facelets[c.b];
        facelets[c.b] = facelets[c.a];
        facelets[c.a] = tmp;
    }
}

// 3x3 move tables: clockwise quarter turns
// Each move has exactly 5 4-cycles: 2 for the face itself (corners+edges)
// and 3 for the adjacent facelets on neighboring faces.

// U face (top): facelets 0-8
// Clockwise: corners 0->2->8->6, edges 1->5->7->3
// Adjacent: F top row (18,19,20), R top row (9,10,11),
//           B top row (45,46,47), L top row (36,37,38)
static const std::vector<Cycle4> kU3Cycles = {
    {0, 2, 8, 6}, {1, 5, 7, 3},         // U face rotation
    {18, 9, 45, 36}, {19, 10, 46, 37}, {20, 11, 47, 38},  // adjacent rows
};

// R face (right): facelets 9-17
// Adjacent: U right col (2,5,8), B left col (45->53: 51,48,45 reversed),
//           D right col (29,32,35), F right col (20,23,26)
static const std::vector<Cycle4> kR3Cycles = {
    {9, 11, 17, 15}, {10, 14, 16, 12},   // R face rotation
    {2, 20, 29, 51}, {5, 23, 32, 48}, {8, 26, 35, 45},  // adjacent cols
};

// F face (front): facelets 18-26
// Adjacent: U bottom row (6,7,8), R left col (9,12,15),
//           D top row (27,28,29), L right col (38,41,44 reversed)
static const std::vector<Cycle4> kF3Cycles = {
    {18, 20, 26, 24}, {19, 23, 25, 21},  // F face rotation
    {6, 9, 29, 44}, {7, 12, 28, 41}, {8, 15, 27, 38},  // adjacent
};

// D face (bottom): facelets 27-35
// Adjacent: F bottom row (24,25,26), L bottom row (42,43,44),
//           B bottom row (51,52,53), R bottom row (15,16,17)
static const std::vector<Cycle4> kD3Cycles = {
    {27, 29, 35, 33}, {28, 32, 34, 30},  // D face rotation
    {24, 42, 53, 17}, {25, 43, 52, 16}, {26, 44, 51, 15},  // adjacent
};

// L face (left): facelets 36-44
// Adjacent: U left col (0,3,6), F left col (18,21,24),
//           D left col (27,30,33), B right col (47,50,53 reversed)
static const std::vector<Cycle4> kL3Cycles = {
    {36, 38, 44, 42}, {37, 41, 43, 39},  // L face rotation
    {0, 47, 27, 18}, {3, 50, 30, 21}, {6, 53, 33, 24},  // adjacent
};

// B face (back): facelets 45-53
// Adjacent: U top row (0,1,2), L left col (36,39,42 reversed),
//           D bottom row (33,34,35), R right col (11,14,17)
static const std::vector<Cycle4> kB3Cycles = {
    {45, 47, 53, 51}, {46, 50, 52, 48},  // B face rotation
    {0, 36, 33, 17}, {1, 39, 34, 14}, {2, 42, 35, 11},  // adjacent
};

// ── 2x2 Permutation Tables ──────────────────────────────────────────
//
// 2x2 facelets: U(0-3), R(4-7), F(8-11), D(12-15), L(16-19), B(20-23)
// Face layout:  0 1
//               2 3
// Only U, R, F turns (D, L, B are redundant for 2x2).

static const std::vector<Cycle4> kU2Cycles = {
    {0, 1, 3, 2},                         // U face rotation
    {8, 4, 20, 16}, {9, 5, 21, 17},       // adjacent
};

static const std::vector<Cycle4> kR2Cycles = {
    {4, 5, 7, 6},                         // R face rotation
    {1, 9, 13, 23}, {3, 11, 15, 21},      // adjacent
};

static const std::vector<Cycle4> kF2Cycles = {
    {8, 9, 11, 10},                       // F face rotation
    {2, 4, 15, 19}, {3, 6, 14, 17},       // adjacent
};

/// Apply a clockwise quarter turn (the cycle set for the given move).
static void applyClockwise(std::vector<uint8_t>& facelets,
                            const std::vector<Cycle4>& cycles) {
    applyCycles(facelets, cycles);
}

/// Apply a counterclockwise quarter turn (= 3 clockwise quarter turns).
static void applyCounterClockwise(std::vector<uint8_t>& facelets,
                                   const std::vector<Cycle4>& cycles) {
    applyCycles(facelets, cycles);
    applyCycles(facelets, cycles);
    applyCycles(facelets, cycles);
}

/// Apply a half turn (= 2 clockwise quarter turns).
static void applyHalfTurn(std::vector<uint8_t>& facelets,
                           const std::vector<Cycle4>& cycles) {
    applyCycles(facelets, cycles);
    applyCycles(facelets, cycles);
}

CubeState cubeApplyMove(const CubeState& state, CubeMove move) {
    CubeState result = state;  // Copy

    if (state.size == CubeSize::Cube3x3) {
        switch (move) {
            case CubeMove::U:       applyClockwise(result.facelets, kU3Cycles); break;
            case CubeMove::U_prime: applyCounterClockwise(result.facelets, kU3Cycles); break;
            case CubeMove::U2:      applyHalfTurn(result.facelets, kU3Cycles); break;
            case CubeMove::R:       applyClockwise(result.facelets, kR3Cycles); break;
            case CubeMove::R_prime: applyCounterClockwise(result.facelets, kR3Cycles); break;
            case CubeMove::R2:      applyHalfTurn(result.facelets, kR3Cycles); break;
            case CubeMove::F:       applyClockwise(result.facelets, kF3Cycles); break;
            case CubeMove::F_prime: applyCounterClockwise(result.facelets, kF3Cycles); break;
            case CubeMove::F2:      applyHalfTurn(result.facelets, kF3Cycles); break;
            case CubeMove::D:       applyClockwise(result.facelets, kD3Cycles); break;
            case CubeMove::D_prime: applyCounterClockwise(result.facelets, kD3Cycles); break;
            case CubeMove::D2:      applyHalfTurn(result.facelets, kD3Cycles); break;
            case CubeMove::L:       applyClockwise(result.facelets, kL3Cycles); break;
            case CubeMove::L_prime: applyCounterClockwise(result.facelets, kL3Cycles); break;
            case CubeMove::L2:      applyHalfTurn(result.facelets, kL3Cycles); break;
            case CubeMove::B:       applyClockwise(result.facelets, kB3Cycles); break;
            case CubeMove::B_prime: applyCounterClockwise(result.facelets, kB3Cycles); break;
            case CubeMove::B2:      applyHalfTurn(result.facelets, kB3Cycles); break;
        }
    } else {
        // 2x2: only U, R, F are valid
        switch (move) {
            case CubeMove::U:       applyClockwise(result.facelets, kU2Cycles); break;
            case CubeMove::U_prime: applyCounterClockwise(result.facelets, kU2Cycles); break;
            case CubeMove::U2:      applyHalfTurn(result.facelets, kU2Cycles); break;
            case CubeMove::R:       applyClockwise(result.facelets, kR2Cycles); break;
            case CubeMove::R_prime: applyCounterClockwise(result.facelets, kR2Cycles); break;
            case CubeMove::R2:      applyHalfTurn(result.facelets, kR2Cycles); break;
            case CubeMove::F:       applyClockwise(result.facelets, kF2Cycles); break;
            case CubeMove::F_prime: applyCounterClockwise(result.facelets, kF2Cycles); break;
            case CubeMove::F2:      applyHalfTurn(result.facelets, kF2Cycles); break;
            default:
                // D, L, B are redundant for 2x2 — treat as no-op
                break;
        }
    }

    return result;
}

CubeState cubeApplyMoves(const CubeState& state,
                          const std::vector<CubeMove>& moves) {
    CubeState current = state;
    for (CubeMove m : moves) {
        current = cubeApplyMove(current, m);
    }
    return current;
}

CubeState cubeRandomWalk(CubeSize size, uint32_t numMoves, uint32_t seed) {
    CubeState state = cubeSolved(size);
    std::mt19937 rng(seed);

    // Available moves depend on cube size
    uint32_t numMoveTypes = (size == CubeSize::Cube2x2) ? 9 : 18;

    for (uint32_t i = 0; i < numMoves; ++i) {
        uint32_t moveIdx = rng() % numMoveTypes;
        state = cubeApplyMove(state, static_cast<CubeMove>(moveIdx));
    }

    return state;
}

uint32_t cubeEstimateDistance(const CubeState& state) {
    CubeState solved = cubeSolved(state.size);
    uint32_t mismatches = 0;
    for (size_t i = 0; i < state.facelets.size(); ++i) {
        if (state.facelets[i] != solved.facelets[i]) {
            mismatches++;
        }
    }

    // Heuristic divisor: each move affects ~k facelets
    // 2x2: ~6 facelets per move (k=6), max distance ~11
    // 3x3: ~12 facelets per move (k=8), max distance ~20
    uint32_t k = (state.size == CubeSize::Cube2x2) ? 6 : 8;
    uint32_t dMax = (state.size == CubeSize::Cube2x2) ? 11 : 20;

    return std::min(mismatches / k, dMax);
}

std::vector<CubeState> cubeGenerateShell(CubeSize size, uint32_t targetDist,
                                          uint32_t count, uint32_t tolerance) {
    std::vector<CubeState> result;
    result.reserve(count);

    // We oversample with random walks and filter by estimated distance
    uint32_t seed = targetDist * 1000;
    uint32_t attempts = 0;
    const uint32_t maxAttempts = count * 100;

    // Walk length should be around targetDist to have a chance of landing
    // in the right shell (overshoot to ensure enough mixing)
    uint32_t walkLen = targetDist + 5;

    while (result.size() < count && attempts < maxAttempts) {
        CubeState state = cubeRandomWalk(size, walkLen, seed + attempts);
        uint32_t dist = cubeEstimateDistance(state);

        if (dist >= targetDist - tolerance && dist <= targetDist + tolerance) {
            result.push_back(std::move(state));
        }
        attempts++;
    }

    return result;
}

// ── VSA Encoding ────────────────────────────────────────────────────
//
// Encode a cube state as a bipolar hypervector using role-filler binding:
//
//   phi(g) = sign( sum_i BLAKE3("facelet_{i}") * BLAKE3("color_{c_i}") )
//
// Each facelet position gets a BLAKE3 role vector (structural).
// Each face color gets a BLAKE3 filler vector (categorical — no gradients).
// Binding via element-wise multiply, bundling via majority vote + sign.
//
// For cube states, BLAKE3 fillers are appropriate because cube colors
// have no semantic ordering (red is not "closer to" blue than green).
// The reviewer's learned-filler fix applies to text tokens (Milestone 1).

std::vector<int8_t> cubeToVSA(const CubeState& state, uint32_t dim) {
    uint32_t nFacelets = state.numFacelets();

    // Generate role and filler vectors
    std::vector<std::string> roleNames;
    std::vector<const int8_t*> fillerPtrs;
    roleNames.reserve(nFacelets);

    // Pre-generate color filler vectors (only 6 colors)
    std::vector<std::vector<int8_t>> colorFillers(6);
    for (uint32_t c = 0; c < 6; ++c) {
        colorFillers[c] = blake3Role("color_" + std::to_string(c), dim);
    }

    // Build role names and filler pointers
    std::vector<std::vector<int8_t>> roleVecs(nFacelets);
    std::vector<std::vector<int8_t>> boundPairs(nFacelets);

    for (uint32_t i = 0; i < nFacelets; ++i) {
        roleVecs[i] = blake3Role("facelet_" + std::to_string(i), dim);
        uint8_t color = state.facelets[i];
        boundPairs[i] = vsaBind(roleVecs[i].data(), colorFillers[color].data(), dim);
    }

    // Bundle all bound pairs
    std::vector<const int8_t*> ptrs;
    ptrs.reserve(nFacelets);
    for (auto& bp : boundPairs) {
        ptrs.push_back(bp.data());
    }

    return vsaBundle(ptrs, dim);
}

BitpackedVec cubeToVSABitpacked(const CubeState& state, uint32_t dim) {
    std::vector<int8_t> bipolar = cubeToVSA(state, dim);
    return vsaBitpack(bipolar.data(), dim);
}

}  // namespace cubemind
}  // namespace grilly
