#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "grilly/cubemind/types.h"

namespace grilly {
namespace cubemind {

// ── Rubik's Cube State Generator ──────────────────────────────────────
//
// Encodes Rubik's cube states as VSA hypervectors for the CubeMind
// cache. The cube provides a structured domain with known distances
// and verifiable geometric properties.
//
// Key properties verified in the paper (Appendix B):
//   - M * M' = identity (move followed by its inverse restores state)
//   - M^4 = identity (any face turn applied 4 times restores state)
//   - 4-cycle permutation tables: each move swaps exactly 4 groups of
//     4 facelets in cyclic order
//
// VSA encoding: phi(g) = sign(sum_i role_i * filler_i)
//   where role_i = BLAKE3("facelet_{i}") — structural position
//   and   filler_i = BLAKE3("color_{c}") — face color
//   (For cube states, BLAKE3 fillers are fine because cube colors
//   have no semantic gradients — red is not "more similar" to blue.)

/// Cube sizes supported.
enum class CubeSize : uint32_t { Cube2x2 = 2, Cube3x3 = 3 };

/// A cube state is an array of facelet colors.
/// 2x2: 24 facelets (6 faces x 4). 3x3: 54 facelets (6 faces x 9).
struct CubeState {
    std::vector<uint8_t> facelets;  // Color per facelet (0-5)
    CubeSize size;

    uint32_t numFacelets() const {
        return (size == CubeSize::Cube2x2) ? 24 : 54;
    }
};

/// Available moves (face turns).
/// Standard Singmaster notation: U=Up, R=Right, F=Front, D=Down, L=Left, B=Back.
/// _prime = counter-clockwise, _2 = half turn (180 degrees).
enum class CubeMove : uint8_t {
    U, U_prime, U2,
    R, R_prime, R2,
    F, F_prime, F2,
    D, D_prime, D2,  // 3x3 only
    L, L_prime, L2,  // 3x3 only
    B, B_prime, B2,  // 3x3 only
};

/// Create the solved state for a given cube size.
/// Face order: U(0), R(1), F(2), D(3), L(4), B(5).
CubeState cubeSolved(CubeSize size);

/// Apply a single move to a cube state (returns new state).
/// Uses verified 4-cycle permutation tables.
CubeState cubeApplyMove(const CubeState& state, CubeMove move);

/// Apply a sequence of moves.
CubeState cubeApplyMoves(const CubeState& state,
                          const std::vector<CubeMove>& moves);

/// Random walk from solved state to generate a scrambled state.
/// @param numMoves Number of random moves to apply
/// @param seed     Random seed for reproducibility
CubeState cubeRandomWalk(CubeSize size, uint32_t numMoves, uint32_t seed = 0);

/// Estimate distance from solved via facelet mismatch heuristic.
/// d_est = min(mismatches / k, d_max) where k depends on cube size.
uint32_t cubeEstimateDistance(const CubeState& state);

/// Generate N states at a target estimated distance (distance-shell sampling).
/// Performs random walks and filters states within tolerance of targetDist.
std::vector<CubeState> cubeGenerateShell(CubeSize size, uint32_t targetDist,
                                          uint32_t count, uint32_t tolerance = 1);

/// Encode a cube state as a bipolar VSA hypervector.
/// phi(g) = sign(sum_i BLAKE3("facelet_i") * BLAKE3("color_c_i"))
std::vector<int8_t> cubeToVSA(const CubeState& state, uint32_t dim);

/// Encode and bitpack in one step.
BitpackedVec cubeToVSABitpacked(const CubeState& state, uint32_t dim);

}  // namespace cubemind
}  // namespace grilly
