// Snippet: Grid cell hexagonal 3-wave activation
// Input: position (rx, ry shifted), spacing, wave vector k
// Output: firing rate [0, max_rate]
float op_grid_cell(float rx, float ry, float k, float max_rate) {
    float u1 = k * rx;
    float u2 = k * (-0.5 * rx + 0.866025 * ry);
    float u3 = k * (-0.5 * rx - 0.866025 * ry);
    float val = (cos(u1) + cos(u2) + cos(u3)) / 3.0 + 0.5;
    return max_rate * max(val, 0.0);
}
