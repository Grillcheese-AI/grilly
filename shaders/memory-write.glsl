#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// New key to write (key_dim)
layout(set = 0, binding = 0) readonly buffer NewKey {
    float new_key[];
};

// New value to write (value_dim)
layout(set = 0, binding = 1) readonly buffer NewValue {
    float new_value[];
};

// Memory keys (num_memories, key_dim)
layout(set = 0, binding = 2) buffer MemoryKeys {
    float keys[];
};

// Memory values (num_memories, value_dim)
layout(set = 0, binding = 3) buffer MemoryValues {
    float values[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint num_memories;
    uint key_dim;
    uint value_dim;
    uint write_index;       // Index to write to (round-robin or LRU)
    uint write_mode;        // 0 = overwrite, 1 = blend with old
    float blend_factor;     // For blend mode: new_val = α*new + (1-α)*old
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    // Write key
    if (gID < key_dim) {
        uint mem_key_idx = write_index * key_dim + gID;
        
        if (write_mode == 0) {
            // Overwrite mode
            keys[mem_key_idx] = new_key[gID];
        } else {
            // Blend mode
            keys[mem_key_idx] = blend_factor * new_key[gID] + 
                               (1.0 - blend_factor) * keys[mem_key_idx];
        }
    }
    
    // Write value
    if (gID < value_dim) {
        uint mem_val_idx = write_index * value_dim + gID;
        
        if (write_mode == 0) {
            // Overwrite mode
            values[mem_val_idx] = new_value[gID];
        } else {
            // Blend mode
            values[mem_val_idx] = blend_factor * new_value[gID] + 
                                 (1.0 - blend_factor) * values[mem_val_idx];
        }
    }
}
