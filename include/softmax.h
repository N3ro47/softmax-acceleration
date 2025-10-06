#pragma once
#include <vector>

// Future declarations (AVX, GPU) will be added here.
void softmax_naive_cpu(std::vector<float>& vec);
void softmax_foolish_handcoding_cpu(std::vector<float>& vec);
void softmax_simd_cpu(std::vector<float>& vec);
void softmax_fused_simd_cpu(std::vector<float>& vec);
