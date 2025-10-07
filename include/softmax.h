#pragma once
#include <vector>

// Future declarations (AVX, GPU) will be added here.
void softmax_naive_cpu(std::vector<float>& vec);
void softmax_foolish_handcoding_cpu(std::vector<float>& vec);
void softmax_simd_cpu(std::vector<float>& vec);
void softmax_fused_simd_cpu(std::vector<float>& vec);

// OpenMP variants
void softmax_naive_omp(std::vector<float>& vec);
void softmax_simd_omp(std::vector<float>& vec);

// oneDNN optional implementation (compiled only if available)
void softmax_onednn_cpu(std::vector<float>& vec);
