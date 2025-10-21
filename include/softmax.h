#pragma once
#include <vector>

void softmax_naive_cpu(std::vector<float>& vec);
void softmax_foolish_handcoding_cpu(std::vector<float>& vec);
void softmax_simd_cpu(std::vector<float>& vec);
void softmax_fused_simd_cpu(std::vector<float>& vec);

void softmax_gpu(std::vector<float>& vec);
void softmax_gpu_thrust(std::vector<float>& vec);
void softmax_gpu_cub_fused(std::vector<float>& vec);

// OpenMP variants
void softmax_naive_omp(std::vector<float>& vec);
void softmax_simd_omp(std::vector<float>& vec);

// oneDNN optional implementation
void softmax_onednn_cpu(std::vector<float>& vec);
