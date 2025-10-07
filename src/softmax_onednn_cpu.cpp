#include "softmax.h"
#include <vector>
#include "dnnl.hpp"

void softmax_onednn_cpu(std::vector<float>& vec) {
    if (vec.empty()) return;

    using namespace dnnl;

    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    const long long vector_size = static_cast<long long>(vec.size());
    memory::dims softmax_dims = {1, vector_size};
    auto softmax_md = memory::desc(softmax_dims, memory::data_type::f32, memory::format_tag::nc);

    auto data_mem = memory(softmax_md, eng, vec.data());

    auto softmax_pd = softmax_forward::primitive_desc(
        eng,
        prop_kind::forward_inference,
        algorithm::softmax_accurate,
        softmax_md,
        softmax_md,
        1
    );

    auto softmax_prim = softmax_forward(softmax_pd);

    softmax_prim.execute(s, {{DNNL_ARG_SRC, data_mem}, {DNNL_ARG_DST, data_mem}});
    s.wait();
}


