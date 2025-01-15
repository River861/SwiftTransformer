#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace st::kernel {

void flashAttentionContextStageAttention(
    half* __restrict__ result,     // [num_tokens, local_q_head_num, head_dim]
    const half* __restrict__ qkvs, // [num_tokens, local_q_head_num+2*local_kv_head_num, head_dim]
    const int32_t* __restrict__ ith_context_req_token_index, // [num_context_reqs+1]
    const float qk_scale,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int num_context_reqs,
    const int max_context_req_len,
    const int num_context_stage_tokens
);

}