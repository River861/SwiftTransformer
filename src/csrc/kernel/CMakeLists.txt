add_library(kernel STATIC
	addbias.cu
	count_nan.cu
	embedding.cu
	findmax.cu
	fused_activ_multiply.cu
	fused_addbias_activ.cu
	fused_context_stage_attention.cu
	fused_decoding_stage_attention.cu
	fused_decoding_stage_attention_mha.cu
	gather_last_tokens.cu
	kvcache_mgmt.cu
	layernorm.cu
	rmsnorm.cu
	rotary_posi_embedding.cu
	softmax.cu
	unfused_attention.cu
)
target_link_libraries(kernel PUBLIC util)
set_property(TARGET kernel PROPERTY POSITION_INDEPENDENT_CODE ON)

file(GLOB flash_attn_impl_cu_files ${CMAKE_CURRENT_SOURCE_DIR}/flash-attention/csrc/flash_attn/src/flash_fwd_*_fp16_causal_*.cu)
message("flash_attn_impl_cu_files=${flash_attn_impl_cu_files}")


add_library(flash_attn_impl STATIC ${flash_attn_impl_cu_files})
target_include_directories(flash_attn_impl PUBLIC flash-attention/csrc/cutlass/include)
set_property(TARGET flash_attn_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(flash_attn_context_stage_attention STATIC
	flash_attn_context_stage_attention.cu
)
target_include_directories(flash_attn_context_stage_attention PRIVATE flash-attention/csrc/cutlass/include)
target_include_directories(flash_attn_context_stage_attention PRIVATE flash-attention/csrc/flash_attn/src)
target_link_libraries(flash_attn_context_stage_attention PUBLIC util flash_attn_impl)
set_property(TARGET flash_attn_context_stage_attention PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(kernel PUBLIC flash_attn_context_stage_attention)