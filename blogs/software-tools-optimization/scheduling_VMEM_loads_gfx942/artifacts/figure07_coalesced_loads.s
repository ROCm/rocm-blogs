; Same inner-loop instruction schedule and dest VGPRs but address VGPRs use 
; a lane-linear coalesced pattern (like global_load_coalesce.s). 
;
; Offsets: v147 = (tid & 0xFF) * 16; v148..v154 add 0x200..0xE00 (512B steps) so
; max byte (lane 255) + 16 <= 8320 = K * sizeof(fp16) for driver’s K=4160.
;
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 4
	.text
	.globl	matmul_kernel
	.p2align	8
	.type	matmul_kernel,@function
matmul_kernel:
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx2 s[12:13], s[0:1], 0x28
	s_load_dword s14, s[0:1], 0x30
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v1, 0xff, v0
	v_lshlrev_b32_e32 v147, 4, v1
	v_add_u32_e32 v148, 0x200, v147
	v_add_u32_e32 v149, 0x400, v147
	v_add_u32_e32 v150, 0x600, v147
	v_add_u32_e32 v151, 0x800, v147
	v_add_u32_e32 v152, 0xa00, v147
	v_add_u32_e32 v153, 0xc00, v147
	v_add_u32_e32 v154, 0xe00, v147
	v_mov_b32_e32 v155, v150
	v_mov_b32_e32 v156, v149
	v_mov_b32_e32 v157, v152
	v_mov_b32_e32 v158, v151
	v_mov_b32_e32 v159, v154
	v_mov_b32_e32 v160, v153
	v_mov_b32_e32 v161, v147
	v_mov_b32_e32 v162, v148
	s_and_b32 s17, s3, 0xffff
	s_mov_b32 s16, s2
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s4, s4
	s_add_i32 s10, s10, 63
	s_lshr_b32 s12, s10, 6
	s_cmp_eq_u32 s12, 0
	s_cbranch_scc1 .LBB0_6
	s_add_i32 s12, s12, -2
	s_cmp_lt_i32 s12, 0
	s_cbranch_scc1 .LBB0_6
.LBB0_5:
	s_barrier
	s_waitcnt vmcnt(0)
	global_load_dwordx4 v[62:65], v150, s[16:17]
	global_load_dwordx4 v[50:53], v149, s[16:17]
	global_load_dwordx4 v[34:37], v152, s[16:17]
	global_load_dwordx4 v[46:49], v151, s[16:17]
	global_load_dwordx4 v[30:33], v154, s[16:17]
	global_load_dwordx4 v[42:45], v153, s[16:17]
	global_load_dwordx4 v[58:61], v147, s[16:17]
	global_load_dwordx4 v[54:57], v148, s[16:17]
	global_load_dwordx4 v[38:41], v155, s[4:5]
	global_load_dwordx4 v[26:29], v156, s[4:5]
	global_load_dwordx4 v[22:25], v157, s[4:5]
	global_load_dwordx4 v[18:21], v158, s[4:5]
	global_load_dwordx4 v[14:17], v159, s[4:5]
	global_load_dwordx4 v[10:13], v160, s[4:5]
	global_load_dwordx4 v[6:9], v161, s[4:5]
	global_load_dwordx4 v[2:5], v162, s[4:5]
	s_barrier
	s_add_u32 s4, s4, 0x80
	s_addc_u32 s5, s5, 0
	s_add_u32 s16, s16, 0x80
	s_addc_u32 s17, s17, 0
	s_add_i32 s12, s12, -1
	s_cmp_lg_u32 s12, 0
	s_cbranch_scc1 .LBB0_5
.LBB0_6:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 16
		.amdhsa_kernarg_size 64
		.amdhsa_user_sgpr_count 15
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 13
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 512
		.amdhsa_next_free_sgpr 29
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	matmul_kernel, .Lfunc_end0-matmul_kernel
	.cfi_endproc
	.set matmul_kernel.num_vgpr, 256
	.set matmul_kernel.num_agpr, 256
	.set matmul_kernel.numbered_sgpr, 29
	.set matmul_kernel.private_seg_size, 16
	.set matmul_kernel.uses_vcc, 1
	.set matmul_kernel.uses_flat_scratch, 0
	.set matmul_kernel.has_dyn_sized_stack, 0
	.set matmul_kernel.has_recursion, 0
	.set matmul_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     256
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .max_flat_workgroup_size: 256
    .name:           matmul_kernel
    .private_segment_fixed_size: 16
    .sgpr_count:     35
    .sgpr_spill_count: 0
    .symbol:         matmul_kernel.kd
    .vgpr_count:     512
    .vgpr_spill_count: 5
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
