	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 4
	.text
	.globl	matmul_kernel                   ; -- Begin function matmul_kernel
	.p2align	8
	.type	matmul_kernel,@function
matmul_kernel:                          ; @matmul_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.7:
	.file	1 "<unknown>"
	.loc	1 0 0 prologue_end              ; <unknown>:0:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx2 s[12:13], s[0:1], 0x28
	s_load_dword s14, s[0:1], 0x30
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.8:
.LBB0_0:
	s_add_i32 s1, s9, 0xff
	s_ashr_i32 s9, s1, 31
	s_lshr_b32 s9, s9, 24
	s_add_i32 s1, s1, s9
	s_ashr_i32 s9, s1, 8
	s_lshl_b32 s9, s9, 2
	s_abs_i32 s14, s9
	v_cvt_f32_u32_e32 v1, s14
	s_ashr_i32 s0, s15, 31
	s_lshr_b32 s0, s0, 29
	s_add_i32 s0, s15, s0
	v_rcp_iflag_f32_e32 v1, v1
	s_ashr_i32 s0, s0, 3
	s_mulk_i32 s15, 0x4c
	s_mulk_i32 s0, 0xfda1
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_add_i32 s0, s0, s15
	s_ashr_i32 s15, s0, 31
	s_ashr_i32 s1, s1, 31
	s_add_i32 s16, s0, s15
	s_xor_b32 s1, s15, s1
	s_xor_b32 s15, s16, s15
	v_readfirstlane_b32 s16, v1
	s_sub_i32 s17, 0, s14
	s_mul_i32 s17, s17, s16
	s_mul_hi_u32 s17, s16, s17
	s_add_i32 s16, s16, s17
	s_mul_hi_u32 s16, s15, s16
	s_mul_i32 s17, s16, s14
	s_sub_i32 s15, s15, s17
	s_add_i32 s17, s16, 1
	s_sub_i32 s18, s15, s14
	s_cmp_ge_u32 s15, s14
	s_cselect_b32 s16, s17, s16
	s_cselect_b32 s15, s18, s15
	s_add_i32 s17, s16, 1
	s_cmp_ge_u32 s15, s14
	s_cselect_b32 s14, s17, s16
	s_addk_i32 s8, 0xff
	s_ashr_i32 s15, s8, 31
	s_xor_b32 s14, s14, s1
	s_lshr_b32 s15, s15, 24
	s_sub_i32 s1, s14, s1
	s_add_i32 s8, s8, s15
	s_lshl_b32 s14, s1, 2
	s_ashr_i32 s8, s8, 8
	s_sub_i32 s8, s8, s14
	s_min_i32 s8, s8, 4
	s_abs_i32 s15, s8
	v_cvt_f32_u32_e32 v1, s15
	s_sub_i32 s16, 0, s15
	s_mul_i32 s1, s1, s9
	s_sub_i32 s0, s0, s1
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s9, s0
	s_xor_b32 s1, s0, s8
	s_ashr_i32 s1, s1, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_lshlrev_b32_e32 v122, 3, v0
	v_and_b32_e32 v2, 56, v122
	v_bfrev_b32_e32 v6, 1
	v_readfirstlane_b32 s17, v1
	s_mul_i32 s16, s16, s17
	s_mul_hi_u32 s16, s17, s16
	s_add_i32 s17, s17, s16
	s_mul_hi_u32 s16, s9, s17
	s_mul_i32 s17, s16, s15
	s_sub_i32 s9, s9, s17
	s_add_i32 s17, s16, 1
	s_sub_i32 s18, s9, s15
	s_cmp_ge_u32 s9, s15
	s_cselect_b32 s16, s17, s16
	s_cselect_b32 s9, s18, s9
	s_add_i32 s17, s16, 1
	s_cmp_ge_u32 s9, s15
	s_cselect_b32 s9, s17, s16
	s_xor_b32 s9, s9, s1
	s_sub_i32 s15, s9, s1
	s_mul_i32 s8, s8, s15
	s_sub_i32 s28, s0, s8
	s_add_i32 s28, s28, s14
	s_lshl_b32 s9, s28, 8
	s_mul_i32 s0, s9, s11
	s_ashr_i32 s1, s0, 31
	s_lshl_b32 s8, s11, 5
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s16, s2, s0
	s_addc_u32 s14, s3, s1
	v_lshrrev_b32_e32 v1, 3, v0
	s_add_u32 s20, s16, 0x80
	v_or_b32_e32 v3, 0x60, v1
	s_addc_u32 s21, s14, 0
	s_add_i32 s10, s10, 63
	v_mad_u64_u32 v[4:5], s[0:1], s11, v1, v[2:3]
	s_cmp_gt_i32 s10, 63
	v_lshlrev_b32_e32 v147, 1, v4
	s_cselect_b64 s[0:1], -1, 0
	v_add_u32_e32 v4, s8, v4
	s_and_b32 s17, s14, 0xffff
	v_cndmask_b32_e64 v5, v6, v147, s[0:1]
	v_lshlrev_b32_e32 v148, 1, v4
	v_add_u32_e32 v7, s8, v4
	v_cndmask_b32_e64 v4, v6, v148, s[0:1]
	global_load_dwordx4 v[66:69], v5, s[16:17]
	global_load_dwordx4 v[70:73], v4, s[16:17]
	v_mul_lo_u32 v5, s11, v3
	v_lshlrev_b32_e32 v149, 1, v7
	v_lshl_add_u32 v7, s11, 6, v7
	v_add_lshl_u32 v150, v5, v2, 1
	v_cndmask_b32_e64 v4, v6, v149, s[0:1]
	v_cndmask_b32_e64 v5, v6, v150, s[0:1]
	v_lshlrev_b32_e32 v151, 1, v7
	v_add_u32_e32 v7, s8, v7
	s_cmpk_gt_i32 s10, 0x7f
	global_load_dwordx4 v[74:77], v4, s[16:17]
	global_load_dwordx4 v[78:81], v5, s[16:17]
	v_or_b32_e32 v5, 0xe0, v1
	v_add_lshl_u32 v153, v7, s8, 1
	s_cselect_b64 vcc, -1, 0
	s_lshl_b32 s8, s15, 8
	v_mul_lo_u32 v8, s11, v5
	s_mul_i32 s14, s8, s12
	v_cndmask_b32_e64 v4, v6, v151, s[0:1]
	v_lshlrev_b32_e32 v152, 1, v7
	v_add_lshl_u32 v154, v8, v2, 1
	s_ashr_i32 s15, s14, 31
	v_cndmask_b32_e64 v9, v6, v152, s[0:1]
	global_load_dwordx4 v[82:85], v4, s[16:17]
	global_load_dwordx4 v[86:89], v9, s[16:17]
	v_cndmask_b32_e64 v4, v6, v153, s[0:1]
	v_cndmask_b32_e64 v7, v6, v154, s[0:1]
	s_and_b32 s21, s21, 0xffff
	s_lshl_b32 s26, s12, 5
	v_mul_lo_u32 v3, s12, v3
	s_lshl_b64 s[14:15], s[14:15], 1
	global_load_dwordx4 v[90:93], v4, s[16:17]
	global_load_dwordx4 v[94:97], v7, s[16:17]
	v_mul_lo_u32 v7, s12, v5
	s_add_u32 s16, s4, s14
	v_mad_u64_u32 v[4:5], s[24:25], s12, v1, v[2:3]
	s_addc_u32 s17, s5, s15
	v_add_u32_e32 v5, s26, v4
	v_add_u32_e32 v8, s26, v5
	s_add_u32 s24, s16, 0x80
	v_lshlrev_b32_e32 v155, 1, v4
	v_lshl_add_u32 v9, s12, 6, v8
	s_addc_u32 s12, s17, 0
	s_and_b32 s17, s17, 0xffff
	v_cndmask_b32_e64 v4, v6, v155, s[0:1]
	v_lshlrev_b32_e32 v156, 1, v5
	v_lshlrev_b32_e32 v157, 1, v8
	v_add_lshl_u32 v158, v3, v2, 1
	v_add_u32_e32 v10, s26, v9
	v_cndmask_b32_e64 v5, v6, v156, s[0:1]
	global_load_dwordx4 v[98:101], v4, s[16:17]
	global_load_dwordx4 v[102:105], v5, s[16:17]
	v_cndmask_b32_e64 v4, v6, v157, s[0:1]
	v_cndmask_b32_e64 v3, v6, v158, s[0:1]
	v_lshlrev_b32_e32 v159, 1, v9
	global_load_dwordx4 v[106:109], v4, s[16:17]
	global_load_dwordx4 v[110:113], v3, s[16:17]
	v_cndmask_b32_e64 v3, v6, v159, s[0:1]
	v_lshlrev_b32_e32 v160, 1, v10
	v_add_lshl_u32 v161, v10, s26, 1
	v_cndmask_b32_e64 v4, v6, v160, s[0:1]
	global_load_dwordx4 v[114:117], v3, s[16:17]
	global_load_dwordx4 v[118:121], v4, s[16:17]
	v_cndmask_b32_e64 v3, v6, v161, s[0:1]
	v_add_lshl_u32 v162, v7, v2, 1
	v_cndmask_b32_e64 v2, v6, v162, s[0:1]
	global_load_dwordx4 v[126:129], v3, s[16:17]
	global_load_dwordx4 v[130:133], v2, s[16:17]
	v_cndmask_b32_e32 v2, v6, v147, vcc
	v_cndmask_b32_e32 v3, v6, v148, vcc
	global_load_dwordx4 v[58:61], v2, s[20:21]
	global_load_dwordx4 v[54:57], v3, s[20:21]
	v_cndmask_b32_e32 v2, v6, v149, vcc
	v_cndmask_b32_e32 v3, v6, v150, vcc
	global_load_dwordx4 v[50:53], v2, s[20:21]
	global_load_dwordx4 v[62:65], v3, s[20:21]
	v_cndmask_b32_e32 v2, v6, v151, vcc
	v_cndmask_b32_e32 v3, v6, v152, vcc
	global_load_dwordx4 v[46:49], v2, s[20:21]
	global_load_dwordx4 v[34:37], v3, s[20:21]
	v_cndmask_b32_e32 v2, v6, v153, vcc
	v_cndmask_b32_e32 v3, v6, v154, vcc
	global_load_dwordx4 v[42:45], v2, s[20:21]
	global_load_dwordx4 v[30:33], v3, s[20:21]
	s_and_b32 s25, s12, 0xffff
	v_cndmask_b32_e32 v2, v6, v155, vcc
	v_cndmask_b32_e32 v3, v6, v156, vcc
	global_load_dwordx4 v[38:41], v2, s[24:25]
	global_load_dwordx4 v[26:29], v3, s[24:25]
	v_cndmask_b32_e32 v2, v6, v157, vcc
	v_cndmask_b32_e32 v3, v6, v158, vcc
	global_load_dwordx4 v[22:25], v2, s[24:25]
	global_load_dwordx4 v[18:21], v3, s[24:25]
	v_cndmask_b32_e32 v2, v6, v159, vcc
	v_cndmask_b32_e32 v3, v6, v160, vcc
	global_load_dwordx4 v[14:17], v2, s[24:25]
	global_load_dwordx4 v[10:13], v3, s[24:25]
	v_cndmask_b32_e32 v2, v6, v161, vcc
	v_cndmask_b32_e32 v3, v6, v162, vcc
	global_load_dwordx4 v[6:9], v2, s[24:25]
	global_load_dwordx4 v[2:5], v3, s[24:25]
	v_mov_b32_e32 v200, 0x100
	v_add_u32_e32 v201, v200, v200
	v_add_u32_e32 v202, v201, v200
	v_add_u32_e32 v163, v147, v200
	v_add_u32_e32 v164, v148, v200
	v_add_u32_e32 v165, v149, v200
	v_add_u32_e32 v166, v150, v200
	v_add_u32_e32 v167, v151, v200
	v_add_u32_e32 v168, v152, v200
	v_add_u32_e32 v169, v153, v200
	v_add_u32_e32 v170, v154, v200
	v_add_u32_e32 v211, v147, v201
	v_add_u32_e32 v212, v148, v201
	v_add_u32_e32 v213, v149, v201
	v_add_u32_e32 v214, v150, v201
	v_add_u32_e32 v215, v151, v201
	v_add_u32_e32 v216, v152, v201
	v_add_u32_e32 v217, v153, v201
	v_add_u32_e32 v218, v154, v201
	v_add_u32_e32 v219, v147, v202
	v_add_u32_e32 v220, v148, v202
	v_add_u32_e32 v221, v149, v202
	v_add_u32_e32 v222, v150, v202
	v_add_u32_e32 v223, v151, v202
	v_add_u32_e32 v224, v152, v202
	v_add_u32_e32 v225, v153, v202
	v_add_u32_e32 v226, v154, v202
	v_add_u32_e32 v227, v155, v200
	v_add_u32_e32 v228, v156, v200
	v_add_u32_e32 v229, v157, v200
	v_add_u32_e32 v230, v158, v200
	v_add_u32_e32 v231, v159, v200
	v_add_u32_e32 v232, v160, v200
	v_add_u32_e32 v233, v161, v200
	v_add_u32_e32 v234, v162, v200
	v_add_u32_e32 v235, v155, v201
	v_add_u32_e32 v236, v156, v201
	v_add_u32_e32 v237, v157, v201
	v_add_u32_e32 v238, v158, v201
	v_add_u32_e32 v239, v159, v201
	v_add_u32_e32 v240, v160, v201
	v_add_u32_e32 v241, v161, v201
	v_add_u32_e32 v242, v162, v201
	v_add_u32_e32 v243, v155, v202
	v_add_u32_e32 v244, v156, v202
	v_add_u32_e32 v245, v157, v202
	v_add_u32_e32 v246, v158, v202
	v_add_u32_e32 v247, v159, v202
	v_add_u32_e32 v248, v160, v202
	v_add_u32_e32 v249, v161, v202
	v_add_u32_e32 v250, v162, v202
	s_waitcnt vmcnt(16)
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmpk_gt_i32 s10, 0xbf
	s_mov_b64 s[0:1], -1
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_6
; %bb.4:                                ; %.lr.ph
	s_lshr_b32 s12, s10, 6
	s_add_u32 s0, s4, s14
	s_addc_u32 s1, s5, s15
	s_add_u32 s4, s0, 0x100
	s_mul_i32 s11, s11, s28
	s_addc_u32 s5, s1, 0
	s_lshl_b32 s0, s11, 8
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, s3
	s_add_u32 s10, s0, 0x100
	s_addc_u32 s11, s1, 0
	s_add_i32 s12, s12, -3
	s_and_b32 s11, s11, 0xffff
	s_and_b32 s5, s5, 0xffff
	;; ===== SWP prologue: pre-issue iter 0's 16 loads (block 0) =====
	global_load_dwordx4 v[0:3], v150, s[10:11]
	global_load_dwordx4 v[32:35], v155, s[4:5]
	global_load_dwordx4 v[4:7], v149, s[10:11]
	global_load_dwordx4 v[36:39], v156, s[4:5]
	global_load_dwordx4 v[8:11], v152, s[10:11]
	global_load_dwordx4 v[40:43], v157, s[4:5]
	global_load_dwordx4 v[12:15], v151, s[10:11]
	global_load_dwordx4 v[44:47], v158, s[4:5]
	global_load_dwordx4 v[16:19], v154, s[10:11]
	global_load_dwordx4 v[48:51], v159, s[4:5]
	global_load_dwordx4 v[20:23], v153, s[10:11]
	global_load_dwordx4 v[52:55], v160, s[4:5]
	global_load_dwordx4 v[24:27], v147, s[10:11]
	global_load_dwordx4 v[56:59], v161, s[4:5]
	global_load_dwordx4 v[28:31], v148, s[10:11]
	global_load_dwordx4 v[60:63], v162, s[4:5]
	;; bump base addresses so the loop body's loads target block 1
	s_add_u32 s4, s4, 0x80
	s_addc_u32 s5, s5, 0
	s_add_u32 s10, s10, 0x80
	s_addc_u32 s11, s11, 0
	s_waitcnt vmcnt(0)
	;; SWP guard: if no pipelined loop iters needed, jump straight to epilogue
	s_cmp_lt_i32 s12, 1
	s_cbranch_scc1 .LBB0_epilogue
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_barrier
	;; --- SWP body, INTERLEAVED schedule (Idea 4) ---
	;; Each sub-block: wait(14) drains the 2 K-1 loads (A_i, B_i).
	;; VALU, MFMA, and VMEM operations are interleaved for each block
	s_waitcnt vmcnt(14)
	v_pk_fma_f32 v[64:65], v[0:1], v[32:33], v[64:65]
	v_mfma_f32_32x32x8_f16 a[0:15],     v[64:65], v[32:33], a[0:15]
	global_load_dwordx4 v[0:3], v150, s[10:11]
	v_pk_fma_f32 v[66:67], v[2:3], v[34:35], v[66:67]
	v_mfma_f32_32x32x8_f16 a[16:31],    v[66:67], v[34:35], a[16:31]
	global_load_dwordx4 v[32:35], v155, s[4:5]

	;; Sub-block 2: pair @ v[4:7] (A) / v[36:39] (B)
	s_waitcnt vmcnt(14)
	v_pk_fma_f32 v[68:69], v[4:5], v[36:37], v[68:69]
	v_mfma_f32_32x32x8_f16 a[32:47],    v[68:69], v[36:37], a[32:47]
	global_load_dwordx4 v[4:7], v149, s[10:11]
	v_pk_fma_f32 v[70:71], v[6:7], v[38:39], v[70:71]
	v_mfma_f32_32x32x8_f16 a[48:63],    v[70:71], v[38:39], a[48:63]
	global_load_dwordx4 v[36:39], v156, s[4:5]

	;; Sub-block 3: pair @ v[8:11] (A) / v[40:43] (B)
	s_waitcnt vmcnt(14)
	v_pk_fma_f32 v[72:73], v[8:9], v[40:41], v[72:73]
	v_mfma_f32_32x32x8_f16 a[64:79],    v[72:73], v[40:41], a[64:79]
	global_load_dwordx4 v[8:11], v152, s[10:11]
	v_pk_fma_f32 v[74:75], v[10:11], v[42:43], v[74:75]
	v_mfma_f32_32x32x8_f16 a[80:95],    v[74:75], v[42:43], a[80:95]
	global_load_dwordx4 v[40:43], v157, s[4:5]

	;; Sub-block 4: pair @ v[12:15] (A) / v[44:47] (B)
	s_waitcnt vmcnt(14)
	v_pk_fma_f32 v[76:77], v[12:13], v[44:45], v[76:77]
	v_mfma_f32_32x32x8_f16 a[96:111],   v[76:77], v[44:45], a[96:111]
	global_load_dwordx4 v[12:15], v151, s[10:11]
	v_pk_fma_f32 v[78:79], v[14:15], v[46:47], v[78:79]
	v_mfma_f32_32x32x8_f16 a[112:127],  v[78:79], v[46:47], a[112:127]
	global_load_dwordx4 v[44:47], v158, s[4:5]

	;; Sub-block 5: pair @ v[16:19] (A) / v[48:51] (B)
	s_waitcnt vmcnt(14)
	v_pk_fma_f32 v[80:81], v[16:17], v[48:49], v[80:81]
	v_mfma_f32_32x32x8_f16 a[128:143],  v[80:81], v[48:49], a[128:143]
	global_load_dwordx4 v[16:19], v154, s[10:11]
	v_pk_fma_f32 v[82:83], v[18:19], v[50:51], v[82:83]
	v_mfma_f32_32x32x8_f16 a[144:159],  v[82:83], v[50:51], a[144:159]
	global_load_dwordx4 v[48:51], v159, s[4:5]

	;; Sub-block 6: pair @ v[20:23] (A) / v[52:55] (B)
	s_waitcnt vmcnt(14)
	v_pk_fma_f32 v[84:85], v[20:21], v[52:53], v[84:85]
	v_mfma_f32_32x32x8_f16 a[160:175],  v[84:85], v[52:53], a[160:175]
	global_load_dwordx4 v[20:23], v153, s[10:11]
	v_pk_fma_f32 v[86:87], v[22:23], v[54:55], v[86:87]
	v_mfma_f32_32x32x8_f16 a[176:191],  v[86:87], v[54:55], a[176:191]
	global_load_dwordx4 v[52:55], v160, s[4:5]

	;; Sub-block 7: pair @ v[24:27] (A) / v[56:59] (B)
	s_waitcnt vmcnt(14)
	v_pk_fma_f32 v[88:89], v[24:25], v[56:57], v[88:89]
	v_mfma_f32_32x32x8_f16 a[192:207],  v[88:89], v[56:57], a[192:207]
	global_load_dwordx4 v[24:27], v147, s[10:11]
	v_pk_fma_f32 v[90:91], v[26:27], v[58:59], v[90:91]
	v_mfma_f32_32x32x8_f16 a[208:223],  v[90:91], v[58:59], a[208:223]
	global_load_dwordx4 v[56:59], v161, s[4:5]

	;; Sub-block 8: pair @ v[28:31] (A) / v[60:63] (B)
	s_waitcnt vmcnt(14)
	v_pk_fma_f32 v[92:93], v[28:29], v[60:61], v[92:93]
	v_mfma_f32_32x32x8_f16 a[224:239],  v[92:93], v[60:61], a[224:239]
	global_load_dwordx4 v[28:31], v148, s[10:11]
	v_pk_fma_f32 v[94:95], v[30:31], v[62:63], v[94:95]
	v_mfma_f32_32x32x8_f16 a[240:255],  v[94:95], v[62:63], a[240:255]
	global_load_dwordx4 v[60:63], v162, s[4:5]

	;; --- end of body: 16 K loads now in flight (overwrite v[0:63] with next block) ---
	s_barrier
	s_add_u32 s4, s4, 0x80
	s_addc_u32 s5, s5, 0
	s_add_u32 s10, s10, 0x80
	s_addc_u32 s11, s11, 0
	s_add_i32 s12, s12, -1
	s_cmp_lg_u32 s12, 0
	s_cbranch_scc1 .LBB0_5
.LBB0_epilogue:
	;; --- SWP epilogue: final iter VALU+MFMA (vmcnt staircase; no new loads) ---
	s_waitcnt vmcnt(14)
	v_pk_fma_f32 v[64:65], v[0:1], v[32:33], v[64:65]
	v_pk_fma_f32 v[66:67], v[2:3], v[34:35], v[66:67]
	v_mfma_f32_32x32x8_f16 a[0:15],     v[64:65], v[32:33], a[0:15]
	v_mfma_f32_32x32x8_f16 a[16:31],    v[66:67], v[34:35], a[16:31]
	s_waitcnt vmcnt(12)
	v_pk_fma_f32 v[68:69], v[4:5], v[36:37], v[68:69]
	v_pk_fma_f32 v[70:71], v[6:7], v[38:39], v[70:71]
	v_mfma_f32_32x32x8_f16 a[32:47],    v[68:69], v[36:37], a[32:47]
	v_mfma_f32_32x32x8_f16 a[48:63],    v[70:71], v[38:39], a[48:63]
	s_waitcnt vmcnt(10)
	v_pk_fma_f32 v[72:73], v[8:9], v[40:41], v[72:73]
	v_pk_fma_f32 v[74:75], v[10:11], v[42:43], v[74:75]
	v_mfma_f32_32x32x8_f16 a[64:79],    v[72:73], v[40:41], a[64:79]
	v_mfma_f32_32x32x8_f16 a[80:95],    v[74:75], v[42:43], a[80:95]
	s_waitcnt vmcnt(8)
	v_pk_fma_f32 v[76:77], v[12:13], v[44:45], v[76:77]
	v_pk_fma_f32 v[78:79], v[14:15], v[46:47], v[78:79]
	v_mfma_f32_32x32x8_f16 a[96:111],   v[76:77], v[44:45], a[96:111]
	v_mfma_f32_32x32x8_f16 a[112:127],  v[78:79], v[46:47], a[112:127]
	s_waitcnt vmcnt(6)
	v_pk_fma_f32 v[80:81], v[16:17], v[48:49], v[80:81]
	v_pk_fma_f32 v[82:83], v[18:19], v[50:51], v[82:83]
	v_mfma_f32_32x32x8_f16 a[128:143],  v[80:81], v[48:49], a[128:143]
	v_mfma_f32_32x32x8_f16 a[144:159],  v[82:83], v[50:51], a[144:159]
	s_waitcnt vmcnt(4)
	v_pk_fma_f32 v[84:85], v[20:21], v[52:53], v[84:85]
	v_pk_fma_f32 v[86:87], v[22:23], v[54:55], v[86:87]
	v_mfma_f32_32x32x8_f16 a[160:175],  v[84:85], v[52:53], a[160:175]
	v_mfma_f32_32x32x8_f16 a[176:191],  v[86:87], v[54:55], a[176:191]
	s_waitcnt vmcnt(2)
	v_pk_fma_f32 v[88:89], v[24:25], v[56:57], v[88:89]
	v_pk_fma_f32 v[90:91], v[26:27], v[58:59], v[90:91]
	v_mfma_f32_32x32x8_f16 a[192:207],  v[88:89], v[56:57], a[192:207]
	v_mfma_f32_32x32x8_f16 a[208:223],  v[90:91], v[58:59], a[208:223]
	s_waitcnt vmcnt(0)
	v_pk_fma_f32 v[92:93], v[28:29], v[60:61], v[92:93]
	v_pk_fma_f32 v[94:95], v[30:31], v[62:63], v[94:95]
	v_mfma_f32_32x32x8_f16 a[224:239],  v[92:93], v[60:61], a[224:239]
	v_mfma_f32_32x32x8_f16 a[240:255],  v[94:95], v[62:63], a[240:255]
.LBB0_6:                                ; %._crit_edge
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
                                        ; -- End function
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
; Kernel info:
; codeLenInByte = 23008
; TotalNumSgprs: 35
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 16
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 35
; NumVGPRsForWavesPerEU: 512
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 15
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	0                               ; DW_CHILDREN_no
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x1b DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"<unknown>"                     ; string offset=7
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
	.section	.debug_line,"",@progbits
.Lline_table_start0:
