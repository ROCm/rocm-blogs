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
	s_mov_b32 s19, 0x27000
	s_mov_b32 s18, 0x7ffffffe
	v_cndmask_b32_e64 v5, v6, v147, s[0:1]
	v_lshlrev_b32_e32 v148, 1, v4
	v_add_u32_e32 v7, s8, v4
	v_cndmask_b32_e64 v4, v6, v148, s[0:1]
	buffer_load_dwordx4 v[66:69], v5, s[16:19], 0 offen
	buffer_load_dwordx4 v[70:73], v4, s[16:19], 0 offen
	v_mul_lo_u32 v5, s11, v3
	v_lshlrev_b32_e32 v149, 1, v7
	v_lshl_add_u32 v7, s11, 6, v7
	v_add_lshl_u32 v150, v5, v2, 1
	v_cndmask_b32_e64 v4, v6, v149, s[0:1]
	v_cndmask_b32_e64 v5, v6, v150, s[0:1]
	v_lshlrev_b32_e32 v151, 1, v7
	v_add_u32_e32 v7, s8, v7
	s_cmpk_gt_i32 s10, 0x7f
	buffer_load_dwordx4 v[74:77], v4, s[16:19], 0 offen
	buffer_load_dwordx4 v[78:81], v5, s[16:19], 0 offen
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
	buffer_load_dwordx4 v[82:85], v4, s[16:19], 0 offen
	buffer_load_dwordx4 v[86:89], v9, s[16:19], 0 offen
	v_cndmask_b32_e64 v4, v6, v153, s[0:1]
	v_cndmask_b32_e64 v7, v6, v154, s[0:1]
	s_and_b32 s21, s21, 0xffff
	s_lshl_b32 s26, s12, 5
	v_mul_lo_u32 v3, s12, v3
	s_lshl_b64 s[14:15], s[14:15], 1
	buffer_load_dwordx4 v[90:93], v4, s[16:19], 0 offen
	buffer_load_dwordx4 v[94:97], v7, s[16:19], 0 offen
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
	buffer_load_dwordx4 v[98:101], v4, s[16:19], 0 offen
	buffer_load_dwordx4 v[102:105], v5, s[16:19], 0 offen
	v_cndmask_b32_e64 v4, v6, v157, s[0:1]
	v_cndmask_b32_e64 v3, v6, v158, s[0:1]
	v_lshlrev_b32_e32 v159, 1, v9
	buffer_load_dwordx4 v[106:109], v4, s[16:19], 0 offen
	buffer_load_dwordx4 v[110:113], v3, s[16:19], 0 offen
	v_cndmask_b32_e64 v3, v6, v159, s[0:1]
	v_lshlrev_b32_e32 v160, 1, v10
	v_add_lshl_u32 v161, v10, s26, 1
	v_cndmask_b32_e64 v4, v6, v160, s[0:1]
	buffer_load_dwordx4 v[114:117], v3, s[16:19], 0 offen
	buffer_load_dwordx4 v[118:121], v4, s[16:19], 0 offen
	v_cndmask_b32_e64 v3, v6, v161, s[0:1]
	v_add_lshl_u32 v162, v7, v2, 1
	v_cndmask_b32_e64 v2, v6, v162, s[0:1]
	buffer_load_dwordx4 v[126:129], v3, s[16:19], 0 offen
	buffer_load_dwordx4 v[130:133], v2, s[16:19], 0 offen
	s_mov_b32 s22, s18
	s_mov_b32 s23, s19
	v_cndmask_b32_e32 v2, v6, v147, vcc
	v_cndmask_b32_e32 v3, v6, v148, vcc
	buffer_load_dwordx4 v[58:61], v2, s[20:23], 0 offen
	buffer_load_dwordx4 v[54:57], v3, s[20:23], 0 offen
	v_cndmask_b32_e32 v2, v6, v149, vcc
	v_cndmask_b32_e32 v3, v6, v150, vcc
	buffer_load_dwordx4 v[50:53], v2, s[20:23], 0 offen
	buffer_load_dwordx4 v[62:65], v3, s[20:23], 0 offen
	v_cndmask_b32_e32 v2, v6, v151, vcc
	v_cndmask_b32_e32 v3, v6, v152, vcc
	buffer_load_dwordx4 v[46:49], v2, s[20:23], 0 offen
	buffer_load_dwordx4 v[34:37], v3, s[20:23], 0 offen
	v_cndmask_b32_e32 v2, v6, v153, vcc
	v_cndmask_b32_e32 v3, v6, v154, vcc
	buffer_load_dwordx4 v[42:45], v2, s[20:23], 0 offen
	buffer_load_dwordx4 v[30:33], v3, s[20:23], 0 offen
	s_and_b32 s25, s12, 0xffff
	s_mov_b32 s26, s18
	s_mov_b32 s27, s19
	v_cndmask_b32_e32 v2, v6, v155, vcc
	v_cndmask_b32_e32 v3, v6, v156, vcc
	buffer_load_dwordx4 v[38:41], v2, s[24:27], 0 offen
	buffer_load_dwordx4 v[26:29], v3, s[24:27], 0 offen
	v_cndmask_b32_e32 v2, v6, v157, vcc
	v_cndmask_b32_e32 v3, v6, v158, vcc
	buffer_load_dwordx4 v[22:25], v2, s[24:27], 0 offen
	buffer_load_dwordx4 v[18:21], v3, s[24:27], 0 offen
	v_cndmask_b32_e32 v2, v6, v159, vcc
	v_cndmask_b32_e32 v3, v6, v160, vcc
	buffer_load_dwordx4 v[14:17], v2, s[24:27], 0 offen
	buffer_load_dwordx4 v[10:13], v3, s[24:27], 0 offen
	v_cndmask_b32_e32 v2, v6, v161, vcc
	v_cndmask_b32_e32 v3, v6, v162, vcc
	buffer_load_dwordx4 v[6:9], v2, s[24:27], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[2:5], v3, s[24:27], 0 offen
	v_xor_b32_e32 v122, v122, v0
	v_and_b32_e32 v122, 56, v122
	v_lshlrev_b32_e32 v122, 1, v122
	v_lshl_or_b32 v122, v1, 7, v122
	v_add_u32_e32 v125, 0, v122
	s_add_i32 s0, 0, 0x8000
	s_waitcnt vmcnt(31)
	ds_write_b128 v125, v[66:69]
	s_waitcnt vmcnt(30)
	ds_write_b128 v125, v[70:73] offset:4096
	s_waitcnt vmcnt(29)
	ds_write_b128 v125, v[74:77] offset:8192
	s_waitcnt vmcnt(28)
	ds_write_b128 v125, v[78:81] offset:12288
	s_waitcnt vmcnt(27)
	ds_write_b128 v125, v[82:85] offset:16384
	s_waitcnt vmcnt(26)
	ds_write_b128 v125, v[86:89] offset:20480
	s_waitcnt vmcnt(25)
	ds_write_b128 v125, v[90:93] offset:24576
	s_waitcnt vmcnt(24)
	ds_write_b128 v125, v[94:97] offset:28672
	s_waitcnt vmcnt(23)
	ds_write_b128 v125, v[98:101] offset:32768
	v_add_u32_e32 v123, s0, v122
	v_and_b32_e32 v66, 15, v0
	v_lshrrev_b32_e32 v144, 2, v0
	s_waitcnt vmcnt(22)
	ds_write_b128 v123, v[102:105] offset:4096
	s_waitcnt vmcnt(21)
	ds_write_b128 v123, v[106:109] offset:8192
	s_waitcnt vmcnt(20)
	ds_write_b128 v123, v[110:113] offset:12288
	s_waitcnt vmcnt(19)
	ds_write_b128 v123, v[114:117] offset:16384
	s_waitcnt vmcnt(18)
	ds_write_b128 v123, v[118:121] offset:20480
	s_waitcnt vmcnt(17)
	ds_write_b128 v123, v[126:129] offset:24576
	s_waitcnt vmcnt(16)
	ds_write_b128 v123, v[130:133] offset:28672
	v_bfe_u32 v108, v0, 4, 2
	v_and_b32_e32 v99, 7, v0
	v_and_or_b32 v0, v144, 16, v66
	v_and_or_b32 v67, v1, 16, v66
	v_xor_b32_e32 v1, v108, v99
	v_lshlrev_b32_e32 v103, 6, v0
	v_lshlrev_b32_e32 v0, 7, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e32 v100, 6, v67
	scratch_store_dword off, v67, off       ; 4-byte Folded Spill
	v_lshl_add_u32 v98, v67, 7, 0
	v_lshlrev_b32_e32 v67, 4, v1
	v_add_u32_e32 v105, s0, v0
	v_lshlrev_b32_e32 v102, 3, v1
	v_add_u32_e32 v1, v98, v67
	v_add3_u32 v124, 0, v67, v0
	v_add_u32_e32 v0, v105, v67
	ds_read_b128 v[94:97], v1
	ds_read_b128 v[90:93], v1 offset:4096
	ds_read_b128 v[78:81], v124 offset:32768
	ds_read_b128 v[74:77], v124 offset:36864
	ds_read_b128 v[86:89], v1 offset:8192
	ds_read_b128 v[82:85], v1 offset:12288
	ds_read_b128 v[70:73], v0 offset:8192
	ds_read_b128 v[66:69], v0 offset:12288
	v_or_b32_e32 v101, 0x800, v100
	v_or_b32_e32 v104, 0x1000, v100
	v_or_b32_e32 v106, 0x1800, v100
	v_or_b32_e32 v107, 0x1000, v103
	s_cmpk_gt_i32 s10, 0xbf
	v_or_b32_e32 v108, 4, v108
	s_cbranch_scc1 .LBB0_2
; %bb.1:                                ; %.._crit_edge_crit_edge
	v_xor_b32_e32 v115, v108, v99
	v_or_b32_e32 v109, 0x2000, v103
	v_or_b32_e32 v110, 0x3000, v103
	v_or_b32_e32 v111, 0x2000, v100
	v_lshlrev_b32_e32 v115, 3, v115
	v_or_b32_e32 v146, v109, v102
	v_or_b32_e32 v112, 0x2800, v100
	v_or_b32_e32 v113, 0x3000, v100
	v_or_b32_e32 v114, 0x3800, v100
	v_or_b32_e32 v132, v115, v109
	v_or_b32_e32 v130, v115, v110
	v_or_b32_e32 v109, v111, v115
	v_or_b32_e32 v145, v110, v102
	v_or_b32_e32 v143, v111, v102
	v_or_b32_e32 v142, v112, v102
	v_or_b32_e32 v141, v113, v102
	v_or_b32_e32 v140, v114, v102
	v_or_b32_e32 v139, v115, v103
	v_or_b32_e32 v137, v115, v100
	v_or_b32_e32 v138, v101, v115
	v_or_b32_e32 v136, v115, v107
	v_or_b32_e32 v134, v104, v115
	v_or_b32_e32 v135, v106, v115
	v_or_b32_e32 v133, 0x800, v132
	v_or_b32_e32 v131, 0x800, v130
	scratch_store_dword off, v109, off offset:4 ; 4-byte Folded Spill
	v_or_b32_e32 v109, v112, v115
	v_or_b32_e32 v250, v113, v115
	v_or_b32_e32 v122, v114, v115
	s_mov_b64 s[0:1], 0
	scratch_store_dword off, v109, off offset:8 ; 4-byte Folded Spill
	s_branch .LBB0_3
.LBB0_2:
                                        ; implicit-def: $vgpr109
	s_mov_b64 s[0:1], -1
                                        ; implicit-def: $vgpr146
                                        ; implicit-def: $vgpr145
                                        ; implicit-def: $vgpr143
                                        ; implicit-def: $vgpr142
                                        ; implicit-def: $vgpr141
                                        ; implicit-def: $vgpr140
                                        ; implicit-def: $vgpr139
                                        ; implicit-def: $vgpr137
                                        ; implicit-def: $vgpr138
                                        ; implicit-def: $vgpr136
                                        ; implicit-def: $vgpr134
                                        ; implicit-def: $vgpr135
                                        ; implicit-def: $vgpr132
                                        ; implicit-def: $vgpr133
                                        ; implicit-def: $vgpr130
                                        ; implicit-def: $vgpr131
                                        ; kill: killed $vgpr109
                                        ; implicit-def: $vgpr109
                                        ; kill: killed $vgpr109
                                        ; implicit-def: $vgpr250
                                        ; implicit-def: $vgpr122
.LBB0_3:                                ; %Flow2534
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a83, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a162, 0
	v_accvgpr_write_b32 a163, 0
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a223, 0
	v_accvgpr_write_b32 a224, 0
	v_accvgpr_write_b32 a225, 0
	v_accvgpr_write_b32 a226, 0
	v_accvgpr_write_b32 a227, 0
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a234, 0
	v_accvgpr_write_b32 a235, 0
	v_accvgpr_write_b32 a236, 0
	v_accvgpr_write_b32 a237, 0
	v_accvgpr_write_b32 a238, 0
	v_accvgpr_write_b32 a239, 0
	v_accvgpr_write_b32 a240, 0
	v_accvgpr_write_b32 a241, 0
	v_accvgpr_write_b32 a242, 0
	v_accvgpr_write_b32 a243, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a252, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a255, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a183, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a144, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_write_b32 a188, 0
	v_accvgpr_write_b32 a189, 0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a204, 0
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a231, 0
	v_accvgpr_write_b32 a196, 0
	v_accvgpr_write_b32 a197, 0
	v_accvgpr_write_b32 a198, 0
	v_accvgpr_write_b32 a199, 0
	v_accvgpr_write_b32 a200, 0
	v_accvgpr_write_b32 a201, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a203, 0
	v_accvgpr_write_b32 a208, 0
	v_accvgpr_write_b32 a209, 0
	v_accvgpr_write_b32 a210, 0
	v_accvgpr_write_b32 a211, 0
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a214, 0
	s_andn2_b64 vcc, exec, s[0:1]
	v_accvgpr_write_b32 a215, 0
	s_cbranch_vccnz .LBB0_6
; %bb.4:                                ; %.lr.ph
	s_lshr_b32 s12, s10, 6
	s_add_u32 s0, s4, s14
	s_addc_u32 s1, s5, s15
	s_add_u32 s4, s0, 0x100
	s_mul_i32 s11, s11, s28
	s_addc_u32 s5, s1, 0
	s_lshl_b32 s0, s11, 8
	v_lshl_add_u32 v110, v102, 1, v105
	s_ashr_i32 s1, s0, 31
	v_or_b32_e32 v109, 0x2000, v103
	v_or_b32_e32 v111, 0x3000, v103
	v_add_u32_e32 v164, 0x6000, v110
	v_or_b32_e32 v112, 0x2800, v100
	v_or_b32_e32 v113, 0x3000, v100
	v_or_b32_e32 v114, 0x3800, v100
	v_xor_b32_e32 v99, v108, v99
	v_add_u32_e32 v163, 0x4000, v110
	v_or_b32_e32 v110, 0x2000, v100
	s_lshl_b64 s[0:1], s[0:1], 1
	v_or_b32_e32 v146, v109, v102
	v_or_b32_e32 v145, v111, v102
	v_or_b32_e32 v143, v110, v102
	v_or_b32_e32 v142, v112, v102
	v_or_b32_e32 v141, v113, v102
	v_or_b32_e32 v140, v114, v102
	v_lshlrev_b32_e32 v102, 3, v99
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, s3
	v_or_b32_e32 v122, v114, v102
	v_accvgpr_write_b32 a15, 0
	s_add_u32 s10, s0, 0x100
	s_addc_u32 s11, s1, 0
	s_add_i32 s12, s12, -2
	v_accvgpr_write_b32 a14, 0
	v_lshlrev_b32_e32 v99, 4, v99
	v_add_u32_e32 v167, v105, v99
	v_or_b32_e32 v137, v102, v100
	v_add_u32_e32 v165, v98, v99
	v_lshlrev_b32_e32 v98, 1, v103
	v_add3_u32 v166, 0, v99, v98
	v_or_b32_e32 v98, v110, v102
	scratch_store_dword off, v98, off offset:4 ; 4-byte Folded Spill
	v_or_b32_e32 v98, v112, v102
	v_accvgpr_write_b32 a13, 0
	v_or_b32_e32 v138, v101, v102
	v_or_b32_e32 v139, v102, v103
	v_or_b32_e32 v136, v102, v107
	v_or_b32_e32 v134, v104, v102
	v_or_b32_e32 v135, v106, v102
	v_or_b32_e32 v132, v102, v109
	v_or_b32_e32 v133, 0x800, v132
	v_or_b32_e32 v130, v102, v111
	v_or_b32_e32 v131, 0x800, v130
	v_or_b32_e32 v250, v113, v102
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a83, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a163, 0
	v_accvgpr_write_b32 a162, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a239, 0
	v_accvgpr_write_b32 a238, 0
	v_accvgpr_write_b32 a237, 0
	v_accvgpr_write_b32 a236, 0
	v_accvgpr_write_b32 a235, 0
	v_accvgpr_write_b32 a234, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a227, 0
	v_accvgpr_write_b32 a226, 0
	v_accvgpr_write_b32 a225, 0
	v_accvgpr_write_b32 a224, 0
	v_accvgpr_write_b32 a223, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a255, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a252, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a243, 0
	v_accvgpr_write_b32 a242, 0
	v_accvgpr_write_b32 a241, 0
	v_accvgpr_write_b32 a240, 0
	v_accvgpr_write_b32 a183, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a189, 0
	v_accvgpr_write_b32 a188, 0
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a144, 0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a231, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a204, 0
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a215, 0
	v_accvgpr_write_b32 a214, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a211, 0
	v_accvgpr_write_b32 a210, 0
	v_accvgpr_write_b32 a209, 0
	v_accvgpr_write_b32 a208, 0
	v_accvgpr_write_b32 a203, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a201, 0
	v_accvgpr_write_b32 a200, 0
	v_accvgpr_write_b32 a199, 0
	v_accvgpr_write_b32 a198, 0
	v_accvgpr_write_b32 a197, 0
	v_accvgpr_write_b32 a196, 0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	scratch_store_dword off, v98, off offset:8 ; 4-byte Folded Spill
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x16_f16 a[0:3], v[78:79], v[94:95], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[80:81], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_f16 a[4:7], v[74:75], v[94:95], a[4:7]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[4:7], v[76:77], v[96:97], a[4:7]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[8:11], v[78:79], v[90:91], a[8:11]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[8:11], v[80:81], v[92:93], a[8:11]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[74:75], v[90:91], a[12:15]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[76:77], v[92:93], a[12:15]
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 a[16:19], v[70:71], v[94:95], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[72:73], v[96:97], a[16:19]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[20:23], v[66:67], v[94:95], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[20:23], v[68:69], v[96:97], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[24:27], v[70:71], v[90:91], a[24:27]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[24:27], v[72:73], v[92:93], a[24:27]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[28:31], v[66:67], v[90:91], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[28:31], v[68:69], v[92:93], a[28:31]
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x16_f16 a[32:35], v[78:79], v[86:87], a[32:35]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[32:35], v[80:81], v[88:89], a[32:35]
	ds_read_b128 v[106:109], v0 offset:16384
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[36:39], v[74:75], v[86:87], a[36:39]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[36:39], v[76:77], v[88:89], a[36:39]
	ds_read_b128 v[110:113], v163 offset:4096
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[40:43], v[78:79], v[82:83], a[40:43]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[40:43], v[80:81], v[84:85], a[40:43]
	ds_read_b128 v[98:101], v0 offset:24576
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[44:47], v[74:75], v[82:83], a[44:47]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[44:47], v[76:77], v[84:85], a[44:47]
	ds_read_b128 v[102:105], v164 offset:4096
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x16_f16 a[48:51], v[70:71], v[86:87], a[48:51]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[48:51], v[72:73], v[88:89], a[48:51]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[52:55], v[66:67], v[86:87], a[52:55]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[52:55], v[68:69], v[88:89], a[52:55]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[56:59], v[70:71], v[82:83], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[56:59], v[72:73], v[84:85], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[60:63], v[66:67], v[82:83], a[60:63]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[60:63], v[68:69], v[84:85], a[60:63]
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 a[64:67], v[106:107], v[94:95], a[64:67]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[64:67], v[108:109], v[96:97], a[64:67]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 a[68:71], v[110:111], v[94:95], a[68:71]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[68:71], v[112:113], v[96:97], a[68:71]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[72:75], v[106:107], v[90:91], a[72:75]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[72:75], v[108:109], v[92:93], a[72:75]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[76:79], v[110:111], v[90:91], a[76:79]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[76:79], v[112:113], v[92:93], a[76:79]
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 a[80:83], v[98:99], v[94:95], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[80:83], v[100:101], v[96:97], a[80:83]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[84:87], v[102:103], v[94:95], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[104:105], v[96:97], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[88:91], v[98:99], v[90:91], a[88:91]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[88:91], v[100:101], v[92:93], a[88:91]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[92:95], v[102:103], v[90:91], a[92:95]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[92:95], v[104:105], v[92:93], a[92:95]
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x16_f16 a[96:99], v[106:107], v[86:87], a[96:99]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[96:99], v[108:109], v[88:89], a[96:99]
	ds_read_b128 v[90:93], v1 offset:16384
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[100:103], v[110:111], v[86:87], a[100:103]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[100:103], v[112:113], v[88:89], a[100:103]
	ds_read_b128 v[94:97], v1 offset:20480
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[104:107], v[106:107], v[82:83], a[104:107]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[104:107], v[108:109], v[84:85], a[104:107]
	ds_read_b128 v[114:117], v1 offset:24576
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[108:111], v[110:111], v[82:83], a[108:111]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[108:111], v[112:113], v[84:85], a[108:111]
	ds_read_b128 v[168:171], v1 offset:28672
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x16_f16 a[112:115], v[98:99], v[86:87], a[112:115]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[112:115], v[100:101], v[88:89], a[112:115]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[116:119], v[102:103], v[86:87], a[116:119]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[116:119], v[104:105], v[88:89], a[116:119]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[120:123], v[98:99], v[82:83], a[120:123]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[120:123], v[100:101], v[84:85], a[120:123]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[124:127], v[102:103], v[82:83], a[124:127]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[124:127], v[104:105], v[84:85], a[124:127]
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 a[132:135], v[78:79], v[90:91], a[132:135]
	; sched_barrier mask(0x000007F6)
	s_and_b32 s1, s11, 0xffff
	s_mov_b32 s0, s10
	v_mfma_f32_16x16x16_f16 a[132:135], v[80:81], v[92:93], a[132:135]
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[136:139], v[74:75], v[90:91], a[136:139]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[136:139], v[76:77], v[92:93], a[136:139]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 a[140:143], v[78:79], v[94:95], a[140:143]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[140:143], v[80:81], v[96:97], a[140:143]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[148:151], v[74:75], v[94:95], a[148:151]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[148:151], v[76:77], v[96:97], a[148:151]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[152:155], v[70:71], v[90:91], a[152:155]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[152:155], v[72:73], v[92:93], a[152:155]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[156:159], v[66:67], v[90:91], a[156:159]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[156:159], v[68:69], v[92:93], a[156:159]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[160:163], v[70:71], v[94:95], a[160:163]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[160:163], v[72:73], v[96:97], a[160:163]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[164:167], v[66:67], v[94:95], a[164:167]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[164:167], v[68:69], v[96:97], a[164:167]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 a[168:171], v[78:79], v[114:115], a[168:171]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[168:171], v[80:81], v[116:117], a[168:171]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[172:175], v[74:75], v[114:115], a[172:175]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[172:175], v[76:77], v[116:117], a[172:175]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[176:179], v[78:79], v[168:169], a[176:179]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[176:179], v[80:81], v[170:171], a[176:179]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[180:183], v[74:75], v[168:169], a[180:183]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[180:183], v[76:77], v[170:171], a[180:183]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[128:131], v[70:71], v[114:115], a[128:131]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[128:131], v[72:73], v[116:117], a[128:131]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[144:147], v[66:67], v[114:115], a[144:147]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[144:147], v[68:69], v[116:117], a[144:147]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[184:187], v[70:71], v[168:169], a[184:187]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[184:187], v[72:73], v[170:171], a[184:187]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[188:191], v[66:67], v[168:169], a[188:191]
	; sched_barrier mask(0x000007F6)
	v_lshl_add_u32 v66, v137, 1, 0
	v_mfma_f32_16x16x16_f16 a[188:191], v[68:69], v[170:171], a[188:191]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000406)
	v_mfma_f32_16x16x16_f16 a[220:223], v[106:107], v[90:91], a[220:223]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[220:223], v[108:109], v[92:93], a[220:223]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[224:227], v[110:111], v[90:91], a[224:227]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[224:227], v[112:113], v[92:93], a[224:227]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[232:235], v[106:107], v[94:95], a[232:235]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[232:235], v[108:109], v[96:97], a[232:235]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[236:239], v[110:111], v[94:95], a[236:239]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[236:239], v[112:113], v[96:97], a[236:239]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000406)
	ds_read_b128 v[172:175], v66
	v_lshl_add_u32 v66, v139, 1, 0
	v_mfma_f32_16x16x16_f16 a[240:243], v[98:99], v[90:91], a[240:243]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[240:243], v[100:101], v[92:93], a[240:243]
	ds_read_b128 v[176:179], v165 offset:4096
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[244:247], v[102:103], v[90:91], a[244:247]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[244:247], v[104:105], v[92:93], a[244:247]
	ds_read_b128 v[78:81], v66 offset:32768
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[248:251], v[98:99], v[94:95], a[248:251]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[248:251], v[100:101], v[96:97], a[248:251]
	ds_read_b128 v[82:85], v166 offset:36864
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[252:255], v[102:103], v[94:95], a[252:255]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[252:255], v[104:105], v[96:97], a[252:255]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000406)
	ds_read_b128 v[118:121], v165 offset:12288
	ds_read_b128 v[94:97], v165 offset:8192
	ds_read_b128 v[70:73], v166 offset:45056
	ds_read_b128 v[86:89], v167 offset:8192
	v_mfma_f32_16x16x16_f16 a[192:195], v[106:107], v[114:115], a[192:195]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[192:195], v[108:109], v[116:117], a[192:195]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[204:207], v[110:111], v[114:115], a[204:207]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[204:207], v[112:113], v[116:117], a[204:207]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[216:219], v[106:107], v[168:169], a[216:219]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[216:219], v[108:109], v[170:171], a[216:219]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[228:231], v[110:111], v[168:169], a[228:231]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[228:231], v[112:113], v[170:171], a[228:231]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000406)
	v_mfma_f32_16x16x16_f16 a[196:199], v[98:99], v[114:115], a[196:199]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[196:199], v[100:101], v[116:117], a[196:199]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[200:203], v[102:103], v[114:115], a[200:203]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[200:203], v[104:105], v[116:117], a[200:203]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[208:211], v[98:99], v[168:169], a[208:211]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[208:211], v[100:101], v[170:171], a[208:211]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[212:215], v[102:103], v[168:169], a[212:215]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[212:215], v[104:105], v[170:171], a[212:215]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000406)
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x16_f16 a[0:3], v[78:79], v[172:173], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[80:81], v[174:175], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_f16 a[4:7], v[82:83], v[172:173], a[4:7]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[4:7], v[84:85], v[174:175], a[4:7]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[8:11], v[78:79], v[176:177], a[8:11]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[8:11], v[80:81], v[178:179], a[8:11]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[82:83], v[176:177], a[12:15]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[84:85], v[178:179], a[12:15]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000406)
	ds_read_b128 v[106:109], v167 offset:16384
	ds_read_b128 v[66:69], v167 offset:20480
	ds_read_b128 v[102:105], v167 offset:24576
	ds_read_b128 v[98:101], v167 offset:28672
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_f16 a[16:19], v[86:87], v[172:173], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[88:89], v[174:175], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[20:23], v[70:71], v[172:173], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[20:23], v[72:73], v[174:175], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[24:27], v[86:87], v[176:177], a[24:27]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[24:27], v[88:89], v[178:179], a[24:27]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[28:31], v[70:71], v[176:177], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[28:31], v[72:73], v[178:179], a[28:31]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000406)
	ds_read_b128 v[90:93], v165 offset:16384
	ds_read_b128 v[74:77], v165 offset:20480
	ds_read_b128 v[114:117], v165 offset:24576
	ds_read_b128 v[110:113], v165 offset:28672
	v_mfma_f32_16x16x16_f16 a[32:35], v[78:79], v[94:95], a[32:35]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x16_f16 a[32:35], v[80:81], v[96:97], a[32:35]
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[36:39], v[82:83], v[94:95], a[36:39]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[36:39], v[84:85], v[96:97], a[36:39]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[40:43], v[78:79], v[118:119], a[40:43]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[40:43], v[80:81], v[120:121], a[40:43]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[44:47], v[82:83], v[118:119], a[44:47]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[44:47], v[84:85], v[120:121], a[44:47]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000406)
	v_mfma_f32_16x16x16_f16 a[48:51], v[86:87], v[94:95], a[48:51]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[48:51], v[88:89], v[96:97], a[48:51]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[52:55], v[70:71], v[94:95], a[52:55]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[52:55], v[72:73], v[96:97], a[52:55]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[56:59], v[86:87], v[118:119], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[56:59], v[88:89], v[120:121], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[60:63], v[70:71], v[118:119], a[60:63]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[60:63], v[72:73], v[120:121], a[60:63]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000406)
	s_waitcnt vmcnt(15)
	ds_write_b64 v125, v[62:63] offset:12288
	ds_write_b64 v125, v[64:65] offset:12296
	buffer_load_dwordx4 v[62:65], v150, s[0:3], 0 offen
	s_waitcnt vmcnt(15)
	ds_write_b64 v125, v[50:51] offset:8192
	ds_write_b64 v125, v[52:53] offset:8200
	buffer_load_dwordx4 v[50:53], v149, s[0:3], 0 offen
	s_waitcnt vmcnt(15)
	ds_write_b64 v125, v[34:35] offset:20480
	ds_write_b64 v125, v[36:37] offset:20488
	buffer_load_dwordx4 v[34:37], v152, s[0:3], 0 offen
	s_waitcnt vmcnt(15)
	ds_write_b64 v125, v[46:47] offset:16384
	ds_write_b64 v125, v[48:49] offset:16392
	buffer_load_dwordx4 v[46:49], v151, s[0:3], 0 offen
	s_waitcnt vmcnt(15)
	ds_write_b64 v125, v[30:31] offset:28672
	ds_write_b64 v125, v[32:33] offset:28680
	buffer_load_dwordx4 v[30:33], v154, s[0:3], 0 offen
	s_waitcnt vmcnt(15)
	ds_write_b64 v125, v[42:43] offset:24576
	ds_write_b64 v125, v[44:45] offset:24584
	buffer_load_dwordx4 v[42:45], v153, s[0:3], 0 offen
	s_waitcnt vmcnt(15)
	ds_write_b64 v125, v[58:59]
	ds_write_b64 v125, v[60:61] offset:8
	buffer_load_dwordx4 v[58:61], v147, s[0:3], 0 offen
	s_waitcnt vmcnt(15)
	ds_write_b64 v125, v[54:55] offset:4096
	ds_write_b64 v125, v[56:57] offset:4104
	buffer_load_dwordx4 v[54:57], v148, s[0:3], 0 offen
	v_mfma_f32_16x16x16_f16 a[64:67], v[106:107], v[172:173], a[64:67]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[64:67], v[108:109], v[174:175], a[64:67]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[68:71], v[66:67], v[172:173], a[68:71]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[68:71], v[68:69], v[174:175], a[68:71]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[72:75], v[106:107], v[176:177], a[72:75]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[72:75], v[108:109], v[178:179], a[72:75]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[76:79], v[66:67], v[176:177], a[76:79]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[76:79], v[68:69], v[178:179], a[76:79]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[80:83], v[102:103], v[172:173], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[80:83], v[104:105], v[174:175], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[98:99], v[172:173], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[100:101], v[174:175], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[88:91], v[102:103], v[176:177], a[88:91]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[88:91], v[104:105], v[178:179], a[88:91]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[92:95], v[98:99], v[176:177], a[92:95]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[92:95], v[100:101], v[178:179], a[92:95]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[96:99], v[106:107], v[94:95], a[96:99]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[96:99], v[108:109], v[96:97], a[96:99]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[100:103], v[66:67], v[94:95], a[100:103]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[100:103], v[68:69], v[96:97], a[100:103]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[104:107], v[106:107], v[118:119], a[104:107]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[104:107], v[108:109], v[120:121], a[104:107]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[108:111], v[66:67], v[118:119], a[108:111]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[108:111], v[68:69], v[120:121], a[108:111]
	; sched_barrier mask(0x000007F6)
	s_and_b32 s1, s5, 0xffff
	s_mov_b32 s0, s4
	v_mfma_f32_16x16x16_f16 a[112:115], v[102:103], v[94:95], a[112:115]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[112:115], v[104:105], v[96:97], a[112:115]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[116:119], v[98:99], v[94:95], a[116:119]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[116:119], v[100:101], v[96:97], a[116:119]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[120:123], v[102:103], v[118:119], a[120:123]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[120:123], v[104:105], v[120:121], a[120:123]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[124:127], v[98:99], v[118:119], a[124:127]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[124:127], v[100:101], v[120:121], a[124:127]
	; sched_barrier mask(0x00000406)
	s_waitcnt vmcnt(15)
	ds_write_b128 v125, v[38:41] offset:32768
	s_waitcnt vmcnt(14)
	ds_write_b128 v123, v[26:29] offset:4096
	s_waitcnt vmcnt(13)
	ds_write_b128 v123, v[22:25] offset:8192
	s_waitcnt vmcnt(12)
	ds_write_b128 v123, v[18:21] offset:12288
	s_waitcnt vmcnt(11)
	ds_write_b128 v123, v[14:17] offset:16384
	s_waitcnt vmcnt(10)
	ds_write_b128 v123, v[10:13] offset:20480
	s_waitcnt vmcnt(9)
	ds_write_b128 v123, v[6:9] offset:24576
	s_waitcnt vmcnt(8)
	ds_write_b128 v123, v[2:5] offset:28672
	buffer_load_dwordx4 v[38:41], v155, s[0:3], 0 offen
	buffer_load_dwordx4 v[26:29], v156, s[0:3], 0 offen
	buffer_load_dwordx4 v[22:25], v157, s[0:3], 0 offen
	buffer_load_dwordx4 v[18:21], v158, s[0:3], 0 offen
	buffer_load_dwordx4 v[14:17], v159, s[0:3], 0 offen
	buffer_load_dwordx4 v[10:13], v160, s[0:3], 0 offen
	buffer_load_dwordx4 v[6:9], v161, s[0:3], 0 offen
	buffer_load_dwordx4 v[2:5], v162, s[0:3], 0 offen
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[132:135], v[78:79], v[90:91], a[132:135]
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[132:135], v[80:81], v[92:93], a[132:135]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[136:139], v[82:83], v[90:91], a[136:139]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[136:139], v[84:85], v[92:93], a[136:139]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[140:143], v[78:79], v[74:75], a[140:143]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[140:143], v[80:81], v[76:77], a[140:143]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[148:151], v[82:83], v[74:75], a[148:151]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[148:151], v[84:85], v[76:77], a[148:151]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[152:155], v[86:87], v[90:91], a[152:155]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[152:155], v[88:89], v[92:93], a[152:155]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[156:159], v[70:71], v[90:91], a[156:159]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[156:159], v[72:73], v[92:93], a[156:159]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[160:163], v[86:87], v[74:75], a[160:163]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[160:163], v[88:89], v[76:77], a[160:163]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[164:167], v[70:71], v[74:75], a[164:167]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[164:167], v[72:73], v[76:77], a[164:167]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[168:171], v[78:79], v[114:115], a[168:171]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[168:171], v[80:81], v[116:117], a[168:171]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[172:175], v[82:83], v[114:115], a[172:175]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[172:175], v[84:85], v[116:117], a[172:175]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[176:179], v[78:79], v[110:111], a[176:179]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[176:179], v[80:81], v[112:113], a[176:179]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[180:183], v[82:83], v[110:111], a[180:183]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[180:183], v[84:85], v[112:113], a[180:183]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[128:131], v[86:87], v[114:115], a[128:131]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[128:131], v[88:89], v[116:117], a[128:131]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[144:147], v[70:71], v[114:115], a[144:147]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[144:147], v[72:73], v[116:117], a[144:147]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[184:187], v[86:87], v[110:111], a[184:187]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[184:187], v[88:89], v[112:113], a[184:187]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[188:191], v[70:71], v[110:111], a[188:191]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[188:191], v[72:73], v[112:113], a[188:191]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x16_f16 a[220:223], v[106:107], v[90:91], a[220:223]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[220:223], v[108:109], v[92:93], a[220:223]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[224:227], v[66:67], v[90:91], a[224:227]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[224:227], v[68:69], v[92:93], a[224:227]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[232:235], v[106:107], v[74:75], a[232:235]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[232:235], v[108:109], v[76:77], a[232:235]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[236:239], v[66:67], v[74:75], a[236:239]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[236:239], v[68:69], v[76:77], a[236:239]
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x16_f16 a[240:243], v[102:103], v[90:91], a[240:243]
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x000007F6)
	s_barrier
	v_mfma_f32_16x16x16_f16 a[240:243], v[104:105], v[92:93], a[240:243]
	ds_read_b128 v[94:97], v1
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[244:247], v[98:99], v[90:91], a[244:247]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[244:247], v[100:101], v[92:93], a[244:247]
	ds_read_b128 v[90:93], v1 offset:4096
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[248:251], v[102:103], v[74:75], a[248:251]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[248:251], v[104:105], v[76:77], a[248:251]
	ds_read_b128 v[78:81], v124 offset:32768
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[252:255], v[98:99], v[74:75], a[252:255]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[252:255], v[100:101], v[76:77], a[252:255]
	ds_read_b128 v[74:77], v124 offset:36864
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x16_f16 a[192:195], v[106:107], v[114:115], a[192:195]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[192:195], v[108:109], v[116:117], a[192:195]
	ds_read_b128 v[86:89], v1 offset:8192
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[204:207], v[66:67], v[114:115], a[204:207]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[204:207], v[68:69], v[116:117], a[204:207]
	ds_read_b128 v[82:85], v1 offset:12288
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[216:219], v[106:107], v[110:111], a[216:219]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[216:219], v[108:109], v[112:113], a[216:219]
	ds_read_b128 v[70:73], v0 offset:8192
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_f32_16x16x16_f16 a[228:231], v[66:67], v[110:111], a[228:231]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[228:231], v[68:69], v[112:113], a[228:231]
	ds_read_b128 v[66:69], v0 offset:12288
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x16_f16 a[196:199], v[102:103], v[114:115], a[196:199]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[196:199], v[104:105], v[116:117], a[196:199]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[200:203], v[98:99], v[114:115], a[200:203]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[200:203], v[100:101], v[116:117], a[200:203]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[208:211], v[102:103], v[110:111], a[208:211]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[208:211], v[104:105], v[112:113], a[208:211]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[212:215], v[98:99], v[110:111], a[212:215]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[212:215], v[100:101], v[112:113], a[212:215]
	; sched_barrier mask(0x000007F6)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_add_u32 s4, s4, 0x80
	s_addc_u32 s5, s5, 0
	s_add_u32 s10, s10, 0x80
	s_addc_u32 s11, s11, 0
	s_add_i32 s12, s12, -1
	s_cmp_lg_u32 s12, 0
	s_cbranch_scc1 .LBB0_5
.LBB0_6:                                ; %._crit_edge
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x16_f16 a[0:3], v[78:79], v[94:95], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_and_b32_e32 v104, 28, v144
	v_or_b32_e32 v251, 0xe0, v104
	v_or_b32_e32 v102, 0x60, v104
	v_mfma_f32_16x16x16_f16 a[0:3], v[80:81], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_or_b32_e32 v103, 64, v104
	v_or_b32_e32 v105, 32, v104
	s_nop 4
	v_accvgpr_read_b32 v101, a3
	v_accvgpr_read_b32 v100, a2
	v_accvgpr_read_b32 v99, a1
	v_accvgpr_read_b32 v98, a0
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_f16 a[0:3], v[74:75], v[94:95], a[4:7]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[76:77], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v151, a3
	v_accvgpr_read_b32 v150, a2
	v_accvgpr_read_b32 v149, a1
	v_accvgpr_read_b32 v148, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[78:79], v[90:91], a[8:11]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[80:81], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v155, a3
	v_accvgpr_read_b32 v154, a2
	v_accvgpr_read_b32 v153, a1
	v_accvgpr_read_b32 v152, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[74:75], v[90:91], a[12:15]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[76:77], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v159, a3
	v_accvgpr_read_b32 v158, a2
	v_accvgpr_read_b32 v157, a1
	v_accvgpr_read_b32 v156, a0
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 a[0:3], v[70:71], v[94:95], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[72:73], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v163, a3
	v_accvgpr_read_b32 v162, a2
	v_accvgpr_read_b32 v161, a1
	v_accvgpr_read_b32 v160, a0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[0:3], v[66:67], v[94:95], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[68:69], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v167, a3
	v_accvgpr_read_b32 v166, a2
	v_accvgpr_read_b32 v165, a1
	v_accvgpr_read_b32 v164, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[70:71], v[90:91], a[24:27]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[72:73], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v171, a3
	v_accvgpr_read_b32 v170, a2
	v_accvgpr_read_b32 v169, a1
	v_accvgpr_read_b32 v168, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[66:67], v[90:91], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[68:69], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v179, a3
	v_accvgpr_read_b32 v178, a2
	v_accvgpr_read_b32 v177, a1
	v_accvgpr_read_b32 v176, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[78:79], v[86:87], a[32:35]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[80:81], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v175, a3
	v_accvgpr_read_b32 v174, a2
	v_accvgpr_read_b32 v173, a1
	v_accvgpr_read_b32 v172, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[74:75], v[86:87], a[36:39]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[76:77], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v183, a3
	v_accvgpr_read_b32 v182, a2
	v_accvgpr_read_b32 v181, a1
	v_accvgpr_read_b32 v180, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[78:79], v[82:83], a[40:43]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[80:81], v[84:85], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v187, a3
	v_accvgpr_read_b32 v186, a2
	v_accvgpr_read_b32 v185, a1
	v_accvgpr_read_b32 v184, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[74:75], v[82:83], a[44:47]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[76:77], v[84:85], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v191, a3
	v_accvgpr_read_b32 v190, a2
	v_accvgpr_read_b32 v189, a1
	v_accvgpr_read_b32 v188, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[70:71], v[86:87], a[48:51]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[72:73], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v195, a3
	v_accvgpr_read_b32 v194, a2
	v_accvgpr_read_b32 v193, a1
	v_accvgpr_read_b32 v192, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[66:67], v[86:87], a[52:55]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[68:69], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v199, a3
	v_accvgpr_read_b32 v198, a2
	v_accvgpr_read_b32 v197, a1
	v_accvgpr_read_b32 v196, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[70:71], v[82:83], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[72:73], v[84:85], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v203, a3
	v_accvgpr_read_b32 v202, a2
	v_accvgpr_read_b32 v201, a1
	v_accvgpr_read_b32 v200, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[66:67], v[82:83], a[60:63]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[68:69], v[84:85], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v207, a3
	v_accvgpr_read_b32 v206, a2
	v_accvgpr_read_b32 v205, a1
	v_accvgpr_read_b32 v204, a0
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v106, v146, 1, 0
	ds_read_b128 v[108:111], v106 offset:32768
	ds_read_b128 v[112:115], v106 offset:36864
	v_lshl_add_u32 v107, v145, 1, 0
	ds_read_b128 v[116:119], v107 offset:32768
	; sched_barrier mask(0x000007F6)
	ds_read_b128 v[144:147], v107 offset:36864
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 a[0:3], v[108:109], v[94:95], a[64:67]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[110:111], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v211, a3
	v_accvgpr_read_b32 v210, a2
	v_accvgpr_read_b32 v209, a1
	v_accvgpr_read_b32 v208, a0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 a[0:3], v[112:113], v[94:95], a[68:71]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[114:115], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v215, a3
	v_accvgpr_read_b32 v214, a2
	v_accvgpr_read_b32 v213, a1
	v_accvgpr_read_b32 v212, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[108:109], v[90:91], a[72:75]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[110:111], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v219, a3
	v_accvgpr_read_b32 v218, a2
	v_accvgpr_read_b32 v217, a1
	v_accvgpr_read_b32 v216, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[112:113], v[90:91], a[76:79]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[20:23], v[114:115], v[92:93], a[0:3]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 a[0:3], v[116:117], v[94:95], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[48:51], v[118:119], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[0:3], v[144:145], v[94:95], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[44:47], v[146:147], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[116:117], v[90:91], a[88:91]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[118:119], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v223, a3
	v_accvgpr_read_b32 v222, a2
	v_accvgpr_read_b32 v221, a1
	v_accvgpr_read_b32 v220, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[144:145], v[90:91], a[92:95]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[146:147], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v227, a3
	v_accvgpr_read_b32 v226, a2
	v_accvgpr_read_b32 v225, a1
	v_accvgpr_read_b32 v224, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[108:109], v[86:87], a[96:99]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[110:111], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v231, a3
	v_accvgpr_read_b32 v230, a2
	v_accvgpr_read_b32 v229, a1
	v_accvgpr_read_b32 v228, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[112:113], v[86:87], a[100:103]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[114:115], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v235, a3
	v_accvgpr_read_b32 v234, a2
	v_accvgpr_read_b32 v233, a1
	v_accvgpr_read_b32 v232, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[108:109], v[82:83], a[104:107]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[110:111], v[84:85], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v239, a3
	v_accvgpr_read_b32 v238, a2
	v_accvgpr_read_b32 v237, a1
	v_accvgpr_read_b32 v236, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[112:113], v[82:83], a[108:111]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[114:115], v[84:85], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v243, a3
	v_accvgpr_read_b32 v242, a2
	v_accvgpr_read_b32 v241, a1
	v_accvgpr_read_b32 v240, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[116:117], v[86:87], a[112:115]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[118:119], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v247, a3
	v_accvgpr_read_b32 v246, a2
	v_accvgpr_read_b32 v245, a1
	v_accvgpr_read_b32 v244, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[144:145], v[86:87], a[116:119]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[146:147], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v255, a3
	v_accvgpr_read_b32 v254, a2
	v_accvgpr_read_b32 v253, a1
	v_accvgpr_read_b32 v252, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[116:117], v[82:83], a[120:123]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[118:119], v[84:85], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v129, a3
	v_accvgpr_read_b32 v128, a2
	v_accvgpr_read_b32 v127, a1
	v_accvgpr_read_b32 v126, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[144:145], v[82:83], a[124:127]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[108:111], v[146:147], v[84:85], a[0:3]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v120, v143, 1, 0
	ds_read_b128 v[86:89], v120
	v_lshl_add_u32 v248, v142, 1, 0
	ds_read_b128 v[90:93], v248
	v_lshl_add_u32 v121, v141, 1, 0
	ds_read_b128 v[94:97], v121
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 a[0:3], v[78:79], v[86:87], a[132:135]
	; sched_barrier mask(0x000007F6)
	v_lshl_add_u32 v249, v140, 1, 0
	ds_read_b128 v[140:143], v249
	v_mfma_f32_16x16x16_f16 a[132:135], v[80:81], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[74:75], v[86:87], a[136:139]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[92:95], v[76:77], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 a[0:3], v[78:79], v[90:91], a[140:143]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[96:99], v[80:81], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[74:75], v[90:91], a[148:151]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[76:77], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v85, a3
	v_accvgpr_read_b32 v84, a2
	v_accvgpr_read_b32 v83, a1
	v_accvgpr_read_b32 v82, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[70:71], v[86:87], a[152:155]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[104:107], v[72:73], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[66:67], v[86:87], a[156:159]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[76:79], v[68:69], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[70:71], v[90:91], a[160:163]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[72:75], v[72:73], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[66:67], v[90:91], a[164:167]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[52:55], v[68:69], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 a[0:3], v[78:79], v[94:95], a[168:171]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[80:81], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[74:75], v[94:95], a[172:175]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[100:103], v[76:77], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[0:3], v[78:79], v[140:141], a[176:179]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[68:71], v[80:81], v[142:143], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[74:75], v[140:141], a[180:183]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[180:183], v[76:77], v[142:143], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[70:71], v[94:95], a[128:131]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[120:123], v[72:73], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[66:67], v[94:95], a[144:147]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[144:147], v[68:69], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[70:71], v[140:141], a[184:187]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[184:187], v[72:73], v[142:143], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[66:67], v[140:141], a[188:191]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[40:43], v[68:69], v[142:143], a[0:3]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x16_f16 a[0:3], v[108:109], v[86:87], a[220:223]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[188:191], v[110:111], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[112:113], v[86:87], a[224:227]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[220:223], v[114:115], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[108:109], v[90:91], a[232:235]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[232:235], v[110:111], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[112:113], v[90:91], a[236:239]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[60:63], v[114:115], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[116:117], v[86:87], a[240:243]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[88:91], v[118:119], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[144:145], v[86:87], a[244:247]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[36:39], v[146:147], v[88:89], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[116:117], v[90:91], a[248:251]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[32:35], v[118:119], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[144:145], v[90:91], a[252:255]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[24:27], v[146:147], v[92:93], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[108:109], v[94:95], a[192:195]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[110:111], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[112:113], v[94:95], a[204:207]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[64:67], v[114:115], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[108:109], v[140:141], a[216:219]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[124:127], v[110:111], v[142:143], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[112:113], v[140:141], a[228:231]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[116:119], v[114:115], v[142:143], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[116:117], v[94:95], a[196:199]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[28:31], v[118:119], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[144:145], v[94:95], a[200:203]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[8:11], v[146:147], v[96:97], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[116:117], v[140:141], a[208:211]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[4:7], v[118:119], v[142:143], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[144:145], v[140:141], a[212:215]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[146:147], v[142:143], a[0:3]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v78, v139, 1, 0
	ds_read_b128 v[86:89], v78 offset:32768
	v_lshl_add_u32 v79, v137, 1, 0
	ds_read_b128 v[66:69], v79
	v_accvgpr_write_b32 a56, v98
	v_accvgpr_write_b32 a57, v99
	v_accvgpr_write_b32 a58, v100
	v_accvgpr_write_b32 a59, v101
	v_lshl_add_u32 v77, v138, 1, 0
	ds_read_b128 v[90:93], v78 offset:36864
	ds_read_b128 v[94:97], v77
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 a[80:83], v[86:87], v[66:67], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_lshl_add_u32 v76, v136, 1, 0
	ds_read_b128 v[108:111], v76 offset:32768
	; sched_barrier mask(0x000007F6)
	ds_read_b128 v[112:115], v76 offset:36864
	v_accvgpr_write_b32 a56, v148
	v_accvgpr_write_b32 a57, v149
	v_accvgpr_write_b32 a58, v150
	v_accvgpr_write_b32 a59, v151
	v_mfma_f32_16x16x16_f16 a[240:243], v[88:89], v[68:69], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_lshl_add_u32 v75, v134, 1, 0
	ds_read_b128 v[116:119], v75
	v_lshl_add_u32 v74, v135, 1, 0
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_f16 a[80:83], v[90:91], v[66:67], a[56:59]
	; sched_barrier mask(0x000007F6)
	ds_read_b128 v[134:137], v74
	s_nop 1
	v_accvgpr_write_b32 a56, v152
	v_accvgpr_write_b32 a57, v153
	v_accvgpr_write_b32 a58, v154
	v_accvgpr_write_b32 a59, v155
	v_mfma_f32_16x16x16_f16 a[204:207], v[92:93], v[68:69], a[80:83]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_f16 a[80:83], v[86:87], v[94:95], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[208:211], v[88:89], v[96:97], a[80:83]
	s_nop 1
	v_accvgpr_write_b32 a56, v156
	v_accvgpr_write_b32 a57, v157
	v_accvgpr_write_b32 a58, v158
	v_accvgpr_write_b32 a59, v159
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[80:83], v[90:91], v[94:95], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[212:215], v[92:93], v[96:97], a[80:83]
	s_nop 1
	v_accvgpr_write_b32 a56, v160
	v_accvgpr_write_b32 a57, v161
	v_accvgpr_write_b32 a58, v162
	v_accvgpr_write_b32 a59, v163
	s_waitcnt lgkmcnt(3)
	s_nop 0
	v_mfma_f32_16x16x16_f16 a[80:83], v[108:109], v[66:67], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[216:219], v[110:111], v[68:69], a[80:83]
	s_nop 1
	v_accvgpr_write_b32 a56, v164
	v_accvgpr_write_b32 a57, v165
	v_accvgpr_write_b32 a58, v166
	v_accvgpr_write_b32 a59, v167
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_f32_16x16x16_f16 a[80:83], v[112:113], v[66:67], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[236:239], v[114:115], v[68:69], a[80:83]
	s_nop 1
	v_accvgpr_write_b32 a56, v168
	v_accvgpr_write_b32 a57, v169
	v_accvgpr_write_b32 a58, v170
	v_accvgpr_write_b32 a59, v171
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[80:83], v[108:109], v[94:95], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[224:227], v[110:111], v[96:97], a[80:83]
	s_nop 1
	v_accvgpr_write_b32 a56, v176
	v_accvgpr_write_b32 a57, v177
	v_accvgpr_write_b32 a58, v178
	v_accvgpr_write_b32 a59, v179
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[80:83], v[112:113], v[94:95], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[168:171], v[114:115], v[96:97], a[80:83]
	s_nop 1
	v_accvgpr_write_b32 a56, v172
	v_accvgpr_write_b32 a57, v173
	v_accvgpr_write_b32 a58, v174
	v_accvgpr_write_b32 a59, v175
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(1)
	s_nop 0
	v_mfma_f32_16x16x16_f16 a[80:83], v[86:87], v[116:117], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[56:59], v[88:89], v[118:119], a[80:83]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v101, a59
	v_accvgpr_read_b32 v100, a58
	v_accvgpr_read_b32 v99, a57
	v_accvgpr_read_b32 v98, a56
	v_accvgpr_write_b32 a56, v180
	v_accvgpr_write_b32 a57, v181
	v_accvgpr_write_b32 a58, v182
	v_accvgpr_write_b32 a59, v183
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[80:83], v[90:91], v[116:117], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[56:59], v[92:93], v[118:119], a[80:83]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v153, a59
	v_accvgpr_read_b32 v152, a58
	v_accvgpr_read_b32 v151, a57
	v_accvgpr_read_b32 v150, a56
	v_accvgpr_write_b32 a56, v184
	v_accvgpr_write_b32 a57, v185
	v_accvgpr_write_b32 a58, v186
	v_accvgpr_write_b32 a59, v187
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_f32_16x16x16_f16 a[80:83], v[86:87], v[134:135], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[56:59], v[88:89], v[136:137], a[80:83]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v157, a59
	v_accvgpr_read_b32 v156, a58
	v_accvgpr_read_b32 v155, a57
	v_accvgpr_read_b32 v154, a56
	v_accvgpr_write_b32 a56, v188
	v_accvgpr_write_b32 a57, v189
	v_accvgpr_write_b32 a58, v190
	v_accvgpr_write_b32 a59, v191
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[80:83], v[90:91], v[134:135], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[56:59], v[92:93], v[136:137], a[80:83]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v161, a59
	v_accvgpr_read_b32 v160, a58
	v_accvgpr_read_b32 v159, a57
	v_accvgpr_read_b32 v158, a56
	v_accvgpr_write_b32 a56, v192
	v_accvgpr_write_b32 a57, v193
	v_accvgpr_write_b32 a58, v194
	v_accvgpr_write_b32 a59, v195
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[80:83], v[108:109], v[116:117], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[56:59], v[110:111], v[118:119], a[80:83]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v165, a59
	v_accvgpr_read_b32 v164, a58
	v_accvgpr_read_b32 v163, a57
	v_accvgpr_read_b32 v162, a56
	v_accvgpr_write_b32 a56, v196
	v_accvgpr_write_b32 a57, v197
	v_accvgpr_write_b32 a58, v198
	v_accvgpr_write_b32 a59, v199
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[80:83], v[112:113], v[116:117], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[56:59], v[114:115], v[118:119], a[80:83]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v169, a59
	v_accvgpr_read_b32 v168, a58
	v_accvgpr_read_b32 v167, a57
	v_accvgpr_read_b32 v166, a56
	v_accvgpr_write_b32 a56, v200
	v_accvgpr_write_b32 a57, v201
	v_accvgpr_write_b32 a58, v202
	v_accvgpr_write_b32 a59, v203
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[80:83], v[108:109], v[134:135], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[56:59], v[110:111], v[136:137], a[80:83]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v173, a59
	v_accvgpr_read_b32 v172, a58
	v_accvgpr_read_b32 v171, a57
	v_accvgpr_read_b32 v170, a56
	v_accvgpr_write_b32 a56, v204
	v_accvgpr_write_b32 a57, v205
	v_accvgpr_write_b32 a58, v206
	v_accvgpr_write_b32 a59, v207
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[80:83], v[112:113], v[134:135], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[56:59], v[114:115], v[136:137], a[80:83]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v177, a59
	v_accvgpr_read_b32 v176, a58
	v_accvgpr_read_b32 v175, a57
	v_accvgpr_read_b32 v174, a56
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v73, v132, 1, 0
	ds_read_b128 v[138:141], v73 offset:32768
	v_lshl_add_u32 v72, v133, 1, 0
	ds_read_b128 v[142:145], v72 offset:32768
	v_accvgpr_write_b32 a56, v208
	v_accvgpr_write_b32 a57, v209
	v_accvgpr_write_b32 a58, v210
	v_accvgpr_write_b32 a59, v211
	v_lshl_add_u32 v71, v130, 1, 0
	ds_read_b128 v[146:149], v71 offset:32768
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 a[80:83], v[138:139], v[66:67], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_lshl_add_u32 v70, v131, 1, 0
	ds_read_b128 v[130:133], v70 offset:32768
	s_nop 0
	v_accvgpr_write_b32 a56, v212
	v_accvgpr_write_b32 a57, v213
	v_accvgpr_write_b32 a58, v214
	v_accvgpr_write_b32 a59, v215
	v_mfma_f32_16x16x16_f16 a[200:203], v[140:141], v[68:69], a[80:83]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 a[80:83], v[142:143], v[66:67], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[144:145], v[68:69], a[80:83]
	s_nop 1
	v_accvgpr_write_b32 a56, v216
	v_accvgpr_write_b32 a57, v217
	v_accvgpr_write_b32 a58, v218
	v_accvgpr_write_b32 a59, v219
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[80:83], v[138:139], v[94:95], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[160:163], v[140:141], v[96:97], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[20:23], v[142:143], v[94:95], a[20:23]
	; sched_barrier mask(0x000007F6)
	s_nop 4
	v_accvgpr_write_b32 a80, v228
	v_mfma_f32_16x16x16_f16 a[56:59], v[144:145], v[96:97], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_write_b32 a81, v229
	v_accvgpr_write_b32 a82, v230
	v_accvgpr_write_b32 a83, v231
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 a[20:23], v[146:147], v[66:67], a[48:51]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[48:51], v[148:149], v[68:69], a[20:23]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[20:23], v[130:131], v[66:67], a[44:47]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[44:47], v[132:133], v[68:69], a[20:23]
	; sched_barrier mask(0x000007F6)
	s_nop 5
	v_accvgpr_write_b32 a20, v220
	v_accvgpr_write_b32 a21, v221
	v_accvgpr_write_b32 a22, v222
	v_accvgpr_write_b32 a23, v223
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[20:23], v[146:147], v[94:95], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[112:115], v[148:149], v[96:97], a[20:23]
	; sched_barrier mask(0x000007F6)
	s_nop 5
	v_accvgpr_write_b32 a20, v224
	v_accvgpr_write_b32 a21, v225
	v_accvgpr_write_b32 a22, v226
	v_accvgpr_write_b32 a23, v227
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[20:23], v[130:131], v[94:95], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[20:23], v[132:133], v[96:97], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[128:131], v[138:139], v[116:117], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[176:179], v[140:141], v[118:119], a[128:131]
	s_nop 1
	v_accvgpr_write_b32 a80, v232
	v_accvgpr_write_b32 a81, v233
	v_accvgpr_write_b32 a82, v234
	v_accvgpr_write_b32 a83, v235
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[128:131], v[142:143], v[116:117], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[136:139], v[144:145], v[118:119], a[128:131]
	s_nop 1
	v_accvgpr_write_b32 a80, v236
	v_accvgpr_write_b32 a81, v237
	v_accvgpr_write_b32 a82, v238
	v_accvgpr_write_b32 a83, v239
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[128:131], v[138:139], v[134:135], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[140:143], v[140:141], v[136:137], a[128:131]
	s_nop 1
	v_accvgpr_write_b32 a80, v240
	v_accvgpr_write_b32 a81, v241
	v_accvgpr_write_b32 a82, v242
	v_accvgpr_write_b32 a83, v243
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[128:131], v[142:143], v[134:135], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[152:155], v[144:145], v[136:137], a[128:131]
	s_nop 1
	v_accvgpr_write_b32 a80, v244
	v_accvgpr_write_b32 a81, v245
	v_accvgpr_write_b32 a82, v246
	v_accvgpr_write_b32 a83, v247
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[128:131], v[146:147], v[116:117], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[164:167], v[148:149], v[118:119], a[128:131]
	s_nop 1
	v_accvgpr_write_b32 a80, v252
	v_accvgpr_write_b32 a81, v253
	v_accvgpr_write_b32 a82, v254
	v_accvgpr_write_b32 a83, v255
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[128:131], v[130:131], v[116:117], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[172:175], v[132:133], v[118:119], a[128:131]
	s_nop 1
	v_accvgpr_write_b32 a80, v126
	v_accvgpr_write_b32 a81, v127
	v_accvgpr_write_b32 a82, v128
	v_accvgpr_write_b32 a83, v129
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[128:131], v[146:147], v[134:135], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[148:151], v[148:149], v[136:137], a[128:131]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[108:111], v[130:131], v[134:135], a[108:111]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[156:159], v[132:133], v[136:137], a[108:111]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000000)
	scratch_load_dword v66, off, off offset:4 ; 4-byte Folded Reload
	v_accvgpr_write_b32 a80, v82
	v_accvgpr_write_b32 a81, v83
	v_accvgpr_write_b32 a82, v84
	v_accvgpr_write_b32 a83, v85
	v_lshl_add_u32 v67, v250, 1, 0
	ds_read_b128 v[134:137], v67
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v69, v66, 1, 0
	scratch_load_dword v66, off, off offset:8 ; 4-byte Folded Reload
	ds_read_b128 v[94:97], v69
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[108:111], v[86:87], v[94:95], a[132:135]
	; sched_barrier mask(0x000007F6)
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v68, v66, 1, 0
	ds_read_b128 v[116:119], v68
	v_mfma_f32_16x16x16_f16 a[132:135], v[88:89], v[96:97], a[108:111]
	; sched_barrier mask(0x000007F6)
	v_lshl_add_u32 v66, v122, 1, 0
	ds_read_b128 v[126:129], v66
	v_mfma_f32_16x16x16_f16 a[92:95], v[90:91], v[94:95], a[92:95]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[108:111], v[92:93], v[96:97], a[92:95]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 a[92:95], v[86:87], v[116:117], a[96:99]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[96:99], v[88:89], v[118:119], a[92:95]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[92:95], v[90:91], v[116:117], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[128:131], v[92:93], v[118:119], a[92:95]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[92:95], v[108:109], v[94:95], a[104:107]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[104:107], v[110:111], v[96:97], a[92:95]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[76:79], v[112:113], v[94:95], a[76:79]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[244:247], v[114:115], v[96:97], a[76:79]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[72:75], v[108:109], v[116:117], a[72:75]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[248:251], v[110:111], v[118:119], a[72:75]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[52:55], v[112:113], v[116:117], a[52:55]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[252:255], v[114:115], v[118:119], a[52:55]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[86:87], v[134:135], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[88:89], v[136:137], a[16:19]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v83, a19
	v_accvgpr_read_b32 v82, a18
	v_accvgpr_read_b32 v81, a17
	v_accvgpr_read_b32 v80, a16
	v_mfma_f32_16x16x16_f16 a[16:19], v[90:91], v[134:135], a[100:103]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[228:231], v[92:93], v[136:137], a[16:19]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[16:19], v[86:87], v[126:127], a[68:71]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[88:89], v[128:129], a[16:19]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v87, a19
	v_accvgpr_read_b32 v86, a18
	v_accvgpr_read_b32 v85, a17
	v_accvgpr_read_b32 v84, a16
	v_mfma_f32_16x16x16_f16 a[16:19], v[90:91], v[126:127], a[180:183]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[92:93], v[128:129], a[16:19]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v91, a19
	v_accvgpr_read_b32 v90, a18
	v_accvgpr_read_b32 v89, a17
	v_accvgpr_read_b32 v88, a16
	v_mfma_f32_16x16x16_f16 a[16:19], v[108:109], v[134:135], a[120:123]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[110:111], v[136:137], a[16:19]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v181, a19
	v_accvgpr_read_b32 v180, a18
	v_accvgpr_read_b32 v179, a17
	v_accvgpr_read_b32 v178, a16
	v_mfma_f32_16x16x16_f16 a[16:19], v[112:113], v[134:135], a[144:147]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[192:195], v[114:115], v[136:137], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[108:109], v[126:127], a[184:187]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[196:199], v[110:111], v[128:129], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[112:113], v[126:127], a[40:43]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[184:187], v[114:115], v[128:129], a[16:19]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000000)
	v_mfma_f32_16x16x16_f16 a[16:19], v[138:139], v[94:95], a[188:191]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[120:123], v[140:141], v[96:97], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[142:143], v[94:95], a[220:223]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[220:223], v[144:145], v[96:97], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[138:139], v[116:117], a[232:235]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[140:141], v[118:119], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[40:43], v[142:143], v[116:117], a[60:63]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[80:83], v[144:145], v[118:119], a[40:43]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[40:43], v[146:147], v[94:95], a[88:91]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[68:71], v[148:149], v[96:97], a[40:43]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[36:39], v[130:131], v[94:95], a[36:39]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[72:75], v[132:133], v[96:97], a[36:39]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[32:35], v[146:147], v[116:117], a[32:35]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[76:79], v[148:149], v[118:119], a[32:35]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[24:27], v[130:131], v[116:117], a[24:27]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[88:91], v[132:133], v[118:119], a[24:27]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[138:139], v[134:135], a[12:15]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[144:147], v[140:141], v[136:137], a[12:15]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[142:143], v[134:135], a[64:67]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[180:183], v[144:145], v[136:137], a[12:15]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[138:139], v[126:127], a[124:127]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[92:95], v[140:141], v[128:129], a[12:15]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[142:143], v[126:127], a[116:119]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[232:235], v[144:145], v[128:129], a[12:15]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[146:147], v[134:135], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[100:103], v[148:149], v[136:137], a[12:15]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[8:11], v[130:131], v[134:135], a[8:11]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[116:119], v[132:133], v[136:137], a[8:11]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[4:7], v[146:147], v[126:127], a[4:7]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[148:149], v[128:129], a[4:7]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[130:131], v[126:127], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[4:7], v[132:133], v[128:129], a[0:3]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000000)
	s_barrier
	ds_write_b128 v125, v[58:61]
	ds_write_b128 v125, v[54:57] offset:4096
	ds_write_b128 v125, v[50:53] offset:8192
	ds_write_b128 v125, v[62:65] offset:12288
	ds_write_b128 v125, v[46:49] offset:16384
	ds_write_b128 v125, v[34:37] offset:20480
	ds_write_b128 v125, v[42:45] offset:24576
	ds_write_b128 v125, v[30:33] offset:28672
	ds_write_b128 v125, v[38:41] offset:32768
	ds_write_b128 v123, v[26:29] offset:4096
	ds_write_b128 v123, v[22:25] offset:8192
	ds_write_b128 v123, v[18:21] offset:12288
	ds_write_b128 v123, v[14:17] offset:16384
	ds_write_b128 v123, v[10:13] offset:20480
	ds_write_b128 v123, v[6:9] offset:24576
	ds_write_b128 v123, v[2:5] offset:28672
	s_waitcnt lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[2:5], v124 offset:32768
	ds_read_b128 v[6:9], v1
	ds_read_b128 v[10:13], v124 offset:36864
	ds_read_b128 v[14:17], v1 offset:4096
	ds_read_b128 v[18:21], v0 offset:8192
	; sched_barrier mask(0x000007F6)
	ds_read_b128 v[22:25], v0 offset:12288
	ds_read_b128 v[26:29], v106 offset:32768
	; sched_barrier mask(0x000007F6)
	ds_read_b128 v[30:33], v106 offset:36864
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[2:3], v[6:7], a[240:243]
	; sched_barrier mask(0x000007F6)
	ds_read_b128 v[34:37], v107 offset:32768
	; sched_barrier mask(0x000007F6)
	ds_read_b128 v[38:41], v107 offset:36864
	v_mfma_f32_16x16x16_f16 a[24:27], v[4:5], v[8:9], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x16_f16 a[0:3], v[10:11], v[6:7], a[204:207]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[8:11], v[12:13], v[8:9], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[2:3], v[14:15], a[208:211]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[32:35], v[4:5], v[16:17], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[28:31], v[10:11], v[14:15], a[212:215]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[36:39], v[12:13], v[16:17], a[28:31]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x16_f16 a[28:31], v[18:19], v[6:7], a[216:219]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[40:43], v[20:21], v[8:9], a[28:31]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_f16 a[28:31], v[22:23], v[6:7], a[236:239]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[52:55], v[24:25], v[8:9], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[28:31], v[18:19], v[14:15], a[224:227]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[60:63], v[20:21], v[16:17], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[28:31], v[22:23], v[14:15], a[168:171]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[64:67], v[24:25], v[16:17], a[28:31]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 a[28:31], v[26:27], v[6:7], a[200:203]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[188:191], v[28:29], v[8:9], a[28:31]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 a[28:31], v[30:31], v[6:7], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[200:203], v[32:33], v[8:9], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[28:31], v[26:27], v[14:15], a[160:163]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[160:163], v[28:29], v[16:17], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[28:31], v[30:31], v[14:15], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[204:207], v[32:33], v[16:17], a[28:31]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 a[28:31], v[34:35], v[6:7], a[48:51]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[240:243], v[36:37], v[8:9], a[28:31]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[28:31], v[38:39], v[6:7], a[44:47]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[236:239], v[40:41], v[8:9], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[28:31], v[34:35], v[14:15], a[112:115]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[112:115], v[36:37], v[16:17], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[20:23], v[38:39], v[14:15], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[224:227], v[40:41], v[16:17], a[20:23]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[6:9], v1 offset:8192
	ds_read_b128 v[14:17], v1 offset:12288
	v_accvgpr_write_b32 a0, v98
	v_accvgpr_write_b32 a1, v99
	v_accvgpr_write_b32 a2, v100
	v_accvgpr_write_b32 a3, v101
	s_waitcnt lgkmcnt(1)
	s_nop 0
	v_mfma_f32_16x16x16_f16 a[20:23], v[2:3], v[6:7], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[124:127], v[4:5], v[8:9], a[20:23]
	s_nop 1
	v_accvgpr_write_b32 a0, v150
	v_accvgpr_write_b32 a1, v151
	v_accvgpr_write_b32 a2, v152
	v_accvgpr_write_b32 a3, v153
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[20:23], v[10:11], v[6:7], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[208:211], v[12:13], v[8:9], a[20:23]
	s_nop 1
	v_accvgpr_write_b32 a0, v154
	v_accvgpr_write_b32 a1, v155
	v_accvgpr_write_b32 a2, v156
	v_accvgpr_write_b32 a3, v157
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_f32_16x16x16_f16 a[20:23], v[2:3], v[14:15], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[20:23], v[4:5], v[16:17], a[20:23]
	s_nop 1
	v_accvgpr_write_b32 a0, v158
	v_accvgpr_write_b32 a1, v159
	v_accvgpr_write_b32 a2, v160
	v_accvgpr_write_b32 a3, v161
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[28:31], v[10:11], v[14:15], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[28:31], v[12:13], v[16:17], a[28:31]
	s_nop 1
	v_accvgpr_write_b32 a0, v162
	v_accvgpr_write_b32 a1, v163
	v_accvgpr_write_b32 a2, v164
	v_accvgpr_write_b32 a3, v165
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[44:47], v[18:19], v[6:7], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[168:171], v[20:21], v[8:9], a[44:47]
	s_nop 1
	v_accvgpr_write_b32 a0, v166
	v_accvgpr_write_b32 a1, v167
	v_accvgpr_write_b32 a2, v168
	v_accvgpr_write_b32 a3, v169
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[44:47], v[22:23], v[6:7], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[44:47], v[24:25], v[8:9], a[44:47]
	s_nop 1
	v_accvgpr_write_b32 a0, v170
	v_accvgpr_write_b32 a1, v171
	v_accvgpr_write_b32 a2, v172
	v_accvgpr_write_b32 a3, v173
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[48:51], v[18:19], v[14:15], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[48:51], v[20:21], v[16:17], a[48:51]
	s_nop 1
	v_accvgpr_write_b32 a0, v174
	v_accvgpr_write_b32 a1, v175
	v_accvgpr_write_b32 a2, v176
	v_accvgpr_write_b32 a3, v177
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[56:59], v[22:23], v[14:15], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[56:59], v[24:25], v[16:17], a[56:59]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[26:27], v[6:7], a[176:179]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[212:215], v[28:29], v[8:9], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[30:31], v[6:7], a[136:139]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[136:139], v[32:33], v[8:9], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[26:27], v[14:15], a[140:143]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[140:143], v[28:29], v[16:17], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[30:31], v[14:15], a[152:155]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[152:155], v[32:33], v[16:17], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[34:35], v[6:7], a[164:167]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[164:167], v[36:37], v[8:9], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[38:39], v[6:7], a[172:175]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[172:175], v[40:41], v[8:9], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[34:35], v[14:15], a[148:151]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[148:151], v[36:37], v[16:17], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[38:39], v[14:15], a[156:159]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[156:159], v[40:41], v[16:17], a[84:87]
	; sched_barrier mask(0x000007F6)
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[6:9], v120
	ds_read_b128 v[14:17], v248
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 a[84:87], v[2:3], v[6:7], a[132:135]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[216:219], v[4:5], v[8:9], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[10:11], v[6:7], a[108:111]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[132:135], v[12:13], v[8:9], a[84:87]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[84:87], v[2:3], v[14:15], a[96:99]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[96:99], v[4:5], v[16:17], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[10:11], v[14:15], a[128:131]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[128:131], v[12:13], v[16:17], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[18:19], v[6:7], a[104:107]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[104:107], v[20:21], v[8:9], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[22:23], v[6:7], a[244:247]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[176:179], v[24:25], v[8:9], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[18:19], v[14:15], a[248:251]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[84:87], v[20:21], v[16:17], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[108:111], v[22:23], v[14:15], a[252:255]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[244:247], v[24:25], v[16:17], a[108:111]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[108:111], v[26:27], v[6:7], a[120:123]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[252:255], v[28:29], v[8:9], a[108:111]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[108:111], v[30:31], v[6:7], a[220:223]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[120:123], v[32:33], v[8:9], a[108:111]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[26:27], v[14:15], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[220:223], v[28:29], v[16:17], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[30:31], v[14:15], a[80:83]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[248:251], v[32:33], v[16:17], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[34:35], v[6:7], a[68:71]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[36:37], v[8:9], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[38:39], v[6:7], a[72:75]
	; sched_barrier mask(0x000007F6)
	s_nop 5
	v_accvgpr_read_b32 v119, a3
	v_accvgpr_read_b32 v118, a2
	v_accvgpr_read_b32 v117, a1
	v_accvgpr_read_b32 v116, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[40:41], v[8:9], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[34:35], v[14:15], a[76:79]
	; sched_barrier mask(0x000007F6)
	s_nop 5
	v_accvgpr_read_b32 v237, a3
	v_accvgpr_read_b32 v236, a2
	v_accvgpr_read_b32 v235, a1
	v_accvgpr_read_b32 v234, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[36:37], v[16:17], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[38:39], v[14:15], a[88:91]
	; sched_barrier mask(0x000007F6)
	s_nop 5
	v_accvgpr_read_b32 v233, a3
	v_accvgpr_read_b32 v232, a2
	v_accvgpr_read_b32 v231, a1
	v_accvgpr_read_b32 v230, a0
	v_mfma_f32_16x16x16_f16 a[0:3], v[40:41], v[16:17], a[16:19]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v229, a3
	v_accvgpr_read_b32 v228, a2
	v_accvgpr_read_b32 v227, a1
	v_accvgpr_read_b32 v226, a0
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[6:9], v121
	ds_read_b128 v[14:17], v249
	v_accvgpr_write_b32 a0, v80
	v_accvgpr_write_b32 a1, v81
	v_accvgpr_write_b32 a2, v82
	v_accvgpr_write_b32 a3, v83
	s_waitcnt lgkmcnt(1)
	s_nop 0
	v_mfma_f32_16x16x16_f16 a[16:19], v[2:3], v[6:7], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[68:71], v[4:5], v[8:9], a[16:19]
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_accvgpr_write_b32 a0, v84
	v_accvgpr_write_b32 a1, v85
	v_accvgpr_write_b32 a2, v86
	v_mfma_f32_16x16x16_f16 a[16:19], v[10:11], v[6:7], a[228:231]
	v_accvgpr_write_b32 a3, v87
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[228:231], v[12:13], v[8:9], a[16:19]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[16:19], v[2:3], v[14:15], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[72:75], v[4:5], v[16:17], a[16:19]
	s_nop 1
	v_accvgpr_write_b32 a0, v88
	v_accvgpr_write_b32 a1, v89
	v_accvgpr_write_b32 a2, v90
	v_accvgpr_write_b32 a3, v91
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[16:19], v[10:11], v[14:15], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[76:79], v[12:13], v[16:17], a[16:19]
	s_nop 1
	v_accvgpr_write_b32 a0, v178
	v_accvgpr_write_b32 a1, v179
	v_accvgpr_write_b32 a2, v180
	v_accvgpr_write_b32 a3, v181
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_mfma_f32_16x16x16_f16 a[16:19], v[18:19], v[6:7], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[108:111], v[20:21], v[8:9], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[22:23], v[6:7], a[192:195]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[88:91], v[24:25], v[8:9], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[18:19], v[14:15], a[196:199]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[192:195], v[20:21], v[16:17], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[22:23], v[14:15], a[184:187]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[196:199], v[24:25], v[16:17], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[26:27], v[6:7], a[144:147]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[184:187], v[28:29], v[8:9], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[30:31], v[6:7], a[180:183]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[144:147], v[32:33], v[8:9], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[26:27], v[14:15], a[92:95]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[180:183], v[28:29], v[16:17], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[30:31], v[14:15], a[232:235]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[92:95], v[32:33], v[16:17], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[34:35], v[6:7], a[100:103]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[232:235], v[36:37], v[8:9], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[16:19], v[38:39], v[6:7], a[116:119]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[116:119], v[40:41], v[8:9], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[34:35], v[14:15], a[12:15]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[100:103], v[36:37], v[16:17], a[12:15]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[4:7], v[38:39], v[14:15], a[4:7]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[40:41], v[16:17], a[4:7]
	; sched_barrier mask(0x000007F6)
	s_nop 6
	v_accvgpr_read_b32 v101, a3
	v_accvgpr_read_b32 v100, a2
	v_accvgpr_read_b32 v99, a1
	v_accvgpr_read_b32 v98, a0
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[24:27], v78 offset:32768
	ds_read_b128 v[32:35], v79
	ds_read_b128 v[28:31], v78 offset:36864
	ds_read_b128 v[36:39], v77
	ds_read_b128 v[20:23], v76 offset:32768
	; sched_barrier mask(0x000007F6)
	ds_read_b128 v[16:19], v76 offset:36864
	ds_read_b128 v[46:49], v75
	; sched_barrier mask(0x000007F6)
	ds_read_b128 v[54:57], v74
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x16_f16 a[4:7], v[24:25], v[32:33], a[24:27]
	; sched_barrier mask(0x000007F6)
	ds_read_b128 v[12:15], v73 offset:32768
	; sched_barrier mask(0x000007F6)
	ds_read_b128 v[8:11], v72 offset:32768
	ds_read_b128 v[4:7], v71 offset:32768
	; sched_barrier mask(0x000007F6)
	ds_read_b128 v[0:3], v70 offset:32768
	ds_read_b128 v[218:221], v69
	v_mfma_f32_16x16x16_f16 a[0:3], v[26:27], v[34:35], a[4:7]
	; sched_barrier mask(0x000007F6)
	ds_read_b128 v[222:225], v68
	ds_read_b128 v[252:255], v67
	s_mul_i32 s0, s9, s13
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_16x16x16_f16 a[4:7], v[28:29], v[32:33], a[8:11]
	; sched_barrier mask(0x000007F6)
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s6, s0
	v_mfma_f32_16x16x16_f16 a[4:7], v[30:31], v[34:35], a[4:7]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v96, a0
	v_accvgpr_read_b32 v97, a1
	v_accvgpr_read_b32 v109, a2
	s_waitcnt lgkmcnt(11)
	v_mfma_f32_16x16x16_f16 a[8:11], v[24:25], v[36:37], a[32:35]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v108, a3
	v_cvt_f16_f32_e32 v96, v96
	v_cvt_f16_f32_e32 v109, v109
	v_mfma_f32_16x16x16_f16 a[12:15], v[26:27], v[38:39], a[8:11]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v93, a4
	v_accvgpr_read_b32 v90, a5
	v_accvgpr_read_b32 v94, a6
	v_mfma_f32_16x16x16_f16 a[8:11], v[28:29], v[36:37], a[36:39]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v95, a7
	v_cvt_f16_f32_e32 v97, v97
	s_addc_u32 s3, s7, s1
	v_mfma_f32_16x16x16_f16 a[24:27], v[30:31], v[38:39], a[8:11]
	v_accvgpr_read_b32 v86, a12
	v_accvgpr_read_b32 v84, a13
	v_accvgpr_read_b32 v91, a14
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x16_f16 a[8:11], v[20:21], v[32:33], a[40:43]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v92, a15
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[0:1], s[8:9], 1
	v_mfma_f32_16x16x16_f16 a[8:11], v[22:23], v[34:35], a[8:11]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v81, a24
	v_accvgpr_read_b32 v85, a25
	v_accvgpr_read_b32 v88, a26
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x16_f16 a[16:19], v[16:17], v[32:33], a[52:55]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v89, a27
	v_cvt_f16_f32_e32 v86, v86
	v_cvt_f16_f32_e32 v91, v91
	v_mfma_f32_16x16x16_f16 a[32:35], v[18:19], v[34:35], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v82, a8
	v_accvgpr_read_b32 v80, a9
	v_accvgpr_read_b32 v83, a10
	v_mfma_f32_16x16x16_f16 a[16:19], v[20:21], v[36:37], a[60:63]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v87, a11
	v_cvt_f16_f32_e32 v92, v92
	v_cvt_f16_f32_e32 v84, v84
	v_mfma_f32_16x16x16_f16 a[40:43], v[22:23], v[38:39], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v76, a32
	v_accvgpr_read_b32 v72, a33
	v_accvgpr_read_b32 v78, a34
	v_mfma_f32_16x16x16_f16 a[16:19], v[16:17], v[36:37], a[64:67]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v79, a35
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	v_mfma_f32_16x16x16_f16 a[64:67], v[18:19], v[38:39], a[16:19]
	v_accvgpr_read_b32 v64, a40
	v_accvgpr_read_b32 v65, a41
	v_accvgpr_read_b32 v70, a42
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x16_f16 a[16:19], v[24:25], v[46:47], a[124:127]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v71, a43
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x27000
	v_mfma_f32_16x16x16_f16 a[60:63], v[26:27], v[48:49], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v73, a64
	v_accvgpr_read_b32 v74, a65
	v_accvgpr_read_b32 v75, a66
	v_mfma_f32_16x16x16_f16 a[16:19], v[28:29], v[46:47], a[208:211]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v77, a67
	s_mov_b32 s2, 0x7ffffffe
	v_cvt_f16_f32_e32 v88, v88
	v_mfma_f32_16x16x16_f16 a[36:39], v[30:31], v[48:49], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_cvt_f16_f32_e32 v89, v89
	s_lshl_b32 s4, s13, 5
	v_cvt_f16_f32_e32 v83, v83
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x16_f16 a[16:19], v[24:25], v[54:55], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_pack_b32_f16 v89, v88, v89
	v_cvt_f16_f32_e32 v87, v87
	v_cvt_f16_f32_e32 v70, v70
	v_mfma_f32_16x16x16_f16 a[16:19], v[26:27], v[56:57], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v40, a39
	v_cvt_f16_f32_e32 v71, v71
	v_cvt_f16_f32_e32 v78, v78
	v_mfma_f32_16x16x16_f16 a[20:23], v[28:29], v[54:55], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_cvt_f16_f32_e32 v79, v79
	v_cvt_f16_f32_e32 v64, v64
	v_mfma_f32_16x16x16_f16 a[124:127], v[30:31], v[56:57], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v44, a16
	v_accvgpr_read_b32 v45, a17
	v_mfma_f32_16x16x16_f16 a[20:23], v[20:21], v[46:47], a[168:171]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[208:211], v[22:23], v[48:49], a[20:23]
	; sched_barrier mask(0x000007F6)
	s_nop 2
	v_accvgpr_read_b32 v50, a124
	v_accvgpr_read_b32 v51, a125
	v_accvgpr_read_b32 v52, a126
	v_mfma_f32_16x16x16_f16 a[20:23], v[16:17], v[46:47], a[44:47]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v53, a127
	v_mfma_f32_16x16x16_f16 a[168:171], v[18:19], v[48:49], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v41, a209
	v_accvgpr_read_b32 v42, a210
	v_accvgpr_read_b32 v43, a211
	v_mfma_f32_16x16x16_f16 a[20:23], v[20:21], v[54:55], a[48:51]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[52:55], v[22:23], v[56:57], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[20:23], v[16:17], v[54:55], a[56:59]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x16_f16 a[28:31], v[12:13], v[32:33], a[188:191]
	s_nop 3
	v_accvgpr_read_b32 v58, a54
	v_accvgpr_read_b32 v59, a55
	v_mfma_f32_16x16x16_f16 a[20:23], v[18:19], v[56:57], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[48:51], v[14:15], v[34:35], a[28:31]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x16_f16 a[28:31], v[8:9], v[32:33], a[200:203]
	; sched_barrier mask(0x000007F6)
	s_nop 3
	v_accvgpr_read_b32 v60, a20
	v_mfma_f32_16x16x16_f16 a[200:203], v[10:11], v[34:35], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v61, a21
	v_accvgpr_read_b32 v62, a22
	v_accvgpr_read_b32 v63, a23
	v_mfma_f32_16x16x16_f16 a[28:31], v[12:13], v[36:37], a[160:163]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v133, a48
	v_accvgpr_read_b32 v146, a49
	v_accvgpr_read_b32 v147, a50
	v_mfma_f32_16x16x16_f16 a[160:163], v[14:15], v[38:39], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v148, a51
	v_accvgpr_read_b32 v159, a200
	v_accvgpr_read_b32 v160, a201
	v_mfma_f32_16x16x16_f16 a[28:31], v[8:9], v[36:37], a[204:207]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v161, a202
	v_accvgpr_read_b32 v174, a203
	v_mfma_f32_16x16x16_f16 a[56:59], v[10:11], v[38:39], a[28:31]
	v_accvgpr_read_b32 v175, a160
	v_accvgpr_read_b32 v176, a161
	v_accvgpr_read_b32 v188, a162
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_f16 a[28:31], v[4:5], v[32:33], a[240:243]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v189, a163
	v_cvt_f16_f32_e32 v122, v188
	v_mfma_f32_16x16x16_f16 a[28:31], v[6:7], v[34:35], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v190, a56
	v_accvgpr_read_b32 v200, a57
	v_accvgpr_read_b32 v201, a58
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 a[44:47], v[0:1], v[32:33], a[236:239]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v202, a59
	v_accvgpr_read_b32 v32, a60
	v_accvgpr_read_b32 v33, a61
	v_mfma_f32_16x16x16_f16 a[188:191], v[2:3], v[34:35], a[44:47]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v168, a28
	v_accvgpr_read_b32 v169, a29
	v_accvgpr_read_b32 v170, a30
	v_mfma_f32_16x16x16_f16 a[44:47], v[4:5], v[36:37], a[112:115]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v180, a31
	v_accvgpr_read_b32 v34, a62
	v_accvgpr_read_b32 v35, a63
	v_mfma_f32_16x16x16_f16 a[204:207], v[6:7], v[38:39], a[44:47]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v181, a188
	v_accvgpr_read_b32 v182, a189
	v_accvgpr_read_b32 v194, a190
	v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[36:37], a[224:227]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v36, a36
	v_accvgpr_read_b32 v195, a191
	v_cvt_f16_f32_e32 v32, v32
	v_mfma_f32_16x16x16_f16 a[44:47], v[2:3], v[38:39], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v38, a37
	v_accvgpr_read_b32 v39, a38
	v_accvgpr_read_b32 v196, a204
	v_mfma_f32_16x16x16_f16 a[0:3], v[12:13], v[46:47], a[212:215]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v206, a205
	v_accvgpr_read_b32 v207, a206
	v_accvgpr_read_b32 v208, a207
	v_mfma_f32_16x16x16_f16 a[80:83], v[14:15], v[48:49], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v212, a44
	v_accvgpr_read_b32 v213, a45
	v_accvgpr_read_b32 v214, a46
	v_mfma_f32_16x16x16_f16 a[0:3], v[8:9], v[46:47], a[136:139]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v216, a47
	ds_read_b128 a[44:47], v66
	v_accvgpr_read_b32 v37, a208
	v_mfma_f32_16x16x16_f16 a[12:15], v[10:11], v[48:49], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v68, a80
	v_accvgpr_read_b32 v69, a81
	v_accvgpr_read_b32 v106, a82
	v_mfma_f32_16x16x16_f16 a[0:3], v[12:13], v[54:55], a[140:143]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v107, a83
	v_mfma_f32_16x16x16_f16 a[8:11], v[14:15], v[56:57], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 0
	v_accvgpr_read_b32 v67, a12
	v_accvgpr_read_b32 v110, a13
	v_accvgpr_read_b32 v137, a14
	v_mfma_f32_16x16x16_f16 a[0:3], v[8:9], v[54:55], a[152:155]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v138, a15
	v_mfma_f32_16x16x16_f16 a[40:43], v[10:11], v[56:57], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v139, a8
	v_accvgpr_read_b32 v140, a9
	v_accvgpr_read_b32 v141, a10
	v_mfma_f32_16x16x16_f16 a[0:3], v[4:5], v[46:47], a[164:167]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v142, a11
	v_mfma_f32_16x16x16_f16 a[64:67], v[6:7], v[48:49], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 0
	v_accvgpr_read_b32 v149, a40
	v_accvgpr_read_b32 v150, a41
	v_accvgpr_read_b32 v151, a42
	v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[46:47], a[172:175]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v46, a18
	v_accvgpr_read_b32 v155, a43
	v_accvgpr_read_b32 v47, a168
	v_mfma_f32_16x16x16_f16 a[112:115], v[2:3], v[48:49], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v49, a19
	v_accvgpr_read_b32 v48, a169
	v_accvgpr_read_b32 v124, a64
	v_mfma_f32_16x16x16_f16 a[0:3], v[4:5], v[54:55], a[148:151]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v125, a65
	v_accvgpr_read_b32 v129, a66
	v_accvgpr_read_b32 v130, a67
	v_mfma_f32_16x16x16_f16 a[124:127], v[6:7], v[56:57], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v131, a112
	v_accvgpr_read_b32 v134, a113
	v_accvgpr_read_b32 v135, a114
	v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[54:55], a[156:159]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v54, a170
	v_accvgpr_read_b32 v55, a171
	v_accvgpr_read_b32 v136, a115
	v_mfma_f32_16x16x16_f16 a[136:139], v[2:3], v[56:57], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v56, a52
	v_accvgpr_read_b32 v57, a53
	v_accvgpr_read_b32 v152, a124
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 a[0:3], v[24:25], v[218:219], a[216:219]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v153, a125
	v_accvgpr_read_b32 v154, a126
	v_accvgpr_read_b32 v162, a127
	v_mfma_f32_16x16x16_f16 a[0:3], v[26:27], v[220:221], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v163, a136
	v_accvgpr_read_b32 v164, a137
	v_accvgpr_read_b32 v183, a138
	v_mfma_f32_16x16x16_f16 a[4:7], v[28:29], v[218:219], a[132:135]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v184, a139
	v_mfma_f32_16x16x16_f16 a[4:7], v[30:31], v[220:221], a[4:7]
	; sched_barrier mask(0x000007F6)
	s_nop 0
	v_accvgpr_read_b32 v132, a0
	v_accvgpr_read_b32 v143, a1
	v_accvgpr_read_b32 v144, a2
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 a[16:19], v[24:25], v[222:223], a[96:99]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v145, a3
	v_mfma_f32_16x16x16_f16 a[16:19], v[26:27], v[224:225], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v156, a4
	v_accvgpr_read_b32 v157, a5
	v_accvgpr_read_b32 v158, a6
	v_mfma_f32_16x16x16_f16 a[20:23], v[28:29], v[222:223], a[128:131]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v171, a7
	v_mfma_f32_16x16x16_f16 a[20:23], v[30:31], v[224:225], a[20:23]
	; sched_barrier mask(0x000007F6)
	s_nop 0
	v_accvgpr_read_b32 v172, a16
	v_accvgpr_read_b32 v173, a17
	v_accvgpr_read_b32 v185, a18
	v_mfma_f32_16x16x16_f16 a[24:27], v[20:21], v[218:219], a[104:107]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v186, a19
	v_mfma_f32_16x16x16_f16 a[24:27], v[22:23], v[220:221], a[24:27]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v187, a20
	v_accvgpr_read_b32 v197, a21
	v_accvgpr_read_b32 v198, a22
	v_mfma_f32_16x16x16_f16 a[28:31], v[16:17], v[218:219], a[176:179]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v199, a23
	v_mfma_f32_16x16x16_f16 a[28:31], v[18:19], v[220:221], a[28:31]
	; sched_barrier mask(0x000007F6)
	s_nop 0
	v_accvgpr_read_b32 v165, a24
	v_accvgpr_read_b32 v166, a25
	v_accvgpr_read_b32 v167, a26
	v_mfma_f32_16x16x16_f16 a[32:35], v[20:21], v[222:223], a[84:87]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v177, a27
	v_mfma_f32_16x16x16_f16 a[32:35], v[22:23], v[224:225], a[32:35]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v178, a28
	v_accvgpr_read_b32 v179, a29
	v_accvgpr_read_b32 v191, a30
	v_mfma_f32_16x16x16_f16 a[36:39], v[16:17], v[222:223], a[244:247]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v192, a31
	v_mfma_f32_16x16x16_f16 a[36:39], v[18:19], v[224:225], a[36:39]
	; sched_barrier mask(0x000007F6)
	s_nop 0
	v_accvgpr_read_b32 v193, a32
	v_accvgpr_read_b32 v203, a33
	v_accvgpr_read_b32 v204, a34
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 a[48:51], v[24:25], v[252:253], a[68:71]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v205, a35
	v_accvgpr_write_b32 a32, v234
	v_accvgpr_write_b32 a33, v235
	v_mfma_f32_16x16x16_f16 a[12:15], v[26:27], v[254:255], a[48:51]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_write_b32 a34, v236
	v_accvgpr_write_b32 a35, v237
	v_accvgpr_read_b32 v209, a36
	v_mfma_f32_16x16x16_f16 a[8:11], v[28:29], v[252:253], a[228:231]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v210, a37
	v_accvgpr_read_b32 v211, a38
	v_accvgpr_read_b32 v215, a39
	v_mfma_f32_16x16x16_f16 a[8:11], v[30:31], v[254:255], a[8:11]
	; sched_barrier mask(0x000007F6)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[40:43], v[24:25], a[44:45], a[72:75]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[40:43], v[26:27], a[46:47], a[40:43]
	; sched_barrier mask(0x000007F6)
	s_nop 3
	v_accvgpr_read_b32 v24, a11
	v_mfma_f32_16x16x16_f16 a[48:51], v[28:29], a[44:45], a[76:79]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[48:51], v[30:31], a[46:47], a[48:51]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v28, a40
	v_mfma_f32_16x16x16_f16 a[52:55], v[20:21], v[252:253], a[108:111]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v29, a41
	v_accvgpr_read_b32 v30, a42
	v_accvgpr_read_b32 v112, a43
	v_mfma_f32_16x16x16_f16 a[52:55], v[22:23], v[254:255], a[52:55]
	; sched_barrier mask(0x000007F6)
	s_nop 0
	v_accvgpr_read_b32 v113, a48
	v_accvgpr_read_b32 v114, a49
	v_mfma_f32_16x16x16_f16 a[0:3], v[16:17], v[252:253], a[88:91]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[0:3], v[18:19], v[254:255], a[0:3]
	; sched_barrier mask(0x000007F6)
	s_nop 1
	v_accvgpr_read_b32 v25, a53
	v_accvgpr_read_b32 v26, a54
	v_accvgpr_read_b32 v27, a55
	v_mfma_f32_16x16x16_f16 a[4:7], v[20:21], a[44:45], a[192:195]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v20, a8
	v_accvgpr_read_b32 v21, a52
	v_mfma_f32_16x16x16_f16 a[4:7], v[22:23], a[46:47], a[4:7]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v22, a9
	v_accvgpr_read_b32 v23, a10
	v_accvgpr_read_b32 v31, a0
	v_mfma_f32_16x16x16_f16 a[16:19], v[16:17], a[44:45], a[196:199]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v16, a12
	v_accvgpr_read_b32 v17, a13
	v_accvgpr_read_b32 v66, a1
	v_mfma_f32_16x16x16_f16 a[16:19], v[18:19], a[46:47], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v18, a14
	v_accvgpr_read_b32 v19, a15
	v_accvgpr_read_b32 v111, a2
	v_mfma_f32_16x16x16_f16 a[20:23], v[12:13], v[218:219], a[252:255]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v115, a3
	v_accvgpr_write_b32 a0, v230
	v_accvgpr_write_b32 a1, v231
	v_mfma_f32_16x16x16_f16 a[20:23], v[14:15], v[220:221], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_write_b32 a2, v232
	v_accvgpr_write_b32 a3, v233
	v_accvgpr_read_b32 v120, a6
	v_mfma_f32_16x16x16_f16 a[24:27], v[8:9], v[218:219], a[120:123]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v121, a7
	v_accvgpr_read_b32 v123, a16
	v_accvgpr_read_b32 v126, a17
	v_mfma_f32_16x16x16_f16 a[24:27], v[10:11], v[220:221], a[24:27]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v127, a18
	v_accvgpr_read_b32 v128, a19
	v_cvt_f16_f32_e32 v16, v16
	v_mfma_f32_16x16x16_f16 a[28:31], v[12:13], v[222:223], a[220:223]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[14:15], v[224:225], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[8:11], v[8:9], v[222:223], a[248:251]
	; sched_barrier mask(0x000007F6)
	s_nop 4
	v_accvgpr_write_b32 a28, v116
	v_accvgpr_write_b32 a29, v117
	v_accvgpr_write_b32 a30, v118
	v_accvgpr_write_b32 a31, v119
	v_mfma_f32_16x16x16_f16 a[8:11], v[10:11], v[224:225], a[8:11]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v116, a4
	v_accvgpr_read_b32 v117, a5
	v_accvgpr_write_b32 a4, v226
	v_mfma_f32_16x16x16_f16 a[28:31], v[4:5], v[218:219], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_write_b32 a5, v227
	v_accvgpr_write_b32 a6, v228
	v_accvgpr_write_b32 a7, v229
	v_mfma_f32_16x16x16_f16 a[28:31], v[6:7], v[220:221], a[28:31]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v229, a12
	v_accvgpr_read_b32 v233, a13
	v_accvgpr_read_b32 v234, a14
	v_mfma_f32_16x16x16_f16 a[32:35], v[0:1], v[218:219], a[32:35]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v235, a15
	v_accvgpr_read_b32 v239, a8
	v_accvgpr_read_b32 v240, a9
	v_mfma_f32_16x16x16_f16 a[32:35], v[2:3], v[220:221], a[32:35]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v241, a10
	v_accvgpr_read_b32 v245, a11
	v_accvgpr_read_b32 v219, a20
	v_mfma_f32_16x16x16_f16 a[0:3], v[4:5], v[222:223], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v220, a21
	v_accvgpr_read_b32 v221, a22
	v_accvgpr_read_b32 v227, a26
	v_mfma_f32_16x16x16_f16 a[0:3], v[6:7], v[224:225], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v228, a27
	v_accvgpr_read_b32 v226, a29
	v_accvgpr_read_b32 v230, a30
	v_mfma_f32_16x16x16_f16 a[4:7], v[0:1], v[222:223], a[4:7]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v222, a23
	v_accvgpr_read_b32 v223, a24
	v_accvgpr_read_b32 v231, a31
	v_mfma_f32_16x16x16_f16 a[4:7], v[2:3], v[224:225], a[4:7]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v242, a0
	v_accvgpr_read_b32 v243, a1
	v_accvgpr_read_b32 v244, a2
	v_mfma_f32_16x16x16_f16 a[16:19], v[12:13], v[252:253], a[184:187]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v246, a3
	v_accvgpr_read_b32 v224, a25
	v_accvgpr_read_b32 v225, a28
	v_mfma_f32_16x16x16_f16 a[16:19], v[14:15], v[254:255], a[16:19]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v247, a4
	v_accvgpr_read_b32 v248, a5
	v_accvgpr_read_b32 v249, a6
	v_mfma_f32_16x16x16_f16 a[12:15], v[8:9], v[252:253], a[144:147]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v250, a7
	v_accvgpr_read_b32 v232, a32
	v_accvgpr_read_b32 v236, a33
	v_mfma_f32_16x16x16_f16 a[12:15], v[10:11], v[254:255], a[12:15]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v237, a34
	v_accvgpr_read_b32 v238, a35
	v_accvgpr_read_b32 v118, a50
	v_mfma_f32_16x16x16_f16 a[8:11], v[12:13], a[44:45], a[180:183]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v119, a51
	v_mfma_f32_16x16x16_f16 a[8:11], v[14:15], a[46:47], a[8:11]
	; sched_barrier mask(0x000007F6)
	s_nop 0
	v_accvgpr_read_b32 v12, a12
	v_accvgpr_read_b32 v13, a13
	v_accvgpr_read_b32 v14, a14
	v_mfma_f32_16x16x16_f16 a[20:23], v[8:9], a[44:45], a[92:95]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v15, a15
	v_accvgpr_read_b32 v8, a16
	v_accvgpr_read_b32 v9, a17
	v_mfma_f32_16x16x16_f16 a[20:23], v[10:11], a[46:47], a[20:23]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v217, a8
	v_accvgpr_read_b32 v218, a9
	v_accvgpr_read_b32 v10, a18
	v_mfma_f32_16x16x16_f16 a[0:3], v[4:5], v[252:253], a[232:235]
	; sched_barrier mask(0x000007F6)
	v_accvgpr_read_b32 v11, a19
	v_cvt_f16_f32_e32 v8, v8
	v_cvt_f16_f32_e32 v9, v9
	v_mfma_f32_16x16x16_f16 a[0:3], v[6:7], v[254:255], a[0:3]
	; sched_barrier mask(0x000007F6)
	v_cvt_f16_f32_e32 v10, v10
	v_cvt_f16_f32_e32 v11, v11
	v_cvt_f16_f32_e32 v12, v12
	v_mfma_f32_16x16x16_f16 a[4:7], v[0:1], v[252:253], a[116:119]
	; sched_barrier mask(0x000007F6)
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v14, v14
	v_cvt_f16_f32_e32 v15, v15
	v_mfma_f32_16x16x16_f16 a[4:7], v[2:3], v[254:255], a[4:7]
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[4:5], a[44:45], a[100:103]
	v_accvgpr_read_b32 v4, a10
	v_accvgpr_read_b32 v5, a11
	v_accvgpr_write_b32 a8, v98
	v_accvgpr_write_b32 a9, v99
	v_accvgpr_write_b32 a10, v100
	v_accvgpr_write_b32 a11, v101
	; sched_barrier mask(0x000007F6)
	v_mfma_f32_16x16x16_f16 a[12:15], v[6:7], a[46:47], a[12:15]
	; sched_barrier mask(0x000007F6)
	v_cvt_f16_f32_e32 v6, v108
	v_cvt_f16_f32_e32 v108, v175
	v_or_b32_e32 v98, 0x80, v104
	v_mfma_f32_16x16x16_f16 a[8:11], v[0:1], a[44:45], a[8:11]
	; sched_barrier mask(0x000007F6)
	v_pack_b32_f16 v7, v109, v6
	v_pack_b32_f16 v6, v96, v97
	v_cvt_f16_f32_e32 v97, v174
	v_mfma_f32_16x16x16_f16 a[8:11], v[2:3], a[46:47], a[8:11]
	; sched_barrier mask(0x00000406)
	scratch_load_dword v1, off, off         ; 4-byte Folded Reload
	; sched_barrier mask(0x00000406)
	v_cvt_f16_f32_e32 v2, v94
	v_cvt_f16_f32_e32 v3, v95
	v_pack_b32_f16 v95, v91, v92
	v_pack_b32_f16 v94, v86, v84
	v_cvt_f16_f32_e32 v86, v81
	v_cvt_f16_f32_e32 v92, v85
	v_pack_b32_f16 v3, v2, v3
	v_cvt_f16_f32_e32 v109, v176
	v_mov_b32_e32 v99, v98
	v_pack_b32_f16 v88, v86, v92
	v_accvgpr_read_b32 v0, a20
	v_cvt_f16_f32_e32 v0, v0
	v_accvgpr_read_b32 v81, a5
	v_accvgpr_read_b32 v85, a6
	v_accvgpr_read_b32 v86, a7
	; sched_barrier mask(0x000007F6)
	s_waitcnt vmcnt(0)
	v_mul_lo_u32 v96, s13, v1
	v_add_lshl_u32 v1, v96, v104, 1
	buffer_store_dwordx2 v[6:7], v1, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v1, v93
	v_cvt_f16_f32_e32 v6, v90
	v_add_u32_e32 v91, s4, v96
	v_add_lshl_u32 v84, v91, v104, 1
	v_add_lshl_u32 v92, v91, v105, 1
	v_pack_b32_f16 v2, v1, v6
	v_add_lshl_u32 v6, v96, v105, 1
	buffer_store_dwordx2 v[2:3], v6, s[0:3], 0 offen
	buffer_store_dwordx2 v[94:95], v84, s[0:3], 0 offen
	buffer_store_dwordx2 v[88:89], v92, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v89, v82
	v_cvt_f16_f32_e32 v92, v80
	v_pack_b32_f16 v95, v83, v87
	v_add_lshl_u32 v87, v96, v103, 1
	v_accvgpr_read_b32 v7, a21
	v_pack_b32_f16 v94, v89, v92
	v_cvt_f16_f32_e32 v89, v76
	v_cvt_f16_f32_e32 v92, v72
	buffer_store_dwordx2 v[94:95], v87, s[0:3], 0 offen
	v_pack_b32_f16 v95, v78, v79
	v_add_lshl_u32 v79, v96, v102, 1
	v_pack_b32_f16 v94, v89, v92
	v_cvt_f16_f32_e32 v89, v65
	v_pack_b32_f16 v65, v70, v71
	v_cvt_f16_f32_e32 v70, v73
	v_cvt_f16_f32_e32 v73, v74
	v_cvt_f16_f32_e32 v71, v75
	v_cvt_f16_f32_e32 v74, v77
	v_pack_b32_f16 v64, v64, v89
	v_add_lshl_u32 v75, v91, v103, 1
	v_pack_b32_f16 v70, v70, v73
	v_pack_b32_f16 v71, v71, v74
	v_add_lshl_u32 v73, v91, v102, 1
	buffer_store_dwordx2 v[94:95], v79, s[0:3], 0 offen
	buffer_store_dwordx2 v[64:65], v75, s[0:3], 0 offen
	buffer_store_dwordx2 v[70:71], v73, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v70, v133
	v_cvt_f16_f32_e32 v73, v146
	v_cvt_f16_f32_e32 v71, v147
	v_cvt_f16_f32_e32 v74, v148
	v_cvt_f16_f32_e32 v75, v159
	v_cvt_f16_f32_e32 v77, v160
	v_cvt_f16_f32_e32 v95, v161
	v_cvt_f16_f32_e32 v133, v189
	v_cvt_f16_f32_e32 v146, v190
	v_cvt_f16_f32_e32 v147, v200
	v_cvt_f16_f32_e32 v148, v201
	v_cvt_f16_f32_e32 v159, v202
	v_pack_b32_f16 v71, v71, v74
	v_pack_b32_f16 v70, v70, v73
	v_add_lshl_u32 v73, v96, v98, 1
	v_or_b32_e32 v74, 0xa0, v104
	buffer_store_dwordx2 v[70:71], v73, s[0:3], 0 offen
	v_pack_b32_f16 v71, v95, v97
	v_pack_b32_f16 v70, v75, v77
	v_add_lshl_u32 v73, v96, v74, 1
	buffer_store_dwordx2 v[70:71], v73, s[0:3], 0 offen
	v_pack_b32_f16 v71, v122, v133
	v_pack_b32_f16 v70, v108, v109
	v_add_lshl_u32 v73, v91, v98, 1
	buffer_store_dwordx2 v[70:71], v73, s[0:3], 0 offen
	v_pack_b32_f16 v71, v148, v159
	v_pack_b32_f16 v70, v146, v147
	v_add_lshl_u32 v73, v91, v74, 1
	v_mov_b32_e32 v98, v74
	buffer_store_dwordx2 v[70:71], v73, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v71, v170
	v_cvt_f16_f32_e32 v74, v180
	v_cvt_f16_f32_e32 v70, v168
	v_cvt_f16_f32_e32 v73, v169
	v_cvt_f16_f32_e32 v75, v181
	v_cvt_f16_f32_e32 v77, v182
	v_cvt_f16_f32_e32 v95, v194
	v_cvt_f16_f32_e32 v97, v195
	v_cvt_f16_f32_e32 v108, v196
	v_cvt_f16_f32_e32 v109, v206
	v_cvt_f16_f32_e32 v122, v207
	v_cvt_f16_f32_e32 v133, v208
	v_cvt_f16_f32_e32 v146, v212
	v_cvt_f16_f32_e32 v147, v213
	v_cvt_f16_f32_e32 v148, v214
	v_cvt_f16_f32_e32 v159, v216
	v_pack_b32_f16 v71, v71, v74
	v_or_b32_e32 v74, 0xc0, v104
	v_pack_b32_f16 v70, v70, v73
	v_add_lshl_u32 v73, v96, v74, 1
	buffer_store_dwordx2 v[70:71], v73, s[0:3], 0 offen
	v_pack_b32_f16 v71, v95, v97
	v_pack_b32_f16 v70, v75, v77
	v_add_lshl_u32 v73, v96, v251, 1
	buffer_store_dwordx2 v[70:71], v73, s[0:3], 0 offen
	v_pack_b32_f16 v71, v122, v133
	v_pack_b32_f16 v70, v108, v109
	v_add_lshl_u32 v73, v91, v74, 1
	buffer_store_dwordx2 v[70:71], v73, s[0:3], 0 offen
	v_pack_b32_f16 v71, v148, v159
	v_pack_b32_f16 v70, v146, v147
	v_add_lshl_u32 v73, v91, v251, 1
	buffer_store_dwordx2 v[70:71], v73, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v70, v33
	v_cvt_f16_f32_e32 v33, v34
	v_cvt_f16_f32_e32 v34, v35
	v_cvt_f16_f32_e32 v35, v36
	v_cvt_f16_f32_e32 v36, v38
	v_cvt_f16_f32_e32 v38, v39
	v_cvt_f16_f32_e32 v39, v40
	v_cvt_f16_f32_e32 v40, v44
	v_cvt_f16_f32_e32 v44, v45
	v_cvt_f16_f32_e32 v45, v46
	v_cvt_f16_f32_e32 v46, v49
	v_add_u32_e32 v79, s4, v91
	v_cvt_f16_f32_e32 v49, v50
	v_cvt_f16_f32_e32 v50, v51
	v_cvt_f16_f32_e32 v51, v52
	v_cvt_f16_f32_e32 v52, v53
	v_pack_b32_f16 v33, v33, v34
	v_pack_b32_f16 v32, v32, v70
	v_add_lshl_u32 v34, v79, v104, 1
	v_add_u32_e32 v92, s4, v79
	; sched_barrier mask(0x00000406)
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v38, v39
	v_pack_b32_f16 v32, v35, v36
	v_add_lshl_u32 v34, v79, v105, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v45, v46
	v_pack_b32_f16 v32, v40, v44
	v_add_lshl_u32 v34, v92, v104, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v51, v52
	v_pack_b32_f16 v32, v49, v50
	v_add_lshl_u32 v34, v92, v105, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v32, v37
	v_cvt_f16_f32_e32 v34, v41
	v_cvt_f16_f32_e32 v33, v42
	v_cvt_f16_f32_e32 v35, v43
	v_cvt_f16_f32_e32 v36, v47
	v_cvt_f16_f32_e32 v37, v48
	v_cvt_f16_f32_e32 v38, v54
	v_cvt_f16_f32_e32 v39, v55
	v_cvt_f16_f32_e32 v40, v56
	v_cvt_f16_f32_e32 v41, v57
	v_cvt_f16_f32_e32 v42, v58
	v_cvt_f16_f32_e32 v43, v59
	v_cvt_f16_f32_e32 v44, v60
	v_cvt_f16_f32_e32 v45, v61
	v_cvt_f16_f32_e32 v46, v62
	v_cvt_f16_f32_e32 v47, v63
	v_pack_b32_f16 v33, v33, v35
	v_pack_b32_f16 v32, v32, v34
	v_add_lshl_u32 v34, v79, v103, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v38, v39
	v_pack_b32_f16 v32, v36, v37
	v_add_lshl_u32 v34, v79, v102, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v42, v43
	v_pack_b32_f16 v32, v40, v41
	v_add_lshl_u32 v34, v92, v103, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v46, v47
	v_pack_b32_f16 v32, v44, v45
	v_add_lshl_u32 v34, v92, v102, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v32, v68
	v_cvt_f16_f32_e32 v34, v69
	v_cvt_f16_f32_e32 v33, v106
	v_cvt_f16_f32_e32 v35, v107
	v_cvt_f16_f32_e32 v36, v67
	v_cvt_f16_f32_e32 v37, v110
	v_cvt_f16_f32_e32 v38, v137
	v_cvt_f16_f32_e32 v39, v138
	v_cvt_f16_f32_e32 v40, v139
	v_cvt_f16_f32_e32 v41, v140
	v_cvt_f16_f32_e32 v42, v141
	v_cvt_f16_f32_e32 v43, v142
	v_cvt_f16_f32_e32 v44, v149
	v_cvt_f16_f32_e32 v45, v150
	v_cvt_f16_f32_e32 v46, v151
	v_cvt_f16_f32_e32 v47, v155
	v_pack_b32_f16 v33, v33, v35
	v_pack_b32_f16 v32, v32, v34
	v_add_lshl_u32 v34, v79, v99, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v38, v39
	v_pack_b32_f16 v32, v36, v37
	v_add_lshl_u32 v34, v79, v98, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v42, v43
	v_pack_b32_f16 v32, v40, v41
	v_add_lshl_u32 v34, v92, v99, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v46, v47
	v_pack_b32_f16 v32, v44, v45
	v_add_lshl_u32 v34, v92, v98, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v32, v124
	v_cvt_f16_f32_e32 v34, v125
	v_cvt_f16_f32_e32 v33, v129
	v_cvt_f16_f32_e32 v35, v130
	v_cvt_f16_f32_e32 v36, v131
	v_cvt_f16_f32_e32 v37, v134
	v_cvt_f16_f32_e32 v38, v135
	v_cvt_f16_f32_e32 v39, v136
	v_cvt_f16_f32_e32 v40, v152
	v_cvt_f16_f32_e32 v41, v153
	v_cvt_f16_f32_e32 v42, v154
	v_cvt_f16_f32_e32 v43, v162
	v_cvt_f16_f32_e32 v44, v163
	v_cvt_f16_f32_e32 v45, v164
	v_cvt_f16_f32_e32 v46, v183
	v_cvt_f16_f32_e32 v47, v184
	v_pack_b32_f16 v33, v33, v35
	v_pack_b32_f16 v32, v32, v34
	v_add_lshl_u32 v34, v79, v74, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v38, v39
	v_pack_b32_f16 v32, v36, v37
	v_add_lshl_u32 v34, v79, v251, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v42, v43
	v_pack_b32_f16 v32, v40, v41
	v_add_lshl_u32 v34, v92, v74, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v46, v47
	v_pack_b32_f16 v32, v44, v45
	v_add_lshl_u32 v34, v92, v251, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v32, v132
	v_cvt_f16_f32_e32 v34, v143
	v_cvt_f16_f32_e32 v33, v144
	v_cvt_f16_f32_e32 v35, v145
	v_cvt_f16_f32_e32 v36, v156
	v_cvt_f16_f32_e32 v37, v157
	v_cvt_f16_f32_e32 v38, v158
	v_cvt_f16_f32_e32 v39, v171
	v_cvt_f16_f32_e32 v40, v172
	v_cvt_f16_f32_e32 v41, v173
	v_cvt_f16_f32_e32 v42, v185
	v_cvt_f16_f32_e32 v43, v186
	v_add_u32_e32 v94, s4, v92
	v_cvt_f16_f32_e32 v44, v187
	v_cvt_f16_f32_e32 v45, v197
	v_cvt_f16_f32_e32 v46, v198
	v_cvt_f16_f32_e32 v47, v199
	v_pack_b32_f16 v33, v33, v35
	v_pack_b32_f16 v32, v32, v34
	v_add_lshl_u32 v34, v94, v104, 1
	v_add_u32_e32 v89, s4, v94
	; sched_barrier mask(0x00000406)
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v38, v39
	v_pack_b32_f16 v32, v36, v37
	v_add_lshl_u32 v34, v94, v105, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v42, v43
	v_pack_b32_f16 v32, v40, v41
	v_add_lshl_u32 v34, v89, v104, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v46, v47
	v_pack_b32_f16 v32, v44, v45
	v_add_lshl_u32 v34, v89, v105, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v32, v165
	v_cvt_f16_f32_e32 v34, v166
	v_cvt_f16_f32_e32 v33, v167
	v_cvt_f16_f32_e32 v35, v177
	v_cvt_f16_f32_e32 v36, v178
	v_cvt_f16_f32_e32 v37, v179
	v_cvt_f16_f32_e32 v38, v191
	v_cvt_f16_f32_e32 v39, v192
	v_cvt_f16_f32_e32 v40, v193
	v_cvt_f16_f32_e32 v41, v203
	v_cvt_f16_f32_e32 v42, v204
	v_cvt_f16_f32_e32 v43, v205
	v_cvt_f16_f32_e32 v44, v209
	v_cvt_f16_f32_e32 v45, v210
	v_cvt_f16_f32_e32 v46, v211
	v_cvt_f16_f32_e32 v47, v215
	v_pack_b32_f16 v33, v33, v35
	v_pack_b32_f16 v32, v32, v34
	v_add_lshl_u32 v34, v94, v103, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v38, v39
	v_pack_b32_f16 v32, v36, v37
	v_add_lshl_u32 v34, v94, v102, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v42, v43
	v_pack_b32_f16 v32, v40, v41
	v_add_lshl_u32 v34, v89, v103, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v46, v47
	v_pack_b32_f16 v32, v44, v45
	v_add_lshl_u32 v34, v89, v102, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v32, v219
	v_cvt_f16_f32_e32 v34, v220
	v_cvt_f16_f32_e32 v33, v221
	v_cvt_f16_f32_e32 v35, v222
	v_cvt_f16_f32_e32 v36, v223
	v_cvt_f16_f32_e32 v37, v224
	v_cvt_f16_f32_e32 v38, v227
	v_cvt_f16_f32_e32 v39, v228
	v_cvt_f16_f32_e32 v40, v229
	v_cvt_f16_f32_e32 v41, v233
	v_cvt_f16_f32_e32 v42, v234
	v_cvt_f16_f32_e32 v43, v235
	v_cvt_f16_f32_e32 v44, v239
	v_cvt_f16_f32_e32 v45, v240
	v_cvt_f16_f32_e32 v46, v241
	v_cvt_f16_f32_e32 v47, v245
	v_pack_b32_f16 v33, v33, v35
	v_pack_b32_f16 v32, v32, v34
	v_add_lshl_u32 v34, v94, v99, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v38, v39
	v_pack_b32_f16 v32, v36, v37
	v_add_lshl_u32 v34, v94, v98, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v42, v43
	v_pack_b32_f16 v32, v40, v41
	v_add_lshl_u32 v34, v89, v99, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v46, v47
	v_pack_b32_f16 v32, v44, v45
	v_add_lshl_u32 v34, v89, v98, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v32, v225
	v_cvt_f16_f32_e32 v34, v226
	v_cvt_f16_f32_e32 v33, v230
	v_cvt_f16_f32_e32 v35, v231
	v_cvt_f16_f32_e32 v36, v232
	v_cvt_f16_f32_e32 v37, v236
	v_cvt_f16_f32_e32 v38, v237
	v_cvt_f16_f32_e32 v39, v238
	v_cvt_f16_f32_e32 v40, v242
	v_cvt_f16_f32_e32 v41, v243
	v_cvt_f16_f32_e32 v42, v244
	v_cvt_f16_f32_e32 v43, v246
	v_cvt_f16_f32_e32 v44, v247
	v_cvt_f16_f32_e32 v45, v248
	v_cvt_f16_f32_e32 v46, v249
	v_cvt_f16_f32_e32 v47, v250
	v_pack_b32_f16 v33, v33, v35
	v_pack_b32_f16 v32, v32, v34
	v_add_lshl_u32 v34, v94, v74, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v38, v39
	v_pack_b32_f16 v32, v36, v37
	v_add_lshl_u32 v34, v94, v251, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v42, v43
	v_pack_b32_f16 v32, v40, v41
	v_add_lshl_u32 v34, v89, v74, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_pack_b32_f16 v33, v46, v47
	v_pack_b32_f16 v32, v44, v45
	v_add_lshl_u32 v34, v89, v251, 1
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v32, v17
	v_cvt_f16_f32_e32 v17, v18
	v_cvt_f16_f32_e32 v18, v19
	v_cvt_f16_f32_e32 v19, v20
	v_cvt_f16_f32_e32 v20, v22
	v_cvt_f16_f32_e32 v22, v23
	v_cvt_f16_f32_e32 v23, v24
	v_cvt_f16_f32_e32 v24, v28
	v_cvt_f16_f32_e32 v28, v29
	v_cvt_f16_f32_e32 v29, v30
	v_cvt_f16_f32_e32 v30, v112
	v_add_u32_e32 v65, s4, v89
	v_cvt_f16_f32_e32 v33, v113
	v_cvt_f16_f32_e32 v34, v114
	v_cvt_f16_f32_e32 v35, v118
	v_cvt_f16_f32_e32 v36, v119
	v_pack_b32_f16 v17, v17, v18
	v_pack_b32_f16 v16, v16, v32
	v_add_lshl_u32 v18, v65, v104, 1
	v_add_u32_e32 v64, s4, v65
	; sched_barrier mask(0x00000406)
	buffer_store_dwordx2 v[16:17], v18, s[0:3], 0 offen
	v_pack_b32_f16 v17, v22, v23
	v_pack_b32_f16 v16, v19, v20
	v_add_lshl_u32 v18, v65, v105, 1
	buffer_store_dwordx2 v[16:17], v18, s[0:3], 0 offen
	v_pack_b32_f16 v17, v29, v30
	v_pack_b32_f16 v16, v24, v28
	v_add_lshl_u32 v18, v64, v104, 1
	buffer_store_dwordx2 v[16:17], v18, s[0:3], 0 offen
	v_pack_b32_f16 v17, v35, v36
	v_pack_b32_f16 v16, v33, v34
	v_add_lshl_u32 v18, v64, v105, 1
	buffer_store_dwordx2 v[16:17], v18, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v16, v21
	v_cvt_f16_f32_e32 v18, v25
	v_cvt_f16_f32_e32 v17, v26
	v_cvt_f16_f32_e32 v19, v27
	v_cvt_f16_f32_e32 v20, v31
	v_cvt_f16_f32_e32 v21, v66
	v_cvt_f16_f32_e32 v22, v111
	v_cvt_f16_f32_e32 v23, v115
	v_cvt_f16_f32_e32 v24, v116
	v_cvt_f16_f32_e32 v25, v117
	v_cvt_f16_f32_e32 v26, v120
	v_cvt_f16_f32_e32 v27, v121
	v_cvt_f16_f32_e32 v28, v123
	v_cvt_f16_f32_e32 v29, v126
	v_cvt_f16_f32_e32 v30, v127
	v_cvt_f16_f32_e32 v31, v128
	v_pack_b32_f16 v17, v17, v19
	v_pack_b32_f16 v16, v16, v18
	v_add_lshl_u32 v18, v65, v103, 1
	buffer_store_dwordx2 v[16:17], v18, s[0:3], 0 offen
	v_pack_b32_f16 v17, v22, v23
	v_pack_b32_f16 v16, v20, v21
	v_add_lshl_u32 v18, v65, v102, 1
	buffer_store_dwordx2 v[16:17], v18, s[0:3], 0 offen
	v_pack_b32_f16 v17, v26, v27
	v_pack_b32_f16 v16, v24, v25
	v_add_lshl_u32 v18, v64, v103, 1
	buffer_store_dwordx2 v[16:17], v18, s[0:3], 0 offen
	v_pack_b32_f16 v17, v30, v31
	v_pack_b32_f16 v16, v28, v29
	v_add_lshl_u32 v18, v64, v102, 1
	v_accvgpr_read_b32 v90, a22
	v_accvgpr_read_b32 v93, a23
	buffer_store_dwordx2 v[16:17], v18, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v16, v217
	v_cvt_f16_f32_e32 v17, v218
	v_cvt_f16_f32_e32 v18, v4
	v_cvt_f16_f32_e32 v19, v5
	v_cvt_f16_f32_e32 v7, v7
	v_cvt_f16_f32_e32 v20, v90
	v_cvt_f16_f32_e32 v21, v93
	v_pack_b32_f16 v5, v10, v11
	v_pack_b32_f16 v4, v8, v9
	v_add_lshl_u32 v8, v65, v99, 1
	buffer_store_dwordx2 v[4:5], v8, s[0:3], 0 offen
	v_pack_b32_f16 v5, v14, v15
	v_pack_b32_f16 v4, v12, v13
	v_add_lshl_u32 v8, v65, v98, 1
	buffer_store_dwordx2 v[4:5], v8, s[0:3], 0 offen
	v_pack_b32_f16 v5, v18, v19
	v_pack_b32_f16 v4, v16, v17
	v_add_lshl_u32 v8, v64, v99, 1
	v_accvgpr_read_b32 v1, a0
	v_accvgpr_read_b32 v2, a1
	v_accvgpr_read_b32 v3, a2
	v_accvgpr_read_b32 v6, a3
	buffer_store_dwordx2 v[4:5], v8, s[0:3], 0 offen
	v_pack_b32_f16 v5, v20, v21
	v_pack_b32_f16 v4, v0, v7
	v_add_lshl_u32 v0, v64, v98, 1
	v_accvgpr_read_b32 v84, a4
	buffer_store_dwordx2 v[4:5], v0, s[0:3], 0 offen
	v_cvt_f16_f32_e32 v0, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v1, v3
	v_cvt_f16_f32_e32 v3, v6
	v_accvgpr_read_b32 v88, a12
	v_accvgpr_read_b32 v80, a13
	v_accvgpr_read_b32 v82, a14
	v_accvgpr_read_b32 v83, a15
	v_cvt_f16_f32_e32 v4, v84
	v_cvt_f16_f32_e32 v5, v81
	v_cvt_f16_f32_e32 v6, v85
	v_cvt_f16_f32_e32 v7, v86
	v_accvgpr_read_b32 v87, a8
	v_accvgpr_read_b32 v72, a9
	v_accvgpr_read_b32 v76, a10
	v_accvgpr_read_b32 v78, a11
	v_cvt_f16_f32_e32 v8, v88
	v_cvt_f16_f32_e32 v9, v80
	v_cvt_f16_f32_e32 v10, v82
	v_cvt_f16_f32_e32 v11, v83
	v_cvt_f16_f32_e32 v12, v87
	v_cvt_f16_f32_e32 v13, v72
	v_cvt_f16_f32_e32 v14, v76
	v_cvt_f16_f32_e32 v15, v78
	v_pack_b32_f16 v1, v1, v3
	v_pack_b32_f16 v0, v0, v2
	v_add_lshl_u32 v2, v65, v74, 1
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
	v_pack_b32_f16 v1, v6, v7
	v_pack_b32_f16 v0, v4, v5
	v_add_lshl_u32 v2, v65, v251, 1
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
	v_pack_b32_f16 v1, v10, v11
	v_pack_b32_f16 v0, v8, v9
	v_add_lshl_u32 v2, v64, v74, 1
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
	v_pack_b32_f16 v1, v14, v15
	v_pack_b32_f16 v0, v12, v13
	v_add_lshl_u32 v2, v64, v251, 1
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
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
