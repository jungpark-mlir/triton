// RUN: triton-opt %s -split-input-file --allocate-shared-memory -test-print-membar | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory -test-tritonamdgpu-membar | FileCheck %s

// Tests for Presburger index analysis in membar.
// These test cases exercise the areIndicesProvablyDifferent() function
// which uses MLIR's Presburger arithmetic library to prove that two
// MemDescIndexOp indices can never be equal, eliminating false-positive
// barriers.

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {

// CHECK-LABEL: constant_indices_disjoint
// Two memdesc_index with different constant indices are provably disjoint.
// The Presburger solver encodes: v0 = 0, v1 = 1, v0 = v1? → 0 = 1 → EMPTY.
// No barrier needed between write to slot 0 and read from slot 1.
tt.func @constant_indices_disjoint() {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  %view0 = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %view1 = ttg.memdesc_index %alloc[%c1_i32] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  ttg.local_store %cst, %view0 : tensor<128x128xf16> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  %load = ttg.local_load %view1 : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
  tt.return
}

// CHECK-LABEL: constant_indices_same
// Same constant index → same slot → barrier required.
tt.func @constant_indices_same() {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16>
  %c0_a = arith.constant 0 : i32
  %c0_b = arith.constant 0 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  %view0 = ttg.memdesc_index %alloc[%c0_a] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %view1 = ttg.memdesc_index %alloc[%c0_b] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  ttg.local_store %cst, %view0 : tensor<128x128xf16> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  %load = ttg.local_load %view1 : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
  tt.return
}

// CHECK-LABEL: unified_phase_counter_disjoint
// Producer: slot = (phase + 2) % 3,  Consumer: slot = phase % 3.
// Presburger: r1 = (phase+2) mod 3, r2 = phase mod 3, r1 = r2?
//   → 3*(q2-q1) = 2, gcd(3)=3, 3 ∤ 2 → EMPTY → disjoint.
tt.func @unified_phase_counter_disjoint(%phase: i32) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16>
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
  // Consumer index: phase % 3
  %consumer_idx = arith.remui %phase, %c3_i32 : i32
  %consumer_view = ttg.memdesc_index %alloc[%consumer_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // Producer index: (phase + 2) % 3
  %phase_plus_2 = arith.addi %phase, %c2_i32 : i32
  %producer_idx = arith.remui %phase_plus_2, %c3_i32 : i32
  %producer_view = ttg.memdesc_index %alloc[%producer_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  ttg.local_store %cst, %producer_view : tensor<128x128xf16> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  %load = ttg.local_load %consumer_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
  tt.return
}

// CHECK-LABEL: unified_phase_counter_aliasing
// Producer: slot = (phase + 3) % 3,  Consumer: slot = phase % 3.
// (phase + 3) mod 3 = phase mod 3 always → MAY alias → barrier needed.
tt.func @unified_phase_counter_aliasing(%phase: i32) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16>
  %c3_add = arith.constant 3 : i32
  %c3_mod = arith.constant 3 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
  %consumer_idx = arith.remui %phase, %c3_mod : i32
  %consumer_view = ttg.memdesc_index %alloc[%consumer_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %phase_plus_3 = arith.addi %phase, %c3_add : i32
  %producer_idx = arith.remui %phase_plus_3, %c3_mod : i32
  %producer_view = ttg.memdesc_index %alloc[%producer_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  ttg.local_store %cst, %producer_view : tensor<128x128xf16> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  %load = ttg.local_load %consumer_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
  tt.return
}

// CHECK-LABEL: four_buffer_stage_distance_3
// 4-buffer pipeline, stage distance 3.
// Producer: (phase+3) % 4, Consumer: phase % 4.
// 4*(q2-q1) = 3, gcd(4)=4, 4 ∤ 3 → EMPTY → disjoint.
tt.func @four_buffer_stage_distance_3(%phase: i32) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16>
  %c3_i32 = arith.constant 3 : i32
  %c4_i32 = arith.constant 4 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable>
  %consumer_idx = arith.remui %phase, %c4_i32 : i32
  %consumer_view = ttg.memdesc_index %alloc[%consumer_idx] : !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %phase_plus_3 = arith.addi %phase, %c3_i32 : i32
  %producer_idx = arith.remui %phase_plus_3, %c4_i32 : i32
  %producer_view = ttg.memdesc_index %alloc[%producer_idx] : !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  ttg.local_store %cst, %producer_view : tensor<128x128xf16> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  %load = ttg.local_load %consumer_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
  tt.return
}

// CHECK-LABEL: four_buffer_degenerate_alias
// 4-buffer pipeline, stage distance 4 (degenerate: offset 4 ≡ 0 mod 4).
// 4*(q2-q1) = 4, gcd(4)=4, 4 | 4 → NOT EMPTY → may alias → barrier.
tt.func @four_buffer_degenerate_alias(%phase: i32) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16>
  %c4_add = arith.constant 4 : i32
  %c4_mod = arith.constant 4 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable>
  %consumer_idx = arith.remui %phase, %c4_mod : i32
  %consumer_view = ttg.memdesc_index %alloc[%consumer_idx] : !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %phase_plus_4 = arith.addi %phase, %c4_add : i32
  %producer_idx = arith.remui %phase_plus_4, %c4_mod : i32
  %producer_view = ttg.memdesc_index %alloc[%producer_idx] : !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  ttg.local_store %cst, %producer_view : tensor<128x128xf16> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  %load = ttg.local_load %consumer_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
  tt.return
}

// CHECK-LABEL: bitwise_and_power_of_2
// phase & 3 vs (phase + 2) & 3.  andi with mask 3 = x mod 4.
// (phase+2) mod 4 vs phase mod 4 → 4*(q2-q1) = 2, 4 ∤ 2 → disjoint.
tt.func @bitwise_and_power_of_2(%phase: i32) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16>
  %c2_i32 = arith.constant 2 : i32
  %c3_mask = arith.constant 3 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable>
  %consumer_idx = arith.andi %phase, %c3_mask : i32
  %consumer_view = ttg.memdesc_index %alloc[%consumer_idx] : !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %phase_plus_2 = arith.addi %phase, %c2_i32 : i32
  %producer_idx = arith.andi %phase_plus_2, %c3_mask : i32
  %producer_view = ttg.memdesc_index %alloc[%producer_idx] : !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  ttg.local_store %cst, %producer_view : tensor<128x128xf16> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  %load = ttg.local_load %consumer_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
  tt.return
}

// CHECK-LABEL: remsi_disjoint
// Same as unified_phase_counter but with arith.remsi instead of remui.
// Buffer phase counters are non-negative so remsi behaves like mod.
tt.func @remsi_disjoint(%phase: i32) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16>
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
  %consumer_idx = arith.remsi %phase, %c3_i32 : i32
  %consumer_view = ttg.memdesc_index %alloc[%consumer_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %phase_plus_2 = arith.addi %phase, %c2_i32 : i32
  %producer_idx = arith.remsi %phase_plus_2, %c3_i32 : i32
  %producer_view = ttg.memdesc_index %alloc[%producer_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  ttg.local_store %cst, %producer_view : tensor<128x128xf16> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  %load = ttg.local_load %consumer_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
  tt.return
}

// CHECK-LABEL: muli_disjoint
// Producer: 4*phase + 2,  Consumer: 4*phase.
// These are always different (differ by 2, constant offset).
tt.func @muli_disjoint(%phase: i32) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16>
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x128x128xf16, #shared, #smem, mutable>
  %base = arith.muli %phase, %c4_i32 : i32
  %consumer_view = ttg.memdesc_index %alloc[%base] : !ttg.memdesc<16x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %producer_idx = arith.addi %base, %c2_i32 : i32
  %producer_view = ttg.memdesc_index %alloc[%producer_idx] : !ttg.memdesc<16x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  ttg.local_store %cst, %producer_view : tensor<128x128xf16> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  %load = ttg.local_load %consumer_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
  tt.return
}

// CHECK-LABEL: opaque_index_conservative
// When the index comes from an unrecognized op (function argument with
// no arith chain), the solver treats it as unconstrained and cannot
// prove disjointness → conservative barrier.
tt.func @opaque_index_conservative(%idx_a: i32, %idx_b: i32) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  %view0 = ttg.memdesc_index %alloc[%idx_a] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %view1 = ttg.memdesc_index %alloc[%idx_b] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  ttg.local_store %cst, %view0 : tensor<128x128xf16> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  %load = ttg.local_load %view1 : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
  tt.return
}

// CHECK-LABEL: loop_carried_conservative
// When one side is a loop-carried block argument, MemDescIndexOp has no
// defining op → Presburger check is skipped → barrier is conservative.
tt.func @loop_carried_conservative(%lb : index, %ub : index) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16>
  %step = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  %view0 = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  ttg.local_store %cst, %view0 : tensor<128x128xf16> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %result = scf.for %iv = %lb to %ub step %step iter_args(%iter_view = %view0) -> (!ttg.memdesc<128x128xf16, #shared, #smem, mutable>) {
    // CHECK: ttg.barrier local
    // CHECK-NEXT: ttg.local_load
    %load = ttg.local_load %iter_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
    %iv_i32 = arith.index_cast %iv : index to i32
    %next_idx = arith.remui %iv_i32, %c2_i32 : i32
    %next_view = ttg.memdesc_index %alloc[%next_idx] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: ttg.barrier local
    // CHECK-NEXT: ttg.local_store
    ttg.local_store %load, %next_view : tensor<128x128xf16> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    scf.yield %next_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  }
  tt.return
}

} // module
