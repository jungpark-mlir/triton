// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --allocate-shared-memory --triton-amdgpu-membar='gfx-arch=gfx942' | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --allocate-shared-memory --triton-amdgpu-membar='gfx-arch=gfx950' | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --allocate-shared-memory --triton-amdgpu-membar='gfx-arch=gfx90a' | FileCheck %s --check-prefix=UNSUPPORTED

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// Cross-warp consumers preserve the barrier-placement coverage.
#CROSS = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#A_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// Check that we only get a single barrier when using AsyncWait
// CHECK-LABEL: pipelined_async_copy_local_to_global
tt.func @pipelined_async_copy_local_to_global(%A: !tt.ptr<f16>) {
  %index_0 = arith.constant 0 : i32
  %index_1 = arith.constant 1 : i32
  %a_ptr = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_a = ttg.memdesc_index %alloc[%index_0] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_b = ttg.memdesc_index %alloc[%index_1] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Load TileA
  %1 = ttg.async_copy_global_to_local %a_ptr, %tile_a: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Wait for TileA
  %2 = ttg.async_wait %1 {num = 4 : i32}
  // Read TileA
  %4 = ttg.local_load %tile_a token %2 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #CROSS>
  // Load into TileB
  %3 = ttg.async_copy_global_to_local %a_ptr, %tile_b : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // There should be a single barrier after async_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.return
  tt.return
}
// Same as above but different order of ops
// CHECK-LABEL: pipelined_async_copy_local_to_global_2
tt.func @pipelined_async_copy_local_to_global_2(%A: !tt.ptr<f16>) {
  %index_0 = arith.constant 0 : i32
  %index_1 = arith.constant 1 : i32
  %a_ptr = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_a = ttg.memdesc_index %alloc[%index_0] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_b = ttg.memdesc_index %alloc[%index_1] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Load Tile
  %1 = ttg.async_copy_global_to_local %a_ptr, %tile_a: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Wait for TileA
  %2 = ttg.async_wait %1 {num = 4 : i32}
  // Load into TileB
  %3 = ttg.async_copy_global_to_local %a_ptr, %tile_b : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Read TileA
  %4 = ttg.local_load %tile_a token %2 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #CROSS>
  // The disjoint copy can issue before the cross-warp rendezvous.
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.async_wait
  // CHECK-NEXT: {{.*}} = ttg.async_copy_global_to_local
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: {{.*}} = ttg.local_load
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.return
  tt.return
}
// Check that multiple LocalLoads waiting on the same AsyncWait produce one barrier
// CHECK-LABEL: pipelined_async_copy_local_to_global_3
tt.func @pipelined_async_copy_local_to_global_3(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>) {
  %index_0 = arith.constant 0 : i32
  %index_1 = arith.constant 1 : i32
  %a_ptr = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #AL>
  %b_ptr = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #AL>

  %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_a_1 = ttg.memdesc_index %alloc_a[%index_0] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_a_2 = ttg.memdesc_index %alloc_a[%index_1] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_b_1 = ttg.memdesc_index %alloc_b[%index_0] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile_b_2 = ttg.memdesc_index %alloc_b[%index_1] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  // Load TileA_1
  %1 = ttg.async_copy_global_to_local %a_ptr, %tile_a_1: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Load TileB_1
  %2 = ttg.async_copy_global_to_local %b_ptr, %tile_b_1: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Wait for TileA
  %3 = ttg.async_wait %1, %2 {num = 4 : i32}
  // Read TileA_1
  %4 = ttg.local_load %tile_a_1 token %3 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #CROSS>
  // Read TileB_1
  %5 = ttg.local_load %tile_b_1 token %3 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #CROSS>
  // Load into TileA_2
  %6 = ttg.async_copy_global_to_local %a_ptr, %tile_a_2 : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Load into TileB_2
  %7 = ttg.async_copy_global_to_local %b_ptr, %tile_b_2 : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  // There should be a single barrier after async_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.return
  tt.return
}

// The wait's publication barrier follows its token across the loop boundary.
// The refill also waits for the local load to finish.
// CHECK-LABEL: async_wait_in_previous_loop_iteration
tt.func @async_wait_in_previous_loop_iteration(%a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}

  // CHECK: ttg.async_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: cf.br
  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    // CHECK: ttg.barrier local
    // CHECK-NEXT: ttg.local_load
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttg.async_copy_global_to_local
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #CROSS>
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

    // CHECK-NOT: ttg.barrier local
    // CHECK: ttg.async_wait
    %8 = ttg.async_wait %7 {num = 4 : i32}
    // CHECK-NOT: ttg.barrier local
    // CHECK: cf.br
    scf.yield %8: !ttg.async.token
  }
  // CHECK: ttg.barrier local
  // CHECK-NEXT: tt.return
  tt.return
}

// Check we do get a barrier for LocalLoad if the initial loop token does not come from AsyncWait
// CHECK-LABEL: intial_loop_token_is_not_from_async_wait
tt.func @intial_loop_token_is_not_from_async_wait(%a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %1) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #CROSS>
    // CHECK: ttg.local_load
    // CHECK: ttg.barrier local
    // CHECK: ttg.async_copy_global_to_local
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    %8 = ttg.async_wait %7 {num = 4 : i32}
    scf.yield %8: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}

// Same as above but the loop carried token does not come from AsyncWait
// CHECK-LABEL: loop_carried_token_not_from_async_wait
tt.func @loop_carried_token_not_from_async_wait(%a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}
  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #CROSS>
    // CHECK: ttg.local_load
    // CHECK: ttg.barrier local
    // CHECK: ttg.async_copy_global_to_local
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    scf.yield %7: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}


// Waits from both branches are published before the next local load.
// The refill also waits for the local load to finish.
// CHECK-LABEL: async_wait_inside_if
tt.func @async_wait_inside_if(%cond: i1, %a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}

  // CHECK: cf.br
  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    // CHECK: ttg.barrier local
    // CHECK-NEXT: ttg.local_load
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttg.async_copy_global_to_local
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #CROSS>
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    %103 = scf.if %cond -> (!ttg.async.token) {
      %8 = ttg.async_wait %7 {num = 4 : i32}
      scf.yield %8 : !ttg.async.token
    } else {
      %9 = ttg.async_wait %7 {num = 4 : i32}
      scf.yield %9 : !ttg.async.token
    }
    scf.yield %103: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}

// Check that we do get a barrier for an if where one branch does not yield an token from AsyncWait
// CHECK-LABEL: non_async_wait_token_from_then
tt.func @non_async_wait_token_from_then(%cond: i1, %a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}

  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #CROSS>
    // We should get a barrier because the then branch does not yield an token from AsyncWait
    // CHECK: ttg.local_load
    // CHECK: ttg.barrier local
    // CHECK: ttg.async_copy_global_to_local
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    %103 = scf.if %cond -> (!ttg.async.token) {
      scf.yield %7 : !ttg.async.token
    } else {
      %8 = ttg.async_wait %7 {num = 4 : i32}
      scf.yield %8 : !ttg.async.token
    }
    scf.yield %103: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}

// See above
// CHECK-LABEL: non_async_wait_token_from_else
tt.func @non_async_wait_token_from_else(%cond: i1, %a_ptr: tensor<16x16x!tt.ptr<f16>, #AL>, %loopIterCount: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %1 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %2 = ttg.async_wait %1 {num = 4 : i32}

  %loop_result:1 = scf.for %arg14 = %c0_i32 to %loopIterCount step %c1_i32 iter_args(%arg10 = %2) -> (!ttg.async.token)  : i32 {
    %6 = ttg.local_load %alloc token %arg10 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #CROSS>
    // We should get a barrier because the else branch does not yield an token from AsyncWait
    // CHECK: ttg.local_load
    // CHECK: ttg.barrier local
    // CHECK: ttg.async_copy_global_to_local
    %7 = ttg.async_copy_global_to_local %a_ptr, %alloc : tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
    %103 = scf.if %cond -> (!ttg.async.token) {
      %8 = ttg.async_wait %7 {num = 4 : i32}
      scf.yield %8 : !ttg.async.token
    } else {
      %9 = ttg.async_copy_global_to_local %a_ptr, %alloc: tensor<16x16x!tt.ptr<f16>, #AL> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
      scf.yield %9 : !ttg.async.token
    }
    scf.yield %103: !ttg.async.token
  }
  // CHECK: tt.return
  tt.return
}

// CHECK-LABEL: missing_barrier_reused_allocation
tt.func @missing_barrier_reused_allocation(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %alloc1 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %offset = arith.constant dense<0> : tensor<128x32xi32, #AL>

  %slice1_0 = ttg.memdesc_index %alloc1[%c0_i32] : !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %async1 = amdg.buffer_load_to_local %A[%offset] into %slice1_0 : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %token1 = ttg.async_commit_group tokens %async1
  %wait1 = amdg.async_wait %token1 {num_inst = 0 : i32}
  // CHECK: ttg.barrier local
  // CHECK: ttg.local_load
  %local_load = ttg.local_load %slice1_0 token %wait1 {ttg.amdg.syncedViaAsyncWait = true} : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<128x32xf16, #CROSS>
  ttg.local_dealloc %alloc1 : !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %alloc2 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %slice2_0 = ttg.memdesc_index %alloc2[%c0_i32] : !ttg.memdesc<2x128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // op2: Async load into alloc2 (overlapping with the dealloc'd alloc1 that is still being local_load'd from)
  // CHECK: ttg.barrier local
  // CHECK-NEXT: amdg.buffer_load_to_local
  %async2 = amdg.buffer_load_to_local %B[%offset] into %slice2_0 : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %token2 = ttg.async_commit_group tokens %async2
  %wait2 = amdg.async_wait %token2 {num_inst = 0 : i32}
  // CHECK: ttg.barrier local
  tt.return
}

}

// -----
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// Cross-warp consumers preserve the barrier-placement coverage.
#CROSS = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: must_barrier_tdm_copy_select_cmpi_unbounded_base
// The select/cmpi modular-wrap idiom only equals (base + 1) % N when
// -1 <= base < N. Here the base is a plain function argument with no
// provable bound, so the matcher must reject the modular form and a barrier
// must be emitted between the TDM copy and the local_load.
// Positive coverage for the iter_arg path lives in
// test/Analysis/test-membar.mlir.
tt.func @must_barrier_tdm_copy_select_cmpi_unbounded_base(%desc: !tt.tensordesc<128x128xf16, #shared>, %phase: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %c_pred = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>

  %write_sum = arith.addi %phase, %c1_i32 : i32
  %write_cmp = arith.cmpi slt, %write_sum, %c3_i32 : i32
  %write_idx = arith.select %write_cmp, %write_sum, %c0_i32 : i32
  %write_view = ttg.memdesc_index %alloc[%write_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

  %read_idx = arith.remsi %phase, %c3_i32 : i32
  %read_view = ttg.memdesc_index %alloc[%read_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

  // CHECK: amdg.async_tdm_copy_global_to_local
  %copy_desc = amdg.update_tensor_descriptor %desc add_offsets = [%c0_i32, %c0_i32] pred = %c_pred {clamp_bounds} : !tt.tensordesc<128x128xf16, #shared>
  %token = amdg.async_tdm_copy_global_to_local %copy_desc into %write_view : !tt.tensordesc<128x128xf16, #shared> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  %load = ttg.local_load %read_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #CROSS>
  tt.return
}
}

// -----
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// Cross-warp consumers preserve the barrier-placement coverage.
#CROSS = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: disjoint_tdm_copy_remsi
// Test that TDM copy and local_load with provably disjoint dynamic indices
// using arith.remsi (gluon/canonical modular wrap) do not require a barrier.
tt.func @disjoint_tdm_copy_remsi(%desc: !tt.tensordesc<128x128xf16, #shared>, %phase: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %c_pred = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>

  // Write index: (phase + 2) % 3
  %write_phase = arith.addi %phase, %c2_i32 : i32
  %write_idx = arith.remsi %write_phase, %c3_i32 : i32
  %write_view = ttg.memdesc_index %alloc[%write_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

  // Read index: phase % 3
  %read_idx = arith.remsi %phase, %c3_i32 : i32
  %read_view = ttg.memdesc_index %alloc[%read_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

  // CHECK: amdg.async_tdm_copy_global_to_local
  %copy_desc = amdg.update_tensor_descriptor %desc add_offsets = [%c0_i32, %c0_i32] pred = %c_pred {clamp_bounds} : !tt.tensordesc<128x128xf16, #shared>
  %token = amdg.async_tdm_copy_global_to_local %copy_desc into %write_view : !tt.tensordesc<128x128xf16, #shared> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  %load = ttg.local_load %read_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #CROSS>
  tt.return
}

// CHECK-LABEL: disjoint_select_cmpi_iter_arg_cf
// After scf-to-cf, the loop-carried phase is a cf block argument. The analysis
// proves the select/cmpi wrap is bounded from the incoming init and backedge
// operands.
tt.func @disjoint_select_cmpi_iter_arg_cf(%cst: tensor<128x128xf16, #AL>, %ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %init = arith.constant -1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>

  // CHECK: cf.br
  %res = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%phase = %init) -> (i32) : i32 {
    %w_sum = arith.addi %phase, %c1_i32 : i32
    %w_cmp = arith.cmpi sge, %w_sum, %c3_i32 : i32
    %w_idx = arith.select %w_cmp, %c0_i32, %w_sum : i32
    %w_view = ttg.memdesc_index %alloc[%w_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    %r_idx = arith.remsi %phase, %c3_i32 : i32
    %r_view = ttg.memdesc_index %alloc[%r_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    // CHECK: ttg.barrier local
    // CHECK: ttg.local_store
    ttg.local_store %cst, %w_view : tensor<128x128xf16, #AL> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK-NOT: ttg.barrier local
    // CHECK: ttg.local_load
    %load = ttg.local_load %r_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #CROSS>

    scf.yield %w_idx : i32
  }
  tt.return
}

// CHECK-LABEL: disjoint_remsi_loop_carried_cf
// Next-iteration hazard after scf-to-cf lowering:
//   read_i  = phase_i % 3
//   write_i = (phase_i + 1) % 3
//   phase_{i+1} = phase_i + 1
// Dynamic index comparisons are disabled across the cf backedge, so the
// carried write cannot be proven disjoint from read_{i+1}.
tt.func @disjoint_remsi_loop_carried_cf(%cst: tensor<128x128xf16, #AL>, %ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>

  // CHECK: cf.br
  %res = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%phase = %c0_i32) -> (i32) : i32 {
    %r_idx = arith.remsi %phase, %c3_i32 : i32
    %r_view = ttg.memdesc_index %alloc[%r_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: ttg.barrier local
    // CHECK-NEXT: ttg.local_load
    %load = ttg.local_load %r_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #CROSS>

    %w_sum = arith.addi %phase, %c1_i32 : i32
    %w_idx = arith.remsi %w_sum, %c3_i32 : i32
    %w_view = ttg.memdesc_index %alloc[%w_idx] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK-NOT: ttg.barrier local
    // CHECK: ttg.local_store
    ttg.local_store %cst, %w_view : tensor<128x128xf16, #AL> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    %next_phase = arith.addi %phase, %c1_i32 : i32
    scf.yield %next_phase : i32
  }
  tt.return
}

// CHECK-LABEL: must_barrier_remsi_loop_carried_future_disjoint_cf
// Future-precision case after scf-to-cf lowering. The current analysis
// invalidates dynamic indices across the cf backedge, so it still requires a
// conservative barrier before the read.
tt.func @must_barrier_remsi_loop_carried_future_disjoint_cf(%cst: tensor<128x128xf16, #AL>, %ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable>

  // CHECK: cf.br
  %res = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%phase = %c0_i32) -> (i32) : i32 {
    %r_idx = arith.remsi %phase, %c4_i32 : i32
    %r_view = ttg.memdesc_index %alloc[%r_idx] : !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: ttg.barrier local
    // CHECK-NEXT: ttg.local_load
    %load = ttg.local_load %r_view : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #CROSS>

    %w_sum = arith.addi %phase, %c1_i32 : i32
    %w_idx = arith.remsi %w_sum, %c4_i32 : i32
    %w_view = ttg.memdesc_index %alloc[%w_idx] : !ttg.memdesc<4x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK-NOT: ttg.barrier local
    // CHECK: ttg.local_store
    ttg.local_store %cst, %w_view : tensor<128x128xf16, #AL> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    %next_phase = arith.addi %phase, %c2_i32 : i32
    scf.yield %next_phase : i32
  }
  tt.return
}
}

// -----

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// Cross-warp consumers preserve the barrier-placement coverage.
#CROSS = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: no_barrier_between_async_loads
tt.func @no_barrier_between_async_loads(%A: !tt.ptr<f16>, %i: i32, %j: i32) {
  %offset = arith.constant dense<0> : tensor<128x32xi32, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
  %tile_a = ttg.memdesc_index %alloc[%i] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  %tile_b = ttg.memdesc_index %alloc[%j] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  // CHECK: amdg.buffer_load_to_local
  // CHECK-NOT: ttg.barrier local
  // CHECK: amdg.buffer_load_to_local
  %async_a = amdg.buffer_load_to_local %A[%offset] into %tile_a : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
  %async_b = amdg.buffer_load_to_local %A[%offset] into %tile_b : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
  // CHECK: tt.return
  tt.return
}
}

// -----

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// Cross-warp consumers preserve the barrier-placement coverage.
#CROSS = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// Three-buffer pipelined loop
//   prologue: load buf[0]; load buf[1]
//   loop i:   load buf[i]; wait(2); j = (i+1)%3; read buf[j]; load buf[j]
// no barrier between async loads
// barrier before the first read after the async_wait
// barrier before the refill of the same buf
// CHECK-LABEL: no_barrier_between_async_loads_in_pipelined_loop
tt.func @no_barrier_between_async_loads_in_pipelined_loop(%A: !tt.ptr<f16>, %ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %offset = arith.constant dense<0> : tensor<128x32xi32, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable>

  %s0 = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  // CHECK: amdg.buffer_load_to_local
  // CHECK-NOT: ttg.barrier local
  %ld0 = amdg.buffer_load_to_local %A[%offset] into %s0 : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
  ttg.async_commit_group

  %s1 = ttg.memdesc_index %alloc[%c1_i32] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  // CHECK: amdg.buffer_load_to_local
  // CHECK-NOT: ttg.barrier local
  %ld1 = amdg.buffer_load_to_local %A[%offset] into %s1 : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
  ttg.async_commit_group

  // CHECK: cf.br
  %res = scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%i = %c2_i32) -> (i32) : i32 {
    %si = ttg.memdesc_index %alloc[%i] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    // CHECK-NOT: ttg.barrier local
    // CHECK: amdg.buffer_load_to_local
    %ldi = amdg.buffer_load_to_local %A[%offset] into %si : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
    ttg.async_commit_group

    // CHECK: ttg.async_wait
    // CHECK-NOT: ttg.barrier local
    ttg.async_wait {num = 2 : i32}

    %ip1 = arith.addi %i, %c1_i32 : i32
    %j = arith.remsi %ip1, %c3_i32 : i32
    %sj = ttg.memdesc_index %alloc[%j] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    // CHECK: ttg.memdesc_index
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttg.local_load
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: amdg.buffer_load_to_local
    %v = ttg.local_load %sj : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #CROSS>
    %ldj = amdg.buffer_load_to_local %A[%offset] into %sj : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
    ttg.async_commit_group

    scf.yield %j : i32
  }
  tt.return
}
}

// -----
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// Cross-warp consumers preserve the barrier-placement coverage.
#CROSS = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: tdm_refill_after_synced_local_load
tt.func @tdm_refill_after_synced_local_load(%desc: !tt.tensordesc<128x128xf16, #shared>, %i: i32, %j: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c_pred = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  %r = ttg.memdesc_index %alloc[%i] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %w = ttg.memdesc_index %alloc[%j] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %d = amdg.update_tensor_descriptor %desc add_offsets = [%c0_i32, %c0_i32] pred = %c_pred {clamp_bounds} : !tt.tensordesc<128x128xf16, #shared>
  // CHECK: amdg.async_tdm_copy_global_to_local
  %tok0 = amdg.async_tdm_copy_global_to_local %d into %r : !tt.tensordesc<128x128xf16, #shared> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK: amdg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NOT: ttg.barrier local
  %wait = amdg.async_wait %tok0 {num_inst = 0 : i32}
  // CHECK: ttg.local_load
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: amdg.async_tdm_copy_global_to_local
  %v = ttg.local_load %r token %wait : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #CROSS>
  %tok1 = amdg.async_tdm_copy_global_to_local %d into %w : !tt.tensordesc<128x128xf16, #shared> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  tt.return
}
}

// -----
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// Cross-warp consumers preserve the barrier-placement coverage.
#CROSS = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// Same three-buffer loop as no_barrier_between_async_loads_in_pipelined_loop,
// except the local_load is synced through async_wait, as a pipelined kernel's
// is. The sync covers the read, not the refill of the same buffer that follows
// it, so the barrier before the refill must stay.
// CHECK-LABEL: pipelined_loop_refill_after_synced_local_load
tt.func @pipelined_loop_refill_after_synced_local_load(%A: !tt.ptr<f16>, %ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %offset = arith.constant dense<0> : tensor<128x32xi32, #AL>
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable>

  %s0 = ttg.memdesc_index %alloc[%c0_i32] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  // CHECK: amdg.buffer_load_to_local
  // CHECK-NOT: ttg.barrier local
  %ld0 = amdg.buffer_load_to_local %A[%offset] into %s0 : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
  %cg0 = ttg.async_commit_group tokens %ld0

  %s1 = ttg.memdesc_index %alloc[%c1_i32] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
  // CHECK: amdg.buffer_load_to_local
  // CHECK-NOT: ttg.barrier local
  %ld1 = amdg.buffer_load_to_local %A[%offset] into %s1 : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
  %cg1 = ttg.async_commit_group tokens %ld1

  // CHECK: cf.br
  %res = scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%i = %c2_i32) -> (i32) : i32 {
    %si = ttg.memdesc_index %alloc[%i] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    // CHECK-NOT: ttg.barrier local
    // CHECK: amdg.buffer_load_to_local
    %ldi = amdg.buffer_load_to_local %A[%offset] into %si : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
    %cgi = ttg.async_commit_group tokens %ldi

    // CHECK: ttg.async_wait
    // CHECK-NOT: ttg.barrier local
    %w = ttg.async_wait %cgi {num = 2 : i32}

    %ip1 = arith.addi %i, %c1_i32 : i32
    %j = arith.remsi %ip1, %c3_i32 : i32
    %sj = ttg.memdesc_index %alloc[%j] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    // CHECK: ttg.memdesc_index
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: ttg.local_load
    // CHECK-NEXT: ttg.barrier local
    // CHECK-NEXT: amdg.buffer_load_to_local
    %v = ttg.local_load %sj token %w : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #CROSS>
    %ldj = amdg.buffer_load_to_local %A[%offset] into %sj : !tt.ptr<f16>[tensor<128x32xi32, #AL>] -> <128x32xf16, #shared, #smem, mutable>
    %cgj = ttg.async_commit_group tokens %ldj

    // CHECK-NOT: ttg.barrier local
    // CHECK: cf.br
    scf.yield %j : i32
  }
  tt.return
}
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: no_barrier_between_tdm_copies_with_mbarrier
// TDM copies that carry an mbarrier operand send an LDS atomic arrive on
// completion, so the mbarrier already orders them across the workgroup and a
// ttg.barrier between them synchronizes nothing.
tt.func @no_barrier_between_tdm_copies_with_mbarrier(
    %in: !tt.tensordesc<128x128xf16, #shared>,
    %out: !tt.tensordesc<128x128xf16, #shared>) {
  %c0_i32 = arith.constant 0 : i32
  %smem = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
  amdg.init_barrier %bar, 8 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
  // CHECK-NOT: ttg.barrier local
  // CHECK: amdg.async_tdm_copy_global_to_local
  %token = amdg.async_tdm_copy_global_to_local %in into %smem, barrier = %bar : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK-NOT: ttg.barrier local
  // CHECK: amdg.wait_barrier
  amdg.wait_barrier %bar, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
  // CHECK-NOT: ttg.barrier local
  // CHECK: amdg.async_tdm_copy_local_to_global
  amdg.async_tdm_copy_local_to_global %out from %smem, barrier = %bar : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !tt.tensordesc<128x128xf16, #shared>
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.return
  tt.return
}
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: must_barrier_tdm_copy_without_mbarrier
// The store carries no mbarrier operand, so it is not part of the mbarrier
// synchronization and the RAW on the tile is not covered by wait_barrier for
// accesses issued by other waves. The barrier must stay.
tt.func @must_barrier_tdm_copy_without_mbarrier(
    %in: !tt.tensordesc<128x128xf16, #shared>,
    %out: !tt.tensordesc<128x128xf16, #shared>) {
  %c0_i32 = arith.constant 0 : i32
  %smem = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
  amdg.init_barrier %bar, 8 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
  %token = amdg.async_tdm_copy_global_to_local %in into %smem, barrier = %bar : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  // CHECK: amdg.wait_barrier
  amdg.wait_barrier %bar, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: amdg.async_tdm_copy_local_to_global
  amdg.async_tdm_copy_local_to_global %out from %smem : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<128x128xf16, #shared>
  tt.return
}
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: no_barrier_tdm_mbarrier_in_loop
// Collective shape: load a tile, wait, fan it out to several descriptors, wait
// again at the back edge. Both the in-body RAW and the loop-carried WAR are
// ordered by the mbarriers, so no barrier is needed in the loop body.
tt.func @no_barrier_tdm_mbarrier_in_loop(
    %in: !tt.tensordesc<128x128xf16, #shared>,
    %o0: !tt.tensordesc<128x128xf16, #shared>,
    %o1: !tt.tensordesc<128x128xf16, #shared>,
    %ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %smem = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
  %barL = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
  %barS = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
  // A TDM copy arrives once per warp, so the expected count is num-warps times
  // the number of copies on that barrier: one load on barL, two stores on barS.
  amdg.init_barrier %barL, 8 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
  amdg.init_barrier %barS, 16 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
  // CHECK: cf.br
  scf.for %i = %c0_i32 to %ub step %c1_i32 : i32 {
    // CHECK-NOT: ttg.barrier local
    // CHECK: amdg.async_tdm_copy_global_to_local
    %t = amdg.async_tdm_copy_global_to_local %in into %smem, barrier = %barL : !tt.tensordesc<128x128xf16, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    amdg.wait_barrier %barL, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK-NOT: ttg.barrier local
    // CHECK: amdg.async_tdm_copy_local_to_global
    amdg.async_tdm_copy_local_to_global %o0 from %smem, barrier = %barS : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !tt.tensordesc<128x128xf16, #shared>
    amdg.async_tdm_copy_local_to_global %o1 from %smem, barrier = %barS : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !tt.tensordesc<128x128xf16, #shared>
    amdg.wait_barrier %barS, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK-NOT: ttg.barrier local
  }
  // CHECK: tt.return
  tt.return
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// UNSUPPORTED-LABEL: @warp_local_buffer_wait(
// UNSUPPORTED: ttg.async_wait
// UNSUPPORTED-NEXT: ttg.barrier local
// UNSUPPORTED-NEXT: ttg.local_load
// CHECK-LABEL: @warp_local_buffer_wait(
tt.func @warp_local_buffer_wait(%ptr: !tt.ptr<f32>) -> tensor<512xf32, #own> {
  %offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %copy = amdg.buffer_load_to_local %ptr[%offsets] into %buf : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
  %group = ttg.async_commit_group tokens %copy
  // CHECK: ttg.async_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  %wait = ttg.async_wait %group {num = 0 : i32}
  %v = ttg.local_load %buf token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
  tt.return %v : tensor<512xf32, #own>
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @warp_local_async_copy_wait(
tt.func @warp_local_async_copy_wait(%ptr: !tt.ptr<f32>) -> tensor<512xf32, #own> {
  %offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %base = tt.splat %ptr : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #own>
  %ptrs = tt.addptr %base, %offsets : tensor<512x!tt.ptr<f32>, #own>, tensor<512xi32, #own>
  %copy = ttg.async_copy_global_to_local %ptrs, %buf : tensor<512x!tt.ptr<f32>, #own> -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %group = ttg.async_commit_group tokens %copy
  // CHECK: ttg.async_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  %wait = ttg.async_wait %group {num = 0 : i32}
  %v = ttg.local_load %buf token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
  tt.return %v : tensor<512xf32, #own>
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @warp_local_wait_after_barrier(
tt.func @warp_local_wait_after_barrier(%ptr: !tt.ptr<f32>) -> tensor<512xf32, #own> {
  %offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %copy = amdg.buffer_load_to_local %ptr[%offsets] into %buf : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
  %group = ttg.async_commit_group tokens %copy
  // This rendezvous clears pending slices before the copy completes.
  // CHECK: ttg.barrier local
  ttg.barrier local
  // CHECK: ttg.async_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  %wait = ttg.async_wait %group {num = 0 : i32}
  %v = ttg.local_load %buf token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
  tt.return %v : tensor<512xf32, #own>
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @cross_warp_wait_after_barrier(
tt.func @cross_warp_wait_after_barrier(%ptr: !tt.ptr<f32>) -> tensor<512xf32, #cross> {
  %offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %copy = amdg.buffer_load_to_local %ptr[%offsets] into %buf : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
  %group = ttg.async_commit_group tokens %copy
  // This rendezvous clears pending slices before the copy completes.
  // CHECK: ttg.barrier local
  ttg.barrier local
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK: ttg.local_load
  %wait = ttg.async_wait %group {num = 0 : i32}
  %v = ttg.local_load %buf token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #cross>
  tt.return %v : tensor<512xf32, #cross>
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @cross_warp_buffer_wait(
tt.func @cross_warp_buffer_wait(%ptr: !tt.ptr<f32>) -> tensor<512xf32, #cross> {
  %offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %copy = amdg.buffer_load_to_local %ptr[%offsets] into %buf : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
  %group = ttg.async_commit_group tokens %copy
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK: ttg.local_load
  %wait = ttg.async_wait %group {num = 0 : i32}
  %v = ttg.local_load %buf token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #cross>
  tt.return %v : tensor<512xf32, #cross>
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @warp_local_tokenless_wait(
tt.func @warp_local_tokenless_wait(%ptr: !tt.ptr<f32>) -> tensor<512xf32, #own> {
  %offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %copy = amdg.buffer_load_to_local %ptr[%offsets] into %buf : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
  %group = ttg.async_commit_group tokens %copy
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK: ttg.local_load
  %wait = ttg.async_wait %group {num = 0 : i32}
  %v = ttg.local_load %buf : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
  tt.return %v : tensor<512xf32, #own>
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// UNSUPPORTED-LABEL: @warp_local_ordinary_hazards(
// UNSUPPORTED: ttg.local_alloc
// UNSUPPORTED-NEXT: ttg.barrier local
// UNSUPPORTED-NEXT: ttg.local_load
// CHECK-LABEL: @warp_local_ordinary_hazards(
tt.func @warp_local_ordinary_hazards(%x: tensor<512xf32, #own>, %y: tensor<512xf32, #own>) -> tensor<512xf32, #own> {
  // Initialized allocation -> load (RAW), load -> store (WAR), store -> store
  // (WAW), and store -> load (RAW) all have the same physical warp ownership.
  // CHECK: ttg.local_alloc
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.return
  %buf = ttg.local_alloc %x : (tensor<512xf32, #own>) -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %v = ttg.local_load %buf : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
  %sum = arith.addf %v, %y : tensor<512xf32, #own>
  ttg.local_store %sum, %buf : tensor<512xf32, #own> -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  ttg.local_store %y, %buf : tensor<512xf32, #own> -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %r = ttg.local_load %buf : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
  %result = arith.addf %r, %sum : tensor<512xf32, #own>
  tt.return %result : tensor<512xf32, #own>
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @warp_local_then_cross_warp_wait(
tt.func @warp_local_then_cross_warp_wait(%ptr: !tt.ptr<f32>) -> (tensor<512xf32, #own>, tensor<512xf32, #cross>) {
  %offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %copy = amdg.buffer_load_to_local %ptr[%offsets] into %buf : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
  %group = ttg.async_commit_group tokens %copy
  // CHECK: ttg.async_wait
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  %wait = ttg.async_wait %group {num = 0 : i32}
  %a = ttg.local_load %buf token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
  %b = ttg.local_load %buf token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #cross>
  tt.return %a, %b : tensor<512xf32, #own>, tensor<512xf32, #cross>
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.padded_shared<[64:+4] {order = [0], shape = [512]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @warp_local_padded_fallback(
tt.func @warp_local_padded_fallback(%x: tensor<512xf32, #own>) -> tensor<512xf32, #own> {
  %buf = ttg.local_alloc %x : (tensor<512xf32, #own>) -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  %v = ttg.local_load %buf : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
  tt.return %v : tensor<512xf32, #own>
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @warp_local_subslice_select_fallback(
tt.func @warp_local_subslice_select_fallback(%x: tensor<512xf32, #own>, %pred: i1) -> tensor<512xf32, #own> {
  %buf = ttg.local_alloc %x : (tensor<512xf32, #own>) -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %sub = ttg.memdesc_subslice %buf [0] : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %view = arith.select %pred, %sub, %buf : !ttg.memdesc<512xf32, #shared, #smem, mutable>
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  %v = ttg.local_load %view : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
  tt.return %v : tensor<512xf32, #own>
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @warp_local_block_argument_fallback(
tt.func @warp_local_block_argument_fallback(%x: tensor<512xf32, #own>) -> tensor<512xf32, #own> {
  %buf = ttg.local_alloc %x : (tensor<512xf32, #own>) -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  cf.br ^next(%buf : !ttg.memdesc<512xf32, #shared, #smem, mutable>)
^next(%view: !ttg.memdesc<512xf32, #shared, #smem, mutable>):
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  %v = ttg.local_load %view : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
  tt.return %v : tensor<512xf32, #own>
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @warp_local_descriptor_argument_fallback(
tt.func @warp_local_descriptor_argument_fallback(%x: tensor<512xf32, #own>, %buf: !ttg.memdesc<512xf32, #shared, #smem, mutable>) -> tensor<512xf32, #own> {
  ttg.local_store %x, %buf : tensor<512xf32, #own> -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  %v = ttg.local_load %buf : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
  tt.return %v : tensor<512xf32, #own>
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @warp_broadcast_fallback(
tt.func @warp_broadcast_fallback(%x: tensor<512xf32, #own>) -> tensor<512xf32, #own> {
  %buf = ttg.local_alloc %x : (tensor<512xf32, #own>) -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  %v = ttg.local_load %buf : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
  tt.return %v : tensor<512xf32, #own>
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @warp_local_two_slot_refill(
tt.func @warp_local_two_slot_refill(%ptr: !tt.ptr<f32>, %end: i32) -> tensor<512xf32, #own> {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  %width = arith.constant 512 : i32
  %init = arith.constant dense<0.0> : tensor<512xf32, #own>
  %base_offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<2x512xf32, #shared, #smem, mutable>
  %result = scf.for %iv = %zero to %end step %one iter_args(%acc = %init) -> (tensor<512xf32, #own>) : i32 {
    %slot = arith.remui %iv, %two : i32
    %view = ttg.memdesc_index %buf[%slot] : !ttg.memdesc<2x512xf32, #shared, #smem, mutable> -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
    %start = arith.muli %iv, %width : i32
    %starts = tt.splat %start : i32 -> tensor<512xi32, #own>
    %offsets = arith.addi %base_offsets, %starts : tensor<512xi32, #own>
    %copy = amdg.buffer_load_to_local %ptr[%offsets] into %view : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %copy
    // CHECK: ttg.async_wait
    // CHECK-NOT: ttg.barrier local
    // CHECK: ttg.local_load
    %wait = ttg.async_wait %group {num = 0 : i32}
    %v = ttg.local_load %view token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
    %sum = arith.addf %acc, %v : tensor<512xf32, #own>
    // CHECK: ttg.barrier local
    // CHECK-NEXT: amdg.buffer_load_to_local
    %refill = amdg.buffer_load_to_local %ptr[%offsets] into %view : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
    %refill_group = ttg.async_commit_group tokens %refill
    ttg.async_wait %refill_group {num = 0 : i32}
    scf.yield %sum : tensor<512xf32, #own>
  }
  tt.return %result : tensor<512xf32, #own>
}
}

// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @warp_local_all_completion(
tt.func @warp_local_all_completion(%ptr: !tt.ptr<f32>) -> tensor<512xf32, #own> {
  %offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %copy = amdg.buffer_load_to_local %ptr[%offsets] into %buf : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
  %group = ttg.async_commit_group tokens %copy
  // CHECK: amdg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK: ttg.local_load
  %wait = amdg.async_wait %group {num_inst = 0 : i32}
  %v = ttg.local_load %buf token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
  tt.return %v : tensor<512xf32, #own>
}
}


// -----
#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @warp_local_call_fallback(
tt.func @warp_local_call_fallback(%ptr: !tt.ptr<f32>) -> tensor<512xf32, #own> {
  %offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %copy = amdg.buffer_load_to_local %ptr[%offsets] into %buf : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
  %group = ttg.async_commit_group tokens %copy
  tt.call @no_async() : () -> ()
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK: ttg.local_load
  %wait = ttg.async_wait %group {num = 0 : i32}
  %v = ttg.local_load %buf token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #own>
  tt.return %v : tensor<512xf32, #own>
}
  tt.func private @no_async() {
    tt.return
  }
}

// -----

#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @completion_scope_shared_global_read(
tt.func @completion_scope_shared_global_read(%ptr: !tt.ptr<f32>, %bias: !tt.ptr<f32>, %n: i32) -> tensor<512xf32, #cross> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %zero = arith.constant 0.0 : f32
  %offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %copy = amdg.buffer_load_to_local %ptr[%offsets] into %buf : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
  %group = ttg.async_commit_group tokens %copy
  // CHECK: amdg.async_wait
  // CHECK-NEXT: {{.*}} = tt.load
  // CHECK: ttg.barrier local
  // CHECK-NEXT: {{.*}} = ttg.local_load
  %wait = amdg.async_wait %group {num_inst = 0 : i32, ttg.amdg.completion_scope = "shared"}
  %sum = tt.load %bias : !tt.ptr<f32>
  %v = ttg.local_load %buf token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #cross>
  %bcast = tt.splat %sum : f32 -> tensor<512xf32, #cross>
  %result = arith.addf %v, %bcast : tensor<512xf32, #cross>
  tt.return %result : tensor<512xf32, #cross>
}
}

// -----

#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @completion_scope_shared_loop(
tt.func @completion_scope_shared_loop(%ptr: !tt.ptr<f32>, %bias: !tt.ptr<f32>, %n: i32) -> tensor<512xf32, #cross> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %zero = arith.constant 0.0 : f32
  %offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %copy = amdg.buffer_load_to_local %ptr[%offsets] into %buf : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
  %group = ttg.async_commit_group tokens %copy
  // CHECK: amdg.async_wait
  // CHECK-NEXT: cf.br
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.load
  // CHECK: ttg.barrier local
  // CHECK-NEXT: {{.*}} = ttg.local_load
  %wait = amdg.async_wait %group {num_inst = 0 : i32, ttg.amdg.completion_scope = "shared"}
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> f32 : i32 {
    %b = tt.load %bias : !tt.ptr<f32>
    %next = arith.addf %acc, %b : f32
    scf.yield %next : f32
  }
  %v = ttg.local_load %buf token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #cross>
  %bcast = tt.splat %sum : f32 -> tensor<512xf32, #cross>
  %result = arith.addf %v, %bcast : tensor<512xf32, #cross>
  tt.return %result : tensor<512xf32, #cross>
}
}

// -----

#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @completion_scope_all_global_read(
tt.func @completion_scope_all_global_read(%ptr: !tt.ptr<f32>, %bias: !tt.ptr<f32>, %n: i32) -> tensor<512xf32, #cross> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %zero = arith.constant 0.0 : f32
  %offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %copy = amdg.buffer_load_to_local %ptr[%offsets] into %buf : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
  %group = ttg.async_commit_group tokens %copy
  // CHECK: amdg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: {{.*}} = tt.load
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  %wait = amdg.async_wait %group {num_inst = 0 : i32, ttg.amdg.completion_scope = "all"}
  %sum = tt.load %bias : !tt.ptr<f32>
  %v = ttg.local_load %buf token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #cross>
  %bcast = tt.splat %sum : f32 -> tensor<512xf32, #cross>
  %result = arith.addf %v, %bcast : tensor<512xf32, #cross>
  tt.return %result : tensor<512xf32, #cross>
}
}

// -----

#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @completion_scope_all_loop(
tt.func @completion_scope_all_loop(%ptr: !tt.ptr<f32>, %bias: !tt.ptr<f32>, %n: i32) -> tensor<512xf32, #cross> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %zero = arith.constant 0.0 : f32
  %offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %copy = amdg.buffer_load_to_local %ptr[%offsets] into %buf : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
  %group = ttg.async_commit_group tokens %copy
  // CHECK: amdg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: cf.br
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.load
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  %wait = amdg.async_wait %group {num_inst = 0 : i32, ttg.amdg.completion_scope = "all"}
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> f32 : i32 {
    %b = tt.load %bias : !tt.ptr<f32>
    %next = arith.addf %acc, %b : f32
    scf.yield %next : f32
  }
  %v = ttg.local_load %buf token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #cross>
  %bcast = tt.splat %sum : f32 -> tensor<512xf32, #cross>
  %result = arith.addf %v, %bcast : tensor<512xf32, #cross>
  tt.return %result : tensor<512xf32, #cross>
}
}
// -----

#own = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#cross = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @completion_scope_retained_ttg_wait(
tt.func @completion_scope_retained_ttg_wait(%ptr: !tt.ptr<f32>, %bias: !tt.ptr<f32>, %n: i32) -> tensor<512xf32, #cross> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %zero = arith.constant 0.0 : f32
  %offsets = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #own>
  %buf = ttg.local_alloc : () -> !ttg.memdesc<512xf32, #shared, #smem, mutable>
  %copy = amdg.buffer_load_to_local %ptr[%offsets] into %buf : !tt.ptr<f32>[tensor<512xi32, #own>] -> <512xf32, #shared, #smem, mutable>
  %group = ttg.async_commit_group tokens %copy
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: {{.*}} = tt.load
  // CHECK-NOT: ttg.barrier local
  // CHECK: ttg.local_load
  %wait = ttg.async_wait %group {num = 0 : i32, ttg.amdg.completion_scope = "all"}
  %sum = tt.load %bias : !tt.ptr<f32>
  %v = ttg.local_load %buf token %wait : !ttg.memdesc<512xf32, #shared, #smem, mutable> -> tensor<512xf32, #cross>
  %bcast = tt.splat %sum : f32 -> tensor<512xf32, #cross>
  %result = arith.addf %v, %bcast : tensor<512xf32, #cross>
  tt.return %result : tensor<512xf32, #cross>
}
}
