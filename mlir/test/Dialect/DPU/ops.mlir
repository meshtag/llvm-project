// RUN: mlir-opt -verify-diagnostics %s | mlir-opt | FileCheck %s

module {
  func @ops(%wram: !llvm.ptr<i8>, %mram: !llvm.ptr<i8, 255>, %size: i32,
            %bar: !llvm.ptr<!llvm.struct<"barrier_t", (i8, i8, i8, i8)>>) {
    // CHECK-LABEL: func @ops
    // CHECK: dpu.tid
    %tid = dpu.tid : i32
    // CHECK: dpu.sdma
    dpu.sdma %wram, %mram, %size : !llvm.ptr<i8>, !llvm.ptr<i8, 255>, i32
    // CHECK: dpu.ldma
    dpu.ldma %wram, %mram, %size : !llvm.ptr<i8>, !llvm.ptr<i8, 255>, i32
    // CHECK: dpu.sdma_unchecked
    dpu.sdma_unchecked %wram, %wram, %size : !llvm.ptr<i8>, !llvm.ptr<i8>, i32
    // CHECK: dpu.ldma_unchecked
    dpu.ldma_unchecked %wram, %wram, %size : !llvm.ptr<i8>, !llvm.ptr<i8>, i32
    // CHECK: dpu.mem_reset
    %heap = dpu.mem_reset : !llvm.ptr<i8>
    // CHECK: dpu.barrier_wait
    dpu.barrier_wait %bar : !llvm.ptr<!llvm.struct<"barrier_t", (i8, i8, i8, i8)>>
    // CHECK: dpu.input_args
    %args = dpu.input_args : !llvm.ptr<i8>
    return
  }
}
