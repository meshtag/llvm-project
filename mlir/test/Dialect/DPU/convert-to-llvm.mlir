// RUN: mlir-opt %s --convert-dpu-to-llvm | FileCheck %s

module {
  llvm.mlir.global external @DPU_INPUT_ARGUMENTS() : i8

  func @kernel(%wram: !llvm.ptr<i8>, %mram: !llvm.ptr<i8, 255>, %size: i32,
               %bar: !llvm.ptr<!llvm.struct<"barrier_t", (i8, i8, i8, i8)>>) {
    %tid = dpu.tid : i32
    dpu.sdma %wram, %mram, %size : !llvm.ptr<i8>, !llvm.ptr<i8, 255>, i32
    dpu.ldma %wram, %mram, %size : !llvm.ptr<i8>, !llvm.ptr<i8, 255>, i32
    dpu.sdma_unchecked %wram, %wram, %size : !llvm.ptr<i8>, !llvm.ptr<i8>, i32
    dpu.ldma_unchecked %wram, %wram, %size : !llvm.ptr<i8>, !llvm.ptr<i8>, i32
    %heap = dpu.mem_reset : !llvm.ptr<i8>
    dpu.barrier_wait %bar : !llvm.ptr<!llvm.struct<"barrier_t", (i8, i8, i8, i8)>>
    %args = dpu.input_args : !llvm.ptr<i8>
    return
  }
}

// CHECK-DAG: llvm.func @llvm.dpu.tid.i32(
// CHECK-DAG: llvm.func @llvm.dpu.sdma(
// CHECK-DAG: llvm.func @llvm.dpu.ldma(
// CHECK-DAG: llvm.func @llvm.dpu.sdma.unchecked(
// CHECK-DAG: llvm.func @llvm.dpu.ldma.unchecked(
// CHECK-DAG: llvm.func @mem_reset
// CHECK-DAG: llvm.func @barrier_wait

// CHECK-LABEL: func @kernel
// CHECK: llvm.call @llvm.dpu.tid.i32(
// CHECK: llvm.call @llvm.dpu.sdma(
// CHECK: llvm.call @llvm.dpu.ldma(
// CHECK: llvm.call @llvm.dpu.sdma.unchecked(
// CHECK: llvm.call @llvm.dpu.ldma.unchecked(
// CHECK: %[[HEAP:.*]] = llvm.call @mem_reset
// CHECK: llvm.call @barrier_wait
// CHECK: %[[ARGS:.*]] = llvm.mlir.addressof @DPU_INPUT_ARGUMENTS : !llvm.ptr<i8>
