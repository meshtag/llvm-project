// RUN: mlir-opt -split-input-file -verify-diagnostics %s

module {
  func @sdma_bad_dst(%src: !llvm.ptr<i8>, %dst: !llvm.ptr<i8>, %size: i32) {
    // expected-error @+1 {{expected dst to be in addrspace 255}}
    dpu.sdma %src, %dst, %size : !llvm.ptr<i8>, !llvm.ptr<i8>, i32
    return
  }
}

// -----

module {
  func @sdma_bad_src_type(%src: !llvm.ptr<i32>, %dst: !llvm.ptr<i8, 255>, %size: i32) {
    // expected-error @+1 {{expected src to be i8 pointer}}
    dpu.sdma %src, %dst, %size : !llvm.ptr<i32>, !llvm.ptr<i8, 255>, i32
    return
  }
}

// -----

module {
  func @barrier_bad(%bar: !llvm.ptr<i32>) {
    // expected-error @+1 {{expected barrier pointer to barrier_t}}
    dpu.barrier_wait %bar : !llvm.ptr<i32>
    return
  }
}
