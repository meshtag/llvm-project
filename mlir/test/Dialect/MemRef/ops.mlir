// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: func.func @check
func.func @check(%0: memref<12x5xf32>) {
  %r0 = memref.expand_shape %0 [[0, 1], [2]] output_shape [3, 4, 5] :
    memref<12x5xf32> into memref<3x4x5xf32>
  return
}
