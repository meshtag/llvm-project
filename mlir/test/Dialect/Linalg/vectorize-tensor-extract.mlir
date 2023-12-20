// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

// -----

func.func @check(%arg0: tensor<80x16xf32>, %extracted_slice : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%extracted_slice : tensor<4x4xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 0 : index
    %3 = linalg.index 1 : index
    %extracted = tensor.extract %arg0[%2, %3] : tensor<80x16xf32>
    linalg.yield %extracted : f32
  } -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
   }
}
