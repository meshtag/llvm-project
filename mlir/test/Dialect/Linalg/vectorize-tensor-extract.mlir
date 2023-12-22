// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s


#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @vectorize_1d_tensor_extract(%arg0: tensor<3xf32>, %arg1: tensor<4x3xi32>, %arg2: tensor<4x7x3x2xf32>) -> tensor<4x7x3x2xf32> {
  %1 = linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<4x3xi32>) outs(%arg2 : tensor<4x7x3x2xf32>) {
  ^bb0(%arg3: i32, %arg4: f32):
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = tensor.extract %arg0[%2] : tensor<3xf32>
    linalg.yield %3 : f32
  } -> tensor<4x7x3x2xf32>
  return %1 : tensor<4x7x3x2xf32>
}
// CHECK-LABEL: func.func @vectorize_1d_tensor_extract
// CHECK-SAME:    %[[ARG0:.*]]: tensor<3xf32>
// CHECK-SAME:    %[[ARG1:.*]]: tensor<4x3xi32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[MASK:.*]] = arith.constant dense<true> : vector<4x7x3x2xi1>
// CHECK: %[[PASSTHRU:.*]] = arith.constant dense<0.000000e+00> : vector<4x7x3x2xf32>
// CHECK: %[[V0:.*]] = vector.transfer_read %[[ARG1]]
// CHECK: %[[CAST:.*]] = arith.index_cast %[[V0]]
// CHECK: %[[BROADCAST:.*]] = vector.broadcast %[[CAST]]
// CHECK: %[[INDICES:.*]] = vector.transpose %[[BROADCAST]]
// CHECK: %[[GATHER:.*]] = vector.gather %[[ARG0]][%[[C0]]] [%[[INDICES]]], %[[MASK]], %[[PASSTHRU]]
// CHECK: vector.transfer_write %[[GATHER]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @vectorize_nd_tensor_extract_constant_idx(%arg0: tensor<3x3xf32>, %arg2: tensor<1x1x3xf32>) -> tensor<1x1x3xf32> {
  %c0 = arith.constant 1 : index
  %c1 = arith.constant 2 : index
  %2 = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } outs(%arg2 : tensor<1x1x3xf32>) {
  ^bb0(%arg4: f32):
    %7 = tensor.extract %arg0[%c0, %c1] : tensor<3x3xf32>
    linalg.yield %7 : f32
  } -> tensor<1x1x3xf32>
  return %2 : tensor<1x1x3xf32>
}

// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (0, 0, 0)>
// CHECK-LABEL:   func.func @vectorize_nd_tensor_extract_constant_idx(
// CHECK-SAME:      %[[ARG_0:.*]]: tensor<3x3xf32>,
// CHECK-SAME:      %[[ARG_1:.*]]: tensor<1x1x3xf32>) -> tensor<1x1x3xf32> {
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:       arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[C0_f32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[READ:.*]] = vector.transfer_read  %[[ARG_0]][%[[C1]], %[[C2]]], %[[C0_f32]] {in_bounds = [true, true, true], permutation_map = #[[$MAP]]} : tensor<3x3xf32>, vector<1x1x3xf32>
// CHECK:           %[[C0_4:.*]] = arith.constant 0 : index
// CHECK:           vector.transfer_write %[[READ]], %[[ARG_1]][%[[C0_4]], %[[C0_4]], %[[C0_4]]]  : vector<1x1x3xf32>, tensor<1x1x3xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 { vectorize_nd_extract }  : !transform.any_op
    transform.yield
   }
}

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

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @vectorize_nd_tensor_extract_transfer_read_basic(%arg0: tensor<3x3x3xf32>, %arg2: tensor<1x1x3xf32>) -> tensor<1x1x3xf32> {
  %1 = linalg.generic {
    indexing_maps = [#map1],
    iterator_types = ["parallel", "parallel", "parallel"]
  } outs(%arg2 : tensor<1x1x3xf32>) {
  ^bb0(%arg4: f32):
    %2 = linalg.index 0 : index
    %3 = linalg.index 1 : index
    %4 = linalg.index 2 : index
    %5 = tensor.extract %arg0[%2, %3, %4] : tensor<3x3x3xf32>
    linalg.yield %5 : f32
  } -> tensor<1x1x3xf32>
  return %1 : tensor<1x1x3xf32>
}

// CHECK-LABEL: func.func @vectorize_nd_tensor_extract_transfer_read_basic
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x3x3xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<1x1x3xf32>
// CHECK:   %[[CST:.*]] = arith.constant dense<0> : vector<1x1x3xindex>
// CHECK:   %[[C0_i32:.*]] = arith.constant 0 : i32
// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:   %[[IDX_VEC0:.*]] = vector.shape_cast %[[CST]] : vector<1x1x3xindex> to vector<3xindex>
// CHECK:   %[[IDX1:.*]] = vector.extractelement %[[IDX_VEC0]][%[[C0_i32]] : i32] : vector<3xindex>
// CHECK:   %[[IDX_VEC:.*]] = vector.shape_cast %[[CST]] : vector<1x1x3xindex> to vector<3xindex>
// CHECK:   %[[IDX2:.*]] = vector.extractelement %[[IDX_VEC]][%[[C0_i32]] : i32] : vector<3xindex>
// CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]][%[[IDX1]], %[[IDX2]], %[[C0:.*]]], %[[CST_0]] {in_bounds = [true, true, true]} : tensor<3x3x3xf32>, vector<1x1x3xf32>
// CHECK:   vector.transfer_write %[[READ]], %[[ARG1]][%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true]} : vector<1x1x3xf32>, tensor<1x1x3xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
