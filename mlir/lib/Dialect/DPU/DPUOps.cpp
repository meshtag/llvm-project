//===- DPUOps.cpp - DPU dialect operations ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DPU/DPUOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::dpu;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

void DPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DPU/DPUOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static LogicalResult verifyI8PtrWithAddrSpace(Operation *op, Type type,
                                              unsigned addrSpace,
                                              StringRef role) {
  auto ptrType = type.dyn_cast<LLVM::LLVMPointerType>();
  if (!ptrType)
    return op->emitOpError("expected ") << role << " to be an LLVM pointer";

  auto elemType = ptrType.getElementType().dyn_cast<IntegerType>();
  if (!elemType || elemType.getWidth() != 8)
    return op->emitOpError("expected ") << role << " to be i8 pointer";

  if (ptrType.getAddressSpace() != addrSpace)
    return op->emitOpError("expected ")
           << role << " to be in addrspace " << addrSpace;
  return success();
}

static bool isBarrierStructType(Type type) {
  auto structType = type.dyn_cast<LLVM::LLVMStructType>();
  if (!structType)
    return false;
  if (structType.isIdentified() && structType.getName() == "barrier_t")
    return true;
  if (structType.isOpaque())
    return false;
  auto body = structType.getBody();
  if (body.size() != 4)
    return false;
  for (Type elem : body) {
    auto intType = elem.dyn_cast<IntegerType>();
    if (!intType || intType.getWidth() != 8)
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

static LogicalResult verify(TidOp op) { return success(); }

static LogicalResult verify(SdmaOp op) {
  if (failed(verifyI8PtrWithAddrSpace(op, op.src().getType(), 0, "src")))
    return failure();
  if (failed(verifyI8PtrWithAddrSpace(op, op.dst().getType(), 255, "dst")))
    return failure();
  return success();
}

static LogicalResult verify(LdmaOp op) {
  if (failed(verifyI8PtrWithAddrSpace(op, op.dst().getType(), 0, "dst")))
    return failure();
  if (failed(verifyI8PtrWithAddrSpace(op, op.src().getType(), 255, "src")))
    return failure();
  return success();
}

static LogicalResult verify(SdmaUncheckedOp op) {
  if (failed(verifyI8PtrWithAddrSpace(op, op.src().getType(), 0, "src")))
    return failure();
  if (failed(verifyI8PtrWithAddrSpace(op, op.dst().getType(), 0, "dst")))
    return failure();
  return success();
}

static LogicalResult verify(LdmaUncheckedOp op) {
  if (failed(verifyI8PtrWithAddrSpace(op, op.dst().getType(), 0, "dst")))
    return failure();
  if (failed(verifyI8PtrWithAddrSpace(op, op.src().getType(), 0, "src")))
    return failure();
  return success();
}

static LogicalResult verify(MemResetOp op) { return success(); }

static LogicalResult verify(BarrierWaitOp op) {
  auto ptrType = op.barrier().getType().dyn_cast<LLVM::LLVMPointerType>();
  if (!ptrType)
    return op.emitOpError("expected barrier to be an LLVM pointer");
  if (!isBarrierStructType(ptrType.getElementType()))
    return op.emitOpError("expected barrier pointer to barrier_t");
  return success();
}

static LogicalResult verify(InputArgsOp op) {
  if (!op.res().getType().isa<LLVM::LLVMPointerType>())
    return op.emitOpError("expected result to be an LLVM pointer");
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/DPU/DPUOps.cpp.inc"
