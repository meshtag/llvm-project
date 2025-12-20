//===- DPUOps.td - DPU dialect ops -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR DPU dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DPU_OPS_TD
#define MLIR_DIALECT_DPU_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// DPU dialect definition.
def DPU_Dialect : Dialect {
  let name = "dpu";
  let cppNamespace = "::mlir::dpu";
}

class DPU_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<DPU_Dialect, mnemonic, traits> {
  let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

def DPU_TidOp : DPU_Op<"tid", [NoSideEffect]> {
  let summary = "Get the tasklet id";
  let results = (outs I32:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

// WRAM -> MRAM transfer.
def DPU_SdmaOp : DPU_Op<"sdma"> {
  let summary = "Store DMA from WRAM to MRAM";
  let arguments = (ins LLVM_AnyPointer:$src, LLVM_AnyPointer:$dst, I32:$size);
  let assemblyFormat = "$src `,` $dst `,` $size attr-dict `:` type($src) `,` "
                       "type($dst) `,` type($size)";
}

// MRAM -> WRAM transfer.
def DPU_LdmaOp : DPU_Op<"ldma"> {
  let summary = "Load DMA from MRAM to WRAM";
  let arguments = (ins LLVM_AnyPointer:$dst, LLVM_AnyPointer:$src, I32:$size);
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict `:` type($dst) `,` "
                       "type($src) `,` type($size)";
}

// WRAM -> WRAM transfer (unchecked).
def DPU_SdmaUncheckedOp : DPU_Op<"sdma_unchecked"> {
  let summary = "Store DMA from WRAM to WRAM (unchecked)";
  let arguments = (ins LLVM_AnyPointer:$src, LLVM_AnyPointer:$dst, I32:$size);
  let assemblyFormat = "$src `,` $dst `,` $size attr-dict `:` type($src) `,` "
                       "type($dst) `,` type($size)";
}

// WRAM -> WRAM transfer (unchecked).
def DPU_LdmaUncheckedOp : DPU_Op<"ldma_unchecked"> {
  let summary = "Load DMA from WRAM to WRAM (unchecked)";
  let arguments = (ins LLVM_AnyPointer:$dst, LLVM_AnyPointer:$src, I32:$size);
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict `:` type($dst) `,` "
                       "type($src) `,` type($size)";
}

// Reset WRAM heap.
def DPU_MemResetOp : DPU_Op<"mem_reset"> {
  let summary = "Reset WRAM heap";
  let results = (outs LLVM_i8Ptr:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

// Barrier wait.
def DPU_BarrierWaitOp : DPU_Op<"barrier_wait"> {
  let summary = "Wait on a DPU barrier";
  let arguments = (ins LLVM_AnyPointer:$barrier);
  let assemblyFormat = "$barrier attr-dict `:` type($barrier)";
}

// Get the address of DPU input arguments.
def DPU_InputArgsOp : DPU_Op<"input_args", [NoSideEffect]> {
  let summary = "Address of DPU input arguments";
  let arguments = (ins DefaultValuedAttr<StrAttr, "DPU_INPUT_ARGUMENTS">:$sym);
  let results = (outs LLVM_AnyPointer:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

#endif // MLIR_DIALECT_DPU_OPS_TD
