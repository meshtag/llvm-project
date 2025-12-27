//===- DPULegalize.cpp - DPU LLVM IR legalization pass -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/DPU/DPULegalize.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

using namespace llvm;

namespace {

constexpr unsigned DpuMramAddrSpace = 255;

static void reportLegalizeError(const Instruction &I, StringRef Msg) {
  std::string Str;
  raw_string_ostream OS(Str);
  OS << "dpu-legalize: " << Msg << " in @" << I.getFunction()->getName()
     << " for instruction: " << I;
  report_fatal_error(OS.str());
}

static bool isThreadIntrinsicName(StringRef Name) {
  return Name == "llvm.nvvm.read.ptx.sreg.tid.x" ||
         Name == "llvm.nvvm.read.ptx.sreg.tid.y" ||
         Name == "llvm.nvvm.read.ptx.sreg.tid.z" ||
         Name == "llvm.nvvm.read.ptx.sreg.ctaid.x" ||
         Name == "llvm.nvvm.read.ptx.sreg.ctaid.y" ||
         Name == "llvm.nvvm.read.ptx.sreg.ctaid.z" ||
         Name == "llvm.nvvm.read.ptx.sreg.ntid.x" ||
         Name == "llvm.nvvm.read.ptx.sreg.ntid.y" ||
         Name == "llvm.nvvm.read.ptx.sreg.ntid.z" ||
         Name == "llvm.nvvm.read.ptx.sreg.nctaid.x" ||
         Name == "llvm.nvvm.read.ptx.sreg.nctaid.y" ||
         Name == "llvm.nvvm.read.ptx.sreg.nctaid.z" ||
         Name == "llvm.nvvm.read.ptx.sreg.laneid" ||
         Name == "llvm.amdgcn.workitem.id.x" ||
         Name == "llvm.amdgcn.workitem.id.y" ||
         Name == "llvm.amdgcn.workitem.id.z" ||
         Name == "llvm.amdgcn.workgroup.id.x" ||
         Name == "llvm.amdgcn.workgroup.id.y" ||
         Name == "llvm.amdgcn.workgroup.id.z";
}

static bool isBarrierIntrinsicName(StringRef Name) {
  return Name == "llvm.nvvm.barrier0" ||
         Name == "llvm.amdgcn.s.barrier";
}

static Value *alignTo8(IRBuilder<> &B, Value *V) {
  auto *Ty = V->getType();
  Value *Add = B.CreateAdd(V, ConstantInt::get(Ty, 7));
  Value *Mask = ConstantInt::getSigned(Ty, -8);
  return B.CreateAnd(Add, Mask);
}

static FunctionCallee getMemAlloc(Module &M) {
  LLVMContext &Ctx = M.getContext();
  return M.getOrInsertFunction("mem_alloc", Type::getInt8PtrTy(Ctx),
                               Type::getInt32Ty(Ctx));
}

static FunctionCallee getLdma(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Type *I8 = Type::getInt8Ty(Ctx);
  return M.getOrInsertFunction(
      "llvm.dpu.ldma", Type::getVoidTy(Ctx), Type::getInt8PtrTy(Ctx),
      PointerType::get(I8, DpuMramAddrSpace), Type::getInt32Ty(Ctx));
}

static FunctionCallee getSdma(Module &M) {
  LLVMContext &Ctx = M.getContext();
  Type *I8 = Type::getInt8Ty(Ctx);
  return M.getOrInsertFunction(
      "llvm.dpu.sdma", Type::getVoidTy(Ctx), Type::getInt8PtrTy(Ctx),
      PointerType::get(I8, DpuMramAddrSpace), Type::getInt32Ty(Ctx));
}

} // namespace

PreservedAnalyses DPULegalizePass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  if (F.isDeclaration())
    return PreservedAnalyses::all();

  Module *M = F.getParent();
  const DataLayout &DL = M->getDataLayout();
  LLVMContext &Ctx = M->getContext();
  bool Changed = false;

  SmallVector<AllocaInst *, 8> Allocas;
  SmallVector<MemCpyInst *, 8> Memcpys;
  SmallVector<CallInst *, 8> ThreadCalls;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *AI = dyn_cast<AllocaInst>(&I)) {
        Allocas.push_back(AI);
        continue;
      }
      if (auto *MI = dyn_cast<MemCpyInst>(&I)) {
        Memcpys.push_back(MI);
        continue;
      }
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;
      Function *Callee = CI->getCalledFunction();
      if (!Callee)
        continue;
      StringRef Name = Callee->getName();
      if (isThreadIntrinsicName(Name)) {
        ThreadCalls.push_back(CI);
        continue;
      }
      if (isBarrierIntrinsicName(Name)) {
        reportLegalizeError(I, "barriers are not supported on DPU yet");
      }
    }
  }

  for (AllocaInst *AI : Allocas) {
    if (AI->getType()->getPointerAddressSpace() != 0) {
      reportLegalizeError(*AI, "alloca in non-zero address space");
    }
    IRBuilder<> B(AI);
    Value *ArraySize = AI->getArraySize();
    Value *ArraySize64 =
        B.CreateZExtOrTrunc(ArraySize, Type::getInt64Ty(Ctx));
    uint64_t ElemSize = DL.getTypeAllocSize(AI->getAllocatedType());
    Value *ElemSize64 = ConstantInt::get(Type::getInt64Ty(Ctx), ElemSize);
    Value *Size64 = B.CreateMul(ArraySize64, ElemSize64);
    if (auto *C = dyn_cast<ConstantInt>(Size64)) {
      if (C->getZExtValue() > std::numeric_limits<uint32_t>::max()) {
        reportLegalizeError(*AI, "alloca size exceeds 32-bit range");
      }
    }
    Value *Size32 = B.CreateTrunc(Size64, Type::getInt32Ty(Ctx));
    Value *Size32Aligned = alignTo8(B, Size32);
    CallInst *Alloc = B.CreateCall(getMemAlloc(*M), Size32Aligned);
    Value *Cast = B.CreateBitCast(Alloc, AI->getType());
    AI->replaceAllUsesWith(Cast);
    AI->eraseFromParent();
    Changed = true;
  }

  for (MemCpyInst *MI : Memcpys) {
    Value *Dst = MI->getRawDest();
    Value *Src = MI->getRawSource();
    unsigned DstAS = Dst->getType()->getPointerAddressSpace();
    unsigned SrcAS = Src->getType()->getPointerAddressSpace();

    bool IsLdma = (SrcAS == DpuMramAddrSpace && DstAS == 0);
    bool IsSdma = (SrcAS == 0 && DstAS == DpuMramAddrSpace);
    if (!IsLdma && !IsSdma)
      continue;

    auto LenConst = dyn_cast<ConstantInt>(MI->getLength());
    if (!LenConst) {
      reportLegalizeError(*MI, "memcpy length must be constant for DPU DMA");
    }
    uint64_t Len = LenConst->getZExtValue();
    if (Len % 8 != 0) {
      reportLegalizeError(*MI, "memcpy length must be 8-byte aligned");
    }
    if (Len > std::numeric_limits<uint32_t>::max()) {
      reportLegalizeError(*MI, "memcpy length exceeds 32-bit range");
    }
    if (auto Align = MI->getDestAlign()) {
      if (Align->value() < 8) {
        reportLegalizeError(*MI, "memcpy dest alignment must be >= 8");
      }
    }
    if (auto Align = MI->getSourceAlign()) {
      if (Align->value() < 8) {
        reportLegalizeError(*MI, "memcpy src alignment must be >= 8");
      }
    }

    IRBuilder<> B(MI);
    Value *Len32 = ConstantInt::get(Type::getInt32Ty(Ctx), Len);
    if (IsLdma) {
      Value *DstCast = B.CreateBitCast(Dst, Type::getInt8PtrTy(Ctx));
      Value *SrcCast = B.CreateBitCast(
          Src, PointerType::get(Type::getInt8Ty(Ctx), DpuMramAddrSpace));
      B.CreateCall(getLdma(*M), {DstCast, SrcCast, Len32});
    } else {
      Value *SrcCast = B.CreateBitCast(Src, Type::getInt8PtrTy(Ctx));
      Value *DstCast = B.CreateBitCast(
          Dst, PointerType::get(Type::getInt8Ty(Ctx), DpuMramAddrSpace));
      B.CreateCall(getSdma(*M), {SrcCast, DstCast, Len32});
    }
    MI->eraseFromParent();
    Changed = true;
  }

  for (CallInst *CI : ThreadCalls) {
    if (!CI->getType()->isIntegerTy()) {
      reportLegalizeError(*CI, "thread id intrinsic returns non-integer type");
    }
    Constant *Zero = ConstantInt::get(CI->getType(), 0);
    CI->replaceAllUsesWith(Zero);
    CI->eraseFromParent();
    Changed = true;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
