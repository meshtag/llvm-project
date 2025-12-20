//===- DPUToLLVM.cpp - DPU to LLVM dialect lowering ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/DPU/DPUOps.h"
#include "mlir/Dialect/DPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

static LLVM::LLVMFuncOp getOrInsertFunc(ModuleOp module, StringRef name,
                                        Type retType, ArrayRef<Type> argTypes) {
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return func;

  auto funcType = LLVM::LLVMFunctionType::get(retType, argTypes, false);
  OpBuilder builder(module.getBodyRegion());
  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, funcType);
}

struct TidOpLowering : public OpConversionPattern<dpu::TidOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dpu::TidOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto func = getOrInsertFunc(module, "llvm.dpu.tid.i32", op.getType(), {});
    auto call = rewriter.create<LLVM::CallOp>(op.getLoc(), func, ValueRange{});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct SdmaOpLowering : public OpConversionPattern<dpu::SdmaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dpu::SdmaOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto func = getOrInsertFunc(
        module, "llvm.dpu.sdma", voidTy,
        {operands[0].getType(), operands[1].getType(), operands[2].getType()});
    rewriter.create<LLVM::CallOp>(
        op.getLoc(), func, ValueRange{operands[0], operands[1], operands[2]});
    rewriter.eraseOp(op);
    return success();
  }
};

struct LdmaOpLowering : public OpConversionPattern<dpu::LdmaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dpu::LdmaOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto func = getOrInsertFunc(
        module, "llvm.dpu.ldma", voidTy,
        {operands[0].getType(), operands[1].getType(), operands[2].getType()});
    rewriter.create<LLVM::CallOp>(
        op.getLoc(), func, ValueRange{operands[0], operands[1], operands[2]});
    rewriter.eraseOp(op);
    return success();
  }
};

struct SdmaUncheckedOpLowering
    : public OpConversionPattern<dpu::SdmaUncheckedOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dpu::SdmaUncheckedOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto func = getOrInsertFunc(
        module, "llvm.dpu.sdma.unchecked", voidTy,
        {operands[0].getType(), operands[1].getType(), operands[2].getType()});
    rewriter.create<LLVM::CallOp>(
        op.getLoc(), func, ValueRange{operands[0], operands[1], operands[2]});
    rewriter.eraseOp(op);
    return success();
  }
};

struct LdmaUncheckedOpLowering
    : public OpConversionPattern<dpu::LdmaUncheckedOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dpu::LdmaUncheckedOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto func = getOrInsertFunc(
        module, "llvm.dpu.ldma.unchecked", voidTy,
        {operands[0].getType(), operands[1].getType(), operands[2].getType()});
    rewriter.create<LLVM::CallOp>(
        op.getLoc(), func, ValueRange{operands[0], operands[1], operands[2]});
    rewriter.eraseOp(op);
    return success();
  }
};

struct MemResetOpLowering : public OpConversionPattern<dpu::MemResetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dpu::MemResetOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto func = getOrInsertFunc(module, "mem_reset", op.getType(), {});
    auto call = rewriter.create<LLVM::CallOp>(op.getLoc(), func, ValueRange{});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct BarrierWaitOpLowering : public OpConversionPattern<dpu::BarrierWaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dpu::BarrierWaitOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto func = getOrInsertFunc(module, "barrier_wait", voidTy,
                                {operands[0].getType()});
    rewriter.create<LLVM::CallOp>(op.getLoc(), func, ValueRange{operands[0]});
    rewriter.eraseOp(op);
    return success();
  }
};

struct InputArgsOpLowering : public OpConversionPattern<dpu::InputArgsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dpu::InputArgsOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto sym = op.sym();
    auto symRef = FlatSymbolRefAttr::get(sym, rewriter.getContext());
    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(op, op.getType(), symRef);
    return success();
  }
};

struct ConvertDPUToLLVMPass
    : public ConvertDPUToLLVMBase<ConvertDPUToLLVMPass> {
  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext *context = module.getContext();

    context->getOrLoadDialect<LLVM::LLVMDialect>();

    OwningRewritePatternList patterns;
    patterns
        .insert<TidOpLowering, SdmaOpLowering, LdmaOpLowering,
                SdmaUncheckedOpLowering, LdmaUncheckedOpLowering,
                MemResetOpLowering, BarrierWaitOpLowering, InputArgsOpLowering>(
            context);

    ConversionTarget target(*context);
    target.addIllegalDialect<dpu::DPUDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();

    FrozenRewritePatternList frozenPatterns(std::move(patterns));
    if (failed(applyPartialConversion(module, target, frozenPatterns)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertDPUToLLVMPass() {
  return std::make_unique<ConvertDPUToLLVMPass>();
}
