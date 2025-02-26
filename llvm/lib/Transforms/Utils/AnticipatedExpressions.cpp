//===-- AnticipatedExpressions.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/AnticipatedExpressions.h"

#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"

#include <map>
#include <set>

using namespace llvm;

void computeAnticipatedExpressions(
    Function &F, std::map<BasicBlock *, std::set<Value *>> &IN,
    std::map<BasicBlock *, std::set<Value *>> &OUT) {

  std::map<BasicBlock *, std::set<Value *>> V_USE, V_DEF;
  std::set<Value *> allExprs;

  // Collect all expressions (instructions) in the function
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      allExprs.insert(&I);
    }
  }

  // Compute V_USE and V_DEF for each block
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto Inst = dyn_cast<Instruction>(&I)) {
        // Skip branching instructions
        if (Inst->isTerminator() || dyn_cast<PHINode>(Inst)) {
          continue;
        }
        bool canInclude = true;

        // llvm::outs() << "a\n";
        // Inst->dump();
        // llvm::outs() << "\n";

        // Check all operands of the instruction
        for (auto &Operand : Inst->operands()) {
          if (auto OpInst = dyn_cast<Instruction>(Operand)) {

            // llvm::outs() << "b\n";
            // OpInst->dump();
            // llvm::outs() << "\n\n";

            if (OpInst->getParent() == Inst->getParent()) {
              canInclude = false;
              // No need to check further if one operand is in the same block.
              break;
            }
          } else {
            // llvm::outs() << "c\n";
            // Operand.get()->dump();
            // llvm::outs() << "\n\n";
            V_USE[&BB].insert(&I);
            canInclude = false;
            break;
          }
        }

        if (canInclude) {
          V_USE[&BB].insert(&I);
        }
      }
    }
  }

  for (Value *I : allExprs) {
    if (auto *Inst = dyn_cast<Instruction>(I)) {
      for (Use &U : Inst->operands()) {
        if (auto *OperandInst = dyn_cast<Instruction>(U.get())) {
          V_DEF[OperandInst->getParent()].insert(Inst);
        }
      }
    }
  }

  for (BasicBlock &BB : F) {
    IN[&BB] = allExprs;
  }

  bool changed;
  do {
    changed = false;

    std::vector<BasicBlock *> BBList;
    for (BasicBlock &BB : F) {
      BBList.push_back(&BB);
    }
    std::reverse(BBList.begin(), BBList.end());

    for (BasicBlock *BB : BBList) {
      std::set<Value *> newOUT;
      for (BasicBlock *Succ : successors(BB)) {
        if (newOUT.empty()) {
          newOUT = IN[Succ]; // First successor, copy directly
        } else {
          std::set<Value *> temp;
          std::set_intersection(newOUT.begin(), newOUT.end(), IN[Succ].begin(),
                                IN[Succ].end(),
                                std::inserter(temp, temp.begin()));
          newOUT = temp;
        }
      }

      // Compute IN[B] = V_USE[B] ∪ (OUT[B] - V_DEF[B])
      std::set<Value *> newIN = newOUT;
      for (Value *V : V_DEF[BB]) {
        newIN.erase(V);
      }
      for (Value *V : V_USE[BB]) {
        newIN.insert(V);
      }

      if (newIN != IN[BB] || newOUT != OUT[BB]) {
        IN[BB] = newIN;
        OUT[BB] = newOUT;
        changed = true;
      }
    }
  } while (changed);

  //   for (BasicBlock &BB : F) {
  //   BB.dump();
  //   llvm::outs() << "\n";
  //   llvm::outs() << "V_USE now\n";
  //   for (Value *V : V_USE[&BB])
  //     V->dump();
  //   // llvm::outs() << "V_DEF now\n";
  //   // for (Value *V : V_DEF[&BB])
  //   //   V->dump();
  //   // llvm::outs() << "IN now\n";
  //   // for (Value *V : IN[&BB])
  //   //   V->dump();
  //   // llvm::outs() << "\n";
  //   // llvm::outs() << "OUT now\n";
  //   // for (Value *V : OUT[&BB])
  //   //   V->dump();
  //   llvm::outs() << "\n\n\n";
  // }
}

PreservedAnalyses AnticipatedExpressionsPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  std::map<BasicBlock *, std::set<Value *>> IN, OUT;

  bool changed = true;
  EarlyCSEPass CSE;
  ADCEPass ADCE;
  FunctionPassManager FPM;
  // FPM.addPass(InstCombinePass());
  FPM.addPass(GVNPass());
  FPM.run(F, AM);

  while (changed) {
    ADCE.run(F, AM);
    // FPM.run(F, AM);
    PreservedAnalyses PA = CSE.run(F, AM);
    computeAnticipatedExpressions(F, IN, OUT);
    DominatorTree DT1(F);
    changed = false;

    for (BasicBlock &BB : reverse(F)) {
      for (Value *V : IN[&BB]) {
        if (auto *I = dyn_cast<Instruction>(V)) {
          // Find the best hoisting point (earliest common dominator)
          BasicBlock *HoistTo = &BB; // Default to function entry
          for (BasicBlock *Succ : successors(&BB)) {
            HoistTo = DT1.findNearestCommonDominator(HoistTo, Succ);
          }

          // Ensure the hoist target is valid
          if (HoistTo && HoistTo != I->getParent()) {

            for (auto IHostB = HoistTo->begin(); IHostB != HoistTo->end();
                 ++IHostB) {
              Instruction &hoistBlockInstr = *IHostB;
              bool dominatesAllOperands = true;

              for (auto &Operand : I->operands()) {
                if (auto OpInst = dyn_cast<Instruction>(Operand)) {

                  // llvm::outs() << "b\n";
                  // OpInst->dump();
                  // llvm::outs() << "\n\n";
                  if (!DT1.dominates(OpInst, &hoistBlockInstr)) {
                    dominatesAllOperands = false;
                    break;
                  }
                }
              }

              if (dominatesAllOperands) {
                HoistTo->splice(IHostB->getIterator(), I->getParent(),
                                I->getIterator());
                break;
              }
            }

            // HoistTo->splice(HoistTo->getTerminator()->getIterator(),
            //                 I->getParent(), I->getIterator());
            changed = true;
          }
        }
      }
    }

    // outs() << "b\n";
    // F.dump();
    // outs() << "\n\n";
  }

  // outs() << "a\n";
  // F.dump();
  // outs() << "\n\n";

  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
