//===-- AnticipatedExpressions.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/AnticipatedExpressions.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include <map>
#include <set>

using namespace llvm;

void computeAnticipatedExpressions(
    Function &F, std::map<BasicBlock *, std::set<Value *>> &IN,
    std::map<BasicBlock *, std::set<Value *>> &OUT) {
  // Initialize GEN and KILL sets for each basic block
  std::map<BasicBlock *, std::set<Value *>> GEN, KILL;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *BI = dyn_cast<BinaryOperator>(&I)) {
        GEN[&BB].insert(BI);
      }
      // Handle other types of instructions if necessary
    }
  }

  // Initialize OUT sets
  for (BasicBlock &BB : F) {
    OUT[&BB] = GEN[&BB];
  }

  // Iterate until convergence
  bool changed;
  do {
    changed = false;
    for (BasicBlock &BB : F) {
      std::set<Value *> newIN;
      for (BasicBlock *Pred : predecessors(&BB)) {
        for (Value *V : OUT[Pred]) {
          newIN.insert(V);
        }
      }
      IN[&BB] = newIN;

      std::set<Value *> newOUT = IN[&BB];
      for (Value *V : KILL[&BB]) {
        newOUT.erase(V);
      }
      for (Value *V : GEN[&BB]) {
        newOUT.insert(V);
      }

      if (newOUT != OUT[&BB]) {
        OUT[&BB] = newOUT;
        changed = true;
      }
    }
  } while (changed);
}

PreservedAnalyses AnticipatedExpressionsPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  errs() << F.getName() << "\n";

  std::map<BasicBlock *, std::set<Value *>> IN, OUT;
  computeAnticipatedExpressions(F, IN, OUT);

  // Hoist anticipated expressions
  bool changed = false;
  for (BasicBlock &BB : F) {
    for (Value *V : IN[&BB]) {
      if (auto *I = dyn_cast<Instruction>(V)) {
        // Check if the instruction can be hoisted
        if (I->getParent() != &BB && I->hasOneUse()) {
          // // Hoist the instruction to the entry block
          // I->moveBefore(&*BB.getFirstInsertionPt());
          // // NumHoisted++;
          // changed = true;

          // // Update all uses of the hoisted instruction
          // for (auto *U : I->users()) {
          //   if (auto *Phi = dyn_cast<PHINode>(U)) {
          //     // Update the incoming value in the phi node
          //     for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
          //       if (Phi->getIncomingValue(i) == I) {
          //         Phi->setIncomingValue(i, I);
          //       }
          //     }
          //   }
          // }
        }
      }
    }
  }

  // return changed;

  return PreservedAnalyses::all();
}
