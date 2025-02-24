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

#include "llvm/ADT/PostOrderIterator.h" // Required for reverse traversal
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>

#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/IR/Dominators.h" // ✅ Correct header for DominatorTree



using namespace llvm;

void computeAnticipatedExpressions(
    Function &F, std::map<BasicBlock *, std::set<Value *>> &IN,
    std::map<BasicBlock *, std::set<Value *>> &OUT) {
  
  std::map<BasicBlock *, std::set<Value *>> GEN, KILL;

  // Compute GEN sets
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *BI = dyn_cast<BinaryOperator>(&I)) {
        GEN[&BB].insert(BI);
      }
    }
  }

  // Initialize OUT sets
  for (BasicBlock &BB : F) {
    OUT[&BB] = {};
  }

  // Iterate until convergence
  bool changed;
  do {
    changed = false;

    for (BasicBlock &BB : reverse(F)) { // ✅ Use reverse() instead of rbegin()/rend()
      std::set<Value *> newOUT;
      for (BasicBlock *Succ : successors(&BB)) {
        if (!IN[Succ].empty()) {
          if (newOUT.empty()) {
            newOUT = IN[Succ]; // First successor, copy directly
          } else {
            std::set<Value *> temp;
            std::set_intersection(newOUT.begin(), newOUT.end(),
                                  IN[Succ].begin(), IN[Succ].end(),
                                  std::inserter(temp, temp.begin()));
            newOUT = temp; // Intersection with next successor
          }
        }
      }
      OUT[&BB] = newOUT;

      // Compute IN[BB] = GEN[BB] ∪ (OUT[BB] - KILL[BB])
      std::set<Value *> newIN = OUT[&BB];
      for (Value *V : KILL[&BB]) {
        newIN.erase(V);
      }
      for (Value *V : GEN[&BB]) {
        newIN.insert(V);
      }

      if (newIN != IN[&BB]) {
        IN[&BB] = newIN;
        changed = true;
      }
    }
  } while (changed);
}


PreservedAnalyses AnticipatedExpressionsPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  errs() << "Processing function: " << F.getName() << "\n";

  std::map<BasicBlock *, std::set<Value *>> IN, OUT;
  computeAnticipatedExpressions(F, IN, OUT);

  bool changed = false;
  DominatorTree DT(F); // ✅ Correct usage of DominatorTree

  for (BasicBlock &BB : F) {
    for (Value *V : OUT[&BB]) { // ✅ Use OUT instead of IN
      if (auto *I = dyn_cast<Instruction>(V)) {
        if (I->getParent() == &BB) continue; // Already in this block

        // Find the best hoisting point (earliest common dominator)
        BasicBlock *HoistTo = &F.getEntryBlock(); // Default to function entry
        for (BasicBlock *Succ : successors(&BB)) {
          HoistTo = DT.findNearestCommonDominator(HoistTo, Succ);
        }

        // Ensure the hoist target is valid
        if (HoistTo && HoistTo != I->getParent()) {
          I->moveBefore(&*HoistTo->getFirstInsertionPt()); // ✅ Move before first real instruction
          changed = true;
        }
      }
    }
  }

  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
