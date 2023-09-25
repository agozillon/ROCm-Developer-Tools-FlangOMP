//===- OpenMPToLLVMIRTranslation.cpp - Translate OpenMP dialect to LLVM IR-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR OpenMP dialect and LLVM
// IR.
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMPCommon.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/TargetParser/Triple.h"

#include <map>
#include <utility>

using namespace mlir;

namespace {
static llvm::omp::ScheduleKind
convertToScheduleKind(std::optional<omp::ClauseScheduleKind> schedKind) {
  if (!schedKind.has_value())
    return llvm::omp::OMP_SCHEDULE_Default;
  switch (schedKind.value()) {
  case omp::ClauseScheduleKind::Static:
    return llvm::omp::OMP_SCHEDULE_Static;
  case omp::ClauseScheduleKind::Dynamic:
    return llvm::omp::OMP_SCHEDULE_Dynamic;
  case omp::ClauseScheduleKind::Guided:
    return llvm::omp::OMP_SCHEDULE_Guided;
  case omp::ClauseScheduleKind::Auto:
    return llvm::omp::OMP_SCHEDULE_Auto;
  case omp::ClauseScheduleKind::Runtime:
    return llvm::omp::OMP_SCHEDULE_Runtime;
  }
  llvm_unreachable("unhandled schedule clause argument");
}

/// ModuleTranslation stack frame for OpenMP operations. This keeps track of the
/// insertion points for allocas.
class OpenMPAllocaStackFrame
    : public LLVM::ModuleTranslation::StackFrameBase<OpenMPAllocaStackFrame> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpenMPAllocaStackFrame)

  explicit OpenMPAllocaStackFrame(llvm::OpenMPIRBuilder::InsertPointTy allocaIP)
      : allocaInsertPoint(allocaIP) {}
  llvm::OpenMPIRBuilder::InsertPointTy allocaInsertPoint;
};

/// ModuleTranslation stack frame containing the partial mapping between MLIR
/// values and their LLVM IR equivalents.
class OpenMPVarMappingStackFrame
    : public LLVM::ModuleTranslation::StackFrameBase<
          OpenMPVarMappingStackFrame> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpenMPVarMappingStackFrame)

  explicit OpenMPVarMappingStackFrame(
      const DenseMap<Value, llvm::Value *> &mapping)
      : mapping(mapping) {}

  DenseMap<Value, llvm::Value *> mapping;
};
} // namespace

/// Find the insertion point for allocas given the current insertion point for
/// normal operations in the builder.
static llvm::OpenMPIRBuilder::InsertPointTy
findAllocaInsertPoint(llvm::IRBuilderBase &builder,
                      const LLVM::ModuleTranslation &moduleTranslation) {
  // If there is an alloca insertion point on stack, i.e. we are in a nested
  // operation and a specific point was provided by some surrounding operation,
  // use it.
  llvm::OpenMPIRBuilder::InsertPointTy allocaInsertPoint;
  WalkResult walkResult = moduleTranslation.stackWalk<OpenMPAllocaStackFrame>(
      [&](const OpenMPAllocaStackFrame &frame) {
        allocaInsertPoint = frame.allocaInsertPoint;
        return WalkResult::interrupt();
      });
  if (walkResult.wasInterrupted())
    return allocaInsertPoint;

  // Otherwise, insert to the entry block of the surrounding function.
  // If the current IRBuilder InsertPoint is the function's entry, it cannot
  // also be used for alloca insertion which would result in insertion order
  // confusion. Create a new BasicBlock for the Builder and use the entry block
  // for the allocs.
  // TODO: Create a dedicated alloca BasicBlock at function creation such that
  // we do not need to move the current InertPoint here.
  if (builder.GetInsertBlock() ==
      &builder.GetInsertBlock()->getParent()->getEntryBlock()) {
    assert(builder.GetInsertPoint() == builder.GetInsertBlock()->end() &&
           "Assuming end of basic block");
    llvm::BasicBlock *entryBB = llvm::BasicBlock::Create(
        builder.getContext(), "entry", builder.GetInsertBlock()->getParent(),
        builder.GetInsertBlock()->getNextNode());
    builder.CreateBr(entryBB);
    builder.SetInsertPoint(entryBB);
  }

  llvm::BasicBlock &funcEntryBlock =
      builder.GetInsertBlock()->getParent()->getEntryBlock();
  return llvm::OpenMPIRBuilder::InsertPointTy(
      &funcEntryBlock, funcEntryBlock.getFirstInsertionPt());
}

/// Converts the given region that appears within an OpenMP dialect operation to
/// LLVM IR, creating a branch from the `sourceBlock` to the entry block of the
/// region, and a branch from any block with an successor-less OpenMP terminator
/// to `continuationBlock`. Populates `continuationBlockPHIs` with the PHI nodes
/// of the continuation block if provided.
static llvm::BasicBlock *convertOmpOpRegions(
    Region &region, StringRef blockName, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation, LogicalResult &bodyGenStatus,
    SmallVectorImpl<llvm::PHINode *> *continuationBlockPHIs = nullptr) {
  llvm::BasicBlock *continuationBlock =
      splitBB(builder, true, "omp.region.cont");
  llvm::BasicBlock *sourceBlock = builder.GetInsertBlock();

  llvm::LLVMContext &llvmContext = builder.getContext();
  for (Block &bb : region) {
    llvm::BasicBlock *llvmBB = llvm::BasicBlock::Create(
        llvmContext, blockName, builder.GetInsertBlock()->getParent(),
        builder.GetInsertBlock()->getNextNode());
    moduleTranslation.mapBlock(&bb, llvmBB);
  }

  llvm::Instruction *sourceTerminator = sourceBlock->getTerminator();

  // Terminators (namely YieldOp) may be forwarding values to the region that
  // need to be available in the continuation block. Collect the types of these
  // operands in preparation of creating PHI nodes.
  SmallVector<llvm::Type *> continuationBlockPHITypes;
  bool operandsProcessed = false;
  unsigned numYields = 0;
  for (Block &bb : region.getBlocks()) {
    if (omp::YieldOp yield = dyn_cast<omp::YieldOp>(bb.getTerminator())) {
      if (!operandsProcessed) {
        for (unsigned i = 0, e = yield->getNumOperands(); i < e; ++i) {
          continuationBlockPHITypes.push_back(
              moduleTranslation.convertType(yield->getOperand(i).getType()));
        }
        operandsProcessed = true;
      } else {
        assert(continuationBlockPHITypes.size() == yield->getNumOperands() &&
               "mismatching number of values yielded from the region");
        for (unsigned i = 0, e = yield->getNumOperands(); i < e; ++i) {
          llvm::Type *operandType =
              moduleTranslation.convertType(yield->getOperand(i).getType());
          (void)operandType;
          assert(continuationBlockPHITypes[i] == operandType &&
                 "values of mismatching types yielded from the region");
        }
      }
      numYields++;
    }
  }

  // Insert PHI nodes in the continuation block for any values forwarded by the
  // terminators in this region.
  if (!continuationBlockPHITypes.empty())
    assert(
        continuationBlockPHIs &&
        "expected continuation block PHIs if converted regions yield values");
  if (continuationBlockPHIs) {
    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    continuationBlockPHIs->reserve(continuationBlockPHITypes.size());
    builder.SetInsertPoint(continuationBlock, continuationBlock->begin());
    for (llvm::Type *ty : continuationBlockPHITypes)
      continuationBlockPHIs->push_back(builder.CreatePHI(ty, numYields));
  }

  // Convert blocks one by one in topological order to ensure
  // defs are converted before uses.
  SetVector<Block *> blocks =
      LLVM::detail::getTopologicallySortedBlocks(region);
  for (Block *bb : blocks) {
    llvm::BasicBlock *llvmBB = moduleTranslation.lookupBlock(bb);
    // Retarget the branch of the entry block to the entry block of the
    // converted region (regions are single-entry).
    if (bb->isEntryBlock()) {
      assert(sourceTerminator->getNumSuccessors() == 1 &&
             "provided entry block has multiple successors");
      assert(sourceTerminator->getSuccessor(0) == continuationBlock &&
             "ContinuationBlock is not the successor of the entry block");
      sourceTerminator->setSuccessor(0, llvmBB);
    }

    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    if (failed(
            moduleTranslation.convertBlock(*bb, bb->isEntryBlock(), builder))) {
      bodyGenStatus = failure();
      return continuationBlock;
    }

    // Special handling for `omp.yield` and `omp.terminator` (we may have more
    // than one): they return the control to the parent OpenMP dialect operation
    // so replace them with the branch to the continuation block. We handle this
    // here to avoid relying inter-function communication through the
    // ModuleTranslation class to set up the correct insertion point. This is
    // also consistent with MLIR's idiom of handling special region terminators
    // in the same code that handles the region-owning operation.
    Operation *terminator = bb->getTerminator();
    if (isa<omp::TerminatorOp, omp::YieldOp>(terminator)) {
      builder.CreateBr(continuationBlock);

      for (unsigned i = 0, e = terminator->getNumOperands(); i < e; ++i)
        (*continuationBlockPHIs)[i]->addIncoming(
            moduleTranslation.lookupValue(terminator->getOperand(i)), llvmBB);
    }
  }
  // After all blocks have been traversed and values mapped, connect the PHI
  // nodes to the results of preceding blocks.
  LLVM::detail::connectPHINodes(region, moduleTranslation);

  // Remove the blocks and values defined in this region from the mapping since
  // they are not visible outside of this region. This allows the same region to
  // be converted several times, that is cloned, without clashes, and slightly
  // speeds up the lookups.
  moduleTranslation.forgetMapping(region);

  return continuationBlock;
}

/// Convert ProcBindKind from MLIR-generated enum to LLVM enum.
static llvm::omp::ProcBindKind getProcBindKind(omp::ClauseProcBindKind kind) {
  switch (kind) {
  case omp::ClauseProcBindKind::Close:
    return llvm::omp::ProcBindKind::OMP_PROC_BIND_close;
  case omp::ClauseProcBindKind::Master:
    return llvm::omp::ProcBindKind::OMP_PROC_BIND_master;
  case omp::ClauseProcBindKind::Primary:
    return llvm::omp::ProcBindKind::OMP_PROC_BIND_primary;
  case omp::ClauseProcBindKind::Spread:
    return llvm::omp::ProcBindKind::OMP_PROC_BIND_spread;
  }
  llvm_unreachable("Unknown ClauseProcBindKind kind");
}

/// Converts an OpenMP 'master' operation into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpMaster(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP) {
    // MasterOp has only one region associated with it.
    auto &region = cast<omp::MasterOp>(opInst).getRegion();
    builder.restoreIP(codeGenIP);
    convertOmpOpRegions(region, "omp.master.region", builder, moduleTranslation,
                        bodyGenStatus);
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createMaster(
      ompLoc, bodyGenCB, finiCB));
  return success();
}

/// Converts an OpenMP 'critical' operation into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpCritical(Operation &opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  auto criticalOp = cast<omp::CriticalOp>(opInst);
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP) {
    // CriticalOp has only one region associated with it.
    auto &region = cast<omp::CriticalOp>(opInst).getRegion();
    builder.restoreIP(codeGenIP);
    convertOmpOpRegions(region, "omp.critical.region", builder,
                        moduleTranslation, bodyGenStatus);
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
  llvm::Constant *hint = nullptr;

  // If it has a name, it probably has a hint too.
  if (criticalOp.getNameAttr()) {
    // The verifiers in OpenMP Dialect guarentee that all the pointers are
    // non-null
    auto symbolRef = cast<SymbolRefAttr>(criticalOp.getNameAttr());
    auto criticalDeclareOp =
        SymbolTable::lookupNearestSymbolFrom<omp::CriticalDeclareOp>(criticalOp,
                                                                     symbolRef);
    hint = llvm::ConstantInt::get(
        llvm::Type::getInt32Ty(llvmContext),
        static_cast<int>(criticalDeclareOp.getHintVal()));
  }
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createCritical(
      ompLoc, bodyGenCB, finiCB, criticalOp.getName().value_or(""), hint));
  return success();
}

/// Returns a reduction declaration that corresponds to the given reduction
/// operation in the given container. Currently only supports reductions inside
/// WsLoopOp and ParallelOp but can be easily extended as long as the given
/// construct implements getNumReductionVars.
template <typename T>
static std::optional<omp::ReductionDeclareOp>
findReductionDeclInContainer(T container, omp::ReductionOp reduction) {
  for (unsigned i = 0, e = container.getNumReductionVars(); i < e; ++i) {
    if (container.getReductionVars()[i] != reduction.getAccumulator())
      continue;

    SymbolRefAttr reductionSymbol =
        cast<SymbolRefAttr>((*container.getReductions())[i]);
    auto declareOp =
        SymbolTable::lookupNearestSymbolFrom<omp::ReductionDeclareOp>(
            container, reductionSymbol);
    return declareOp;
  }
  return std::nullopt;
}

/// Searches for a reduction in a provided region and the regions
/// it is nested in
static omp::ReductionDeclareOp findReductionDecl(Operation &containerOp,
                                                 omp::ReductionOp reduction) {
  std::optional<omp::ReductionDeclareOp> declareOp = std::nullopt;
  Operation *container = &containerOp;

  while (!declareOp.has_value() && container) {
    // Check if current container is supported for reductions searches
    if (auto par = dyn_cast<omp::ParallelOp>(*container)) {
      declareOp = findReductionDeclInContainer(par, reduction);
    } else if (auto loop = dyn_cast<omp::WsLoopOp>(*container)) {
      declareOp = findReductionDeclInContainer(loop, reduction);
    } else {
      break;
    }

    // See if we can search parent for reductions as well
    container = containerOp.getParentOp();
  }

  assert(declareOp.has_value() &&
         "reduction operation must be associated with a declaration");

  return *declareOp;
}

/// Populates `reductions` with reduction declarations used in the given loop.
template <typename T>
static void
collectReductionDecls(T loop,
                      SmallVectorImpl<omp::ReductionDeclareOp> &reductions) {
  std::optional<ArrayAttr> attr = loop.getReductions();
  if (!attr)
    return;

  reductions.reserve(reductions.size() + loop.getNumReductionVars());
  for (auto symbolRef : attr->getAsRange<SymbolRefAttr>()) {
    reductions.push_back(
        SymbolTable::lookupNearestSymbolFrom<omp::ReductionDeclareOp>(
            loop, symbolRef));
  }
}

/// Translates the blocks contained in the given region and appends them to at
/// the current insertion point of `builder`. The operations of the entry block
/// are appended to the current insertion block, which is not expected to have a
/// terminator. If set, `continuationBlockArgs` is populated with translated
/// values that correspond to the values omp.yield'ed from the region.
static LogicalResult inlineConvertOmpRegions(
    Region &region, StringRef blockName, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation,
    SmallVectorImpl<llvm::Value *> *continuationBlockArgs = nullptr) {
  if (region.empty())
    return success();

  // Special case for single-block regions that don't create additional blocks:
  // insert operations without creating additional blocks.
  if (llvm::hasSingleElement(region)) {
    moduleTranslation.mapBlock(&region.front(), builder.GetInsertBlock());
    if (failed(moduleTranslation.convertBlock(
            region.front(), /*ignoreArguments=*/true, builder)))
      return failure();

    // The continuation arguments are simply the translated terminator operands.
    if (continuationBlockArgs)
      llvm::append_range(
          *continuationBlockArgs,
          moduleTranslation.lookupValues(region.front().back().getOperands()));

    // Drop the mapping that is no longer necessary so that the same region can
    // be processed multiple times.
    moduleTranslation.forgetMapping(region);
    return success();
  }

  LogicalResult bodyGenStatus = success();
  SmallVector<llvm::PHINode *> phis;
  llvm::BasicBlock *continuationBlock = convertOmpOpRegions(
      region, blockName, builder, moduleTranslation, bodyGenStatus, &phis);
  if (failed(bodyGenStatus))
    return failure();
  if (continuationBlockArgs)
    llvm::append_range(*continuationBlockArgs, phis);
  builder.SetInsertPoint(continuationBlock,
                         continuationBlock->getFirstInsertionPt());
  return success();
}

namespace {
/// Owning equivalents of OpenMPIRBuilder::(Atomic)ReductionGen that are used to
/// store lambdas with capture.
using OwningReductionGen = std::function<llvm::OpenMPIRBuilder::InsertPointTy(
    llvm::OpenMPIRBuilder::InsertPointTy, llvm::Value *, llvm::Value *,
    llvm::Value *&)>;
using OwningAtomicReductionGen =
    std::function<llvm::OpenMPIRBuilder::InsertPointTy(
        llvm::OpenMPIRBuilder::InsertPointTy, llvm::Type *, llvm::Value *,
        llvm::Value *)>;
} // namespace

/// Create an OpenMPIRBuilder-compatible reduction generator for the given
/// reduction declaration. The generator uses `builder` but ignores its
/// insertion point.
static OwningReductionGen
makeReductionGen(omp::ReductionDeclareOp decl, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  // The lambda is mutable because we need access to non-const methods of decl
  // (which aren't actually mutating it), and we must capture decl by-value to
  // avoid the dangling reference after the parent function returns.
  OwningReductionGen gen =
      [&, decl](llvm::OpenMPIRBuilder::InsertPointTy insertPoint,
                llvm::Value *lhs, llvm::Value *rhs,
                llvm::Value *&result) mutable {
        Region &reductionRegion = decl.getReductionRegion();
        moduleTranslation.mapValue(reductionRegion.front().getArgument(0), lhs);
        moduleTranslation.mapValue(reductionRegion.front().getArgument(1), rhs);
        builder.restoreIP(insertPoint);
        SmallVector<llvm::Value *> phis;
        if (failed(inlineConvertOmpRegions(reductionRegion,
                                           "omp.reduction.nonatomic.body",
                                           builder, moduleTranslation, &phis)))
          return llvm::OpenMPIRBuilder::InsertPointTy();
        assert(phis.size() == 1);
        result = phis[0];
        return builder.saveIP();
      };
  return gen;
}

/// Create an OpenMPIRBuilder-compatible atomic reduction generator for the
/// given reduction declaration. The generator uses `builder` but ignores its
/// insertion point. Returns null if there is no atomic region available in the
/// reduction declaration.
static OwningAtomicReductionGen
makeAtomicReductionGen(omp::ReductionDeclareOp decl,
                       llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation) {
  if (decl.getAtomicReductionRegion().empty())
    return OwningAtomicReductionGen();

  // The lambda is mutable because we need access to non-const methods of decl
  // (which aren't actually mutating it), and we must capture decl by-value to
  // avoid the dangling reference after the parent function returns.
  OwningAtomicReductionGen atomicGen =
      [&, decl](llvm::OpenMPIRBuilder::InsertPointTy insertPoint, llvm::Type *,
                llvm::Value *lhs, llvm::Value *rhs) mutable {
        Region &atomicRegion = decl.getAtomicReductionRegion();
        moduleTranslation.mapValue(atomicRegion.front().getArgument(0), lhs);
        moduleTranslation.mapValue(atomicRegion.front().getArgument(1), rhs);
        builder.restoreIP(insertPoint);
        SmallVector<llvm::Value *> phis;
        if (failed(inlineConvertOmpRegions(atomicRegion,
                                           "omp.reduction.atomic.body", builder,
                                           moduleTranslation, &phis)))
          return llvm::OpenMPIRBuilder::InsertPointTy();
        assert(phis.empty());
        return builder.saveIP();
      };
  return atomicGen;
}

/// Converts an OpenMP 'ordered' operation into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpOrdered(Operation &opInst, llvm::IRBuilderBase &builder,
                  LLVM::ModuleTranslation &moduleTranslation) {
  auto orderedOp = cast<omp::OrderedOp>(opInst);

  omp::ClauseDepend dependType = *orderedOp.getDependTypeVal();
  bool isDependSource = dependType == omp::ClauseDepend::dependsource;
  unsigned numLoops = *orderedOp.getNumLoopsVal();
  SmallVector<llvm::Value *> vecValues =
      moduleTranslation.lookupValues(orderedOp.getDependVecVars());

  size_t indexVecValues = 0;
  while (indexVecValues < vecValues.size()) {
    SmallVector<llvm::Value *> storeValues;
    storeValues.reserve(numLoops);
    for (unsigned i = 0; i < numLoops; i++) {
      storeValues.push_back(vecValues[indexVecValues]);
      indexVecValues++;
    }
    llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
        findAllocaInsertPoint(builder, moduleTranslation);
    llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
    builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createOrderedDepend(
        ompLoc, allocaIP, numLoops, storeValues, ".cnt.addr", isDependSource));
  }
  return success();
}

/// Converts an OpenMP 'ordered_region' operation into LLVM IR using
/// OpenMPIRBuilder.
static LogicalResult
convertOmpOrderedRegion(Operation &opInst, llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  auto orderedRegionOp = cast<omp::OrderedRegionOp>(opInst);

  // TODO: The code generation for ordered simd directive is not supported yet.
  if (orderedRegionOp.getSimd())
    return failure();

  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP) {
    // OrderedOp has only one region associated with it.
    auto &region = cast<omp::OrderedRegionOp>(opInst).getRegion();
    builder.restoreIP(codeGenIP);
    convertOmpOpRegions(region, "omp.ordered.region", builder,
                        moduleTranslation, bodyGenStatus);
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(
      moduleTranslation.getOpenMPBuilder()->createOrderedThreadsSimd(
          ompLoc, bodyGenCB, finiCB, !orderedRegionOp.getSimd()));
  return bodyGenStatus;
}

static LogicalResult
convertOmpSections(Operation &opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  using StorableBodyGenCallbackTy =
      llvm::OpenMPIRBuilder::StorableBodyGenCallbackTy;

  auto sectionsOp = cast<omp::SectionsOp>(opInst);

  // TODO: Support the following clauses: private, firstprivate, lastprivate,
  // reduction, allocate
  if (!sectionsOp.getReductionVars().empty() || sectionsOp.getReductions() ||
      !sectionsOp.getAllocateVars().empty() ||
      !sectionsOp.getAllocatorsVars().empty())
    return emitError(sectionsOp.getLoc())
           << "reduction and allocate clauses are not supported for sections "
              "construct";

  LogicalResult bodyGenStatus = success();
  SmallVector<StorableBodyGenCallbackTy> sectionCBs;

  for (Operation &op : *sectionsOp.getRegion().begin()) {
    auto sectionOp = dyn_cast<omp::SectionOp>(op);
    if (!sectionOp) // omp.terminator
      continue;

    Region &region = sectionOp.getRegion();
    auto sectionCB = [&region, &builder, &moduleTranslation, &bodyGenStatus](
                         InsertPointTy allocaIP, InsertPointTy codeGenIP) {
      builder.restoreIP(codeGenIP);
      convertOmpOpRegions(region, "omp.section.region", builder,
                          moduleTranslation, bodyGenStatus);
    };
    sectionCBs.push_back(sectionCB);
  }

  // No sections within omp.sections operation - skip generation. This situation
  // is only possible if there is only a terminator operation inside the
  // sections operation
  if (sectionCBs.empty())
    return success();

  assert(isa<omp::SectionOp>(*sectionsOp.getRegion().op_begin()));

  // TODO: Perform appropriate actions according to the data-sharing
  // attribute (shared, private, firstprivate, ...) of variables.
  // Currently defaults to shared.
  auto privCB = [&](InsertPointTy, InsertPointTy codeGenIP, llvm::Value &,
                    llvm::Value &vPtr,
                    llvm::Value *&replacementValue) -> InsertPointTy {
    replacementValue = &vPtr;
    return codeGenIP;
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createSections(
      ompLoc, allocaIP, sectionCBs, privCB, finiCB, false,
      sectionsOp.getNowait()));
  return bodyGenStatus;
}

/// Converts an OpenMP single construct into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpSingle(omp::SingleOp &singleOp, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  LogicalResult bodyGenStatus = success();
  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codegenIP) {
    builder.restoreIP(codegenIP);
    convertOmpOpRegions(singleOp.getRegion(), "omp.single.region", builder,
                        moduleTranslation, bodyGenStatus);
  };
  auto finiCB = [&](InsertPointTy codeGenIP) {};
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createSingle(
      ompLoc, bodyCB, finiCB, singleOp.getNowait(), /*DidIt=*/nullptr));
  return bodyGenStatus;
}

/// Converts an OpenMP task construct into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpTaskOp(omp::TaskOp taskOp, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  LogicalResult bodyGenStatus = success();
  if (taskOp.getUntiedAttr() || taskOp.getMergeableAttr() ||
      taskOp.getInReductions() || taskOp.getPriority() ||
      !taskOp.getAllocateVars().empty()) {
    return taskOp.emitError("unhandled clauses for translation to LLVM IR");
  }
  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codegenIP) {
    // Save the alloca insertion point on ModuleTranslation stack for use in
    // nested regions.
    LLVM::ModuleTranslation::SaveStack<OpenMPAllocaStackFrame> frame(
        moduleTranslation, allocaIP);

    builder.restoreIP(codegenIP);
    convertOmpOpRegions(taskOp.getRegion(), "omp.task.region", builder,
                        moduleTranslation, bodyGenStatus);
  };

  SmallVector<llvm::OpenMPIRBuilder::DependData> dds;
  if (!taskOp.getDependVars().empty() && taskOp.getDepends()) {
    for (auto dep :
         llvm::zip(taskOp.getDependVars(), taskOp.getDepends()->getValue())) {
      llvm::omp::RTLDependenceKindTy type;
      switch (
          cast<mlir::omp::ClauseTaskDependAttr>(std::get<1>(dep)).getValue()) {
      case mlir::omp::ClauseTaskDepend::taskdependin:
        type = llvm::omp::RTLDependenceKindTy::DepIn;
        break;
      // The OpenMP runtime requires that the codegen for 'depend' clause for
      // 'out' dependency kind must be the same as codegen for 'depend' clause
      // with 'inout' dependency.
      case mlir::omp::ClauseTaskDepend::taskdependout:
      case mlir::omp::ClauseTaskDepend::taskdependinout:
        type = llvm::omp::RTLDependenceKindTy::DepInOut;
        break;
      };
      llvm::Value *depVal = moduleTranslation.lookupValue(std::get<0>(dep));
      llvm::OpenMPIRBuilder::DependData dd(type, depVal->getType(), depVal);
      dds.emplace_back(dd);
    }
  }

  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createTask(
      ompLoc, allocaIP, bodyCB, !taskOp.getUntied(),
      moduleTranslation.lookupValue(taskOp.getFinalExpr()),
      moduleTranslation.lookupValue(taskOp.getIfExpr()), dds));
  return bodyGenStatus;
}

/// Converts an OpenMP taskgroup construct into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpTaskgroupOp(omp::TaskGroupOp tgOp, llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  LogicalResult bodyGenStatus = success();
  if (!tgOp.getTaskReductionVars().empty() || !tgOp.getAllocateVars().empty()) {
    return tgOp.emitError("unhandled clauses for translation to LLVM IR");
  }
  auto bodyCB = [&](InsertPointTy allocaIP, InsertPointTy codegenIP) {
    builder.restoreIP(codegenIP);
    convertOmpOpRegions(tgOp.getRegion(), "omp.taskgroup.region", builder,
                        moduleTranslation, bodyGenStatus);
  };
  InsertPointTy allocaIP = findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createTaskgroup(
      ompLoc, allocaIP, bodyCB));
  return bodyGenStatus;
}

/// Allocate space for privatized reduction variables.
template <typename T>
static void
allocReductionVars(T loop, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation,
                   llvm::OpenMPIRBuilder::InsertPointTy &allocaIP,
                   SmallVector<omp::ReductionDeclareOp> &reductionDecls,
                   SmallVector<llvm::Value *> &privateReductionVariables,
                   DenseMap<Value, llvm::Value *> &reductionVariableMap) {
  unsigned numReductions = loop.getNumReductionVars();
  privateReductionVariables.reserve(numReductions);
  if (numReductions != 0) {
    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    builder.restoreIP(allocaIP);
    for (unsigned i = 0; i < numReductions; ++i) {
      llvm::Value *var = builder.CreateAlloca(
          moduleTranslation.convertType(reductionDecls[i].getType()));
      privateReductionVariables.push_back(var);
      reductionVariableMap.try_emplace(loop.getReductionVars()[i], var);
    }
  }
}

/// Collect reduction info
template <typename T>
static void collectReductionInfo(
    T loop, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation,
    SmallVector<omp::ReductionDeclareOp> &reductionDecls,
    SmallVector<OwningReductionGen> &owningReductionGens,
    SmallVector<OwningAtomicReductionGen> &owningAtomicReductionGens,
    const SmallVector<llvm::Value *> &privateReductionVariables,
    SmallVector<llvm::OpenMPIRBuilder::ReductionInfo> &reductionInfos) {
  unsigned numReductions = loop.getNumReductionVars();

  for (unsigned i = 0; i < numReductions; ++i) {
    owningReductionGens.push_back(
        makeReductionGen(reductionDecls[i], builder, moduleTranslation));
    owningAtomicReductionGens.push_back(
        makeAtomicReductionGen(reductionDecls[i], builder, moduleTranslation));
  }

  // Collect the reduction information.
  reductionInfos.reserve(numReductions);
  for (unsigned i = 0; i < numReductions; ++i) {
    llvm::OpenMPIRBuilder::AtomicReductionGenTy atomicGen = nullptr;
    if (owningAtomicReductionGens[i])
      atomicGen = owningAtomicReductionGens[i];
    llvm::Value *variable =
        moduleTranslation.lookupValue(loop.getReductionVars()[i]);
    reductionInfos.push_back(
        {moduleTranslation.convertType(reductionDecls[i].getType()), variable,
         privateReductionVariables[i], owningReductionGens[i], atomicGen});
  }
}

/// Converts an OpenMP workshare loop into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpWsLoop(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  auto loop = cast<omp::WsLoopOp>(opInst);
  // TODO: this should be in the op verifier instead.
  if (loop.getLowerBound().empty())
    return failure();

  // Static is the default.
  auto schedule =
      loop.getScheduleVal().value_or(omp::ClauseScheduleKind::Static);

  // Find the loop configuration.
  llvm::Value *step = moduleTranslation.lookupValue(loop.getStep()[0]);
  llvm::Type *ivType = step->getType();
  llvm::Value *chunk = nullptr;
  if (loop.getScheduleChunkVar()) {
    llvm::Value *chunkVar =
        moduleTranslation.lookupValue(loop.getScheduleChunkVar());
    chunk = builder.CreateSExtOrTrunc(chunkVar, ivType);
  }

  SmallVector<omp::ReductionDeclareOp> reductionDecls;
  collectReductionDecls(loop, reductionDecls);
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);

  SmallVector<llvm::Value *> privateReductionVariables;
  DenseMap<Value, llvm::Value *> reductionVariableMap;
  allocReductionVars(loop, builder, moduleTranslation, allocaIP, reductionDecls,
                     privateReductionVariables, reductionVariableMap);

  // Store the mapping between reduction variables and their private copies on
  // ModuleTranslation stack. It can be then recovered when translating
  // omp.reduce operations in a separate call.
  LLVM::ModuleTranslation::SaveStack<OpenMPVarMappingStackFrame> mappingGuard(
      moduleTranslation, reductionVariableMap);

  // Before the loop, store the initial values of reductions into reduction
  // variables. Although this could be done after allocas, we don't want to mess
  // up with the alloca insertion point.
  for (unsigned i = 0; i < loop.getNumReductionVars(); ++i) {
    SmallVector<llvm::Value *> phis;
    if (failed(inlineConvertOmpRegions(reductionDecls[i].getInitializerRegion(),
                                       "omp.reduction.neutral", builder,
                                       moduleTranslation, &phis)))
      return failure();
    assert(phis.size() == 1 && "expected one value to be yielded from the "
                               "reduction neutral element declaration region");
    builder.CreateStore(phis[0], privateReductionVariables[i]);
  }

  // Set up the source location value for OpenMP runtime.
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  // Generator of the canonical loop body.
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  SmallVector<llvm::CanonicalLoopInfo *> loopInfos;
  SmallVector<llvm::OpenMPIRBuilder::InsertPointTy> bodyInsertPoints;
  LogicalResult bodyGenStatus = success();
  auto bodyGen = [&](llvm::OpenMPIRBuilder::InsertPointTy ip, llvm::Value *iv) {
    // Make sure further conversions know about the induction variable.
    moduleTranslation.mapValue(
        loop.getRegion().front().getArgument(loopInfos.size()), iv);

    // Capture the body insertion point for use in nested loops. BodyIP of the
    // CanonicalLoopInfo always points to the beginning of the entry block of
    // the body.
    bodyInsertPoints.push_back(ip);

    if (loopInfos.size() != loop.getNumLoops() - 1)
      return;

    // Convert the body of the loop.
    builder.restoreIP(ip);
    convertOmpOpRegions(loop.getRegion(), "omp.wsloop.region", builder,
                        moduleTranslation, bodyGenStatus);
  };

  // Delegate actual loop construction to the OpenMP IRBuilder.
  // TODO: this currently assumes WsLoop is semantically similar to SCF loop,
  // i.e. it has a positive step, uses signed integer semantics. Reconsider
  // this code when WsLoop clearly supports more cases.
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  for (unsigned i = 0, e = loop.getNumLoops(); i < e; ++i) {
    llvm::Value *lowerBound =
        moduleTranslation.lookupValue(loop.getLowerBound()[i]);
    llvm::Value *upperBound =
        moduleTranslation.lookupValue(loop.getUpperBound()[i]);
    llvm::Value *step = moduleTranslation.lookupValue(loop.getStep()[i]);

    // Make sure loop trip count are emitted in the preheader of the outermost
    // loop at the latest so that they are all available for the new collapsed
    // loop will be created below.
    llvm::OpenMPIRBuilder::LocationDescription loc = ompLoc;
    llvm::OpenMPIRBuilder::InsertPointTy computeIP = ompLoc.IP;
    if (i != 0) {
      loc = llvm::OpenMPIRBuilder::LocationDescription(bodyInsertPoints.back());
      computeIP = loopInfos.front()->getPreheaderIP();
    }
    loopInfos.push_back(ompBuilder->createCanonicalLoop(
        loc, bodyGen, lowerBound, upperBound, step,
        /*IsSigned=*/true, loop.getInclusive(), computeIP));

    if (failed(bodyGenStatus))
      return failure();
  }

  // Collapse loops. Store the insertion point because LoopInfos may get
  // invalidated.
  llvm::IRBuilderBase::InsertPoint afterIP = loopInfos.front()->getAfterIP();
  llvm::CanonicalLoopInfo *loopInfo =
      ompBuilder->collapseLoops(ompLoc.DL, loopInfos, {});

  allocaIP = findAllocaInsertPoint(builder, moduleTranslation);

  // TODO: Handle doacross loops when the ordered clause has a parameter.
  bool isOrdered = loop.getOrderedVal().has_value();
  std::optional<omp::ScheduleModifier> scheduleModifier =
      loop.getScheduleModifier();
  bool isSimd = loop.getSimdModifier();

  ompBuilder->applyWorkshareLoop(
      ompLoc.DL, loopInfo, allocaIP, !loop.getNowait(),
      convertToScheduleKind(schedule), chunk, isSimd,
      scheduleModifier == omp::ScheduleModifier::monotonic,
      scheduleModifier == omp::ScheduleModifier::nonmonotonic, isOrdered);

  // Continue building IR after the loop. Note that the LoopInfo returned by
  // `collapseLoops` points inside the outermost loop and is intended for
  // potential further loop transformations. Use the insertion point stored
  // before collapsing loops instead.
  builder.restoreIP(afterIP);

  // Process the reductions if required.
  if (loop.getNumReductionVars() == 0)
    return success();

  // Create the reduction generators. We need to own them here because
  // ReductionInfo only accepts references to the generators.
  SmallVector<OwningReductionGen> owningReductionGens;
  SmallVector<OwningAtomicReductionGen> owningAtomicReductionGens;
  SmallVector<llvm::OpenMPIRBuilder::ReductionInfo> reductionInfos;
  collectReductionInfo(loop, builder, moduleTranslation, reductionDecls,
                       owningReductionGens, owningAtomicReductionGens,
                       privateReductionVariables, reductionInfos);

  // The call to createReductions below expects the block to have a
  // terminator. Create an unreachable instruction to serve as terminator
  // and remove it later.
  llvm::UnreachableInst *tempTerminator = builder.CreateUnreachable();
  builder.SetInsertPoint(tempTerminator);
  llvm::OpenMPIRBuilder::InsertPointTy contInsertPoint =
      ompBuilder->createReductions(builder.saveIP(), allocaIP, reductionInfos,
                                   loop.getNowait());
  if (!contInsertPoint.getBlock())
    return loop->emitOpError() << "failed to convert reductions";
  auto nextInsertionPoint =
      ompBuilder->createBarrier(contInsertPoint, llvm::omp::OMPD_for);
  tempTerminator->eraseFromParent();
  builder.restoreIP(nextInsertionPoint);

  return success();
}

/// Converts the OpenMP parallel operation to LLVM IR.
static LogicalResult
convertOmpParallel(omp::ParallelOp opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  LogicalResult bodyGenStatus = success();
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  auto bodyGenCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP) {
    // Collect reduction declarations
    SmallVector<omp::ReductionDeclareOp> reductionDecls;
    collectReductionDecls(opInst, reductionDecls);

    // Allocate reduction vars
    SmallVector<llvm::Value *> privateReductionVariables;
    DenseMap<Value, llvm::Value *> reductionVariableMap;
    allocReductionVars(opInst, builder, moduleTranslation, allocaIP,
                       reductionDecls, privateReductionVariables,
                       reductionVariableMap);

    // Store the mapping between reduction variables and their private copies on
    // ModuleTranslation stack. It can be then recovered when translating
    // omp.reduce operations in a separate call.
    LLVM::ModuleTranslation::SaveStack<OpenMPVarMappingStackFrame> mappingGuard(
        moduleTranslation, reductionVariableMap);

    // Initialize reduction vars
    builder.restoreIP(allocaIP);
    for (unsigned i = 0; i < opInst.getNumReductionVars(); ++i) {
      SmallVector<llvm::Value *> phis;
      if (failed(inlineConvertOmpRegions(
              reductionDecls[i].getInitializerRegion(), "omp.reduction.neutral",
              builder, moduleTranslation, &phis)))
        bodyGenStatus = failure();
      assert(phis.size() == 1 &&
             "expected one value to be yielded from the "
             "reduction neutral element declaration region");
      builder.restoreIP(allocaIP);
      builder.CreateStore(phis[0], privateReductionVariables[i]);
    }

    // Save the alloca insertion point on ModuleTranslation stack for use in
    // nested regions.
    LLVM::ModuleTranslation::SaveStack<OpenMPAllocaStackFrame> frame(
        moduleTranslation, allocaIP);

    // ParallelOp has only one region associated with it.
    builder.restoreIP(codeGenIP);
    auto regionBlock =
        convertOmpOpRegions(opInst.getRegion(), "omp.par.region", builder,
                            moduleTranslation, bodyGenStatus);

    // Process the reductions if required.
    if (opInst.getNumReductionVars() > 0) {
      // Collect reduction info
      SmallVector<OwningReductionGen> owningReductionGens;
      SmallVector<OwningAtomicReductionGen> owningAtomicReductionGens;
      SmallVector<llvm::OpenMPIRBuilder::ReductionInfo> reductionInfos;
      collectReductionInfo(opInst, builder, moduleTranslation, reductionDecls,
                           owningReductionGens, owningAtomicReductionGens,
                           privateReductionVariables, reductionInfos);

      // Move to region cont block
      builder.SetInsertPoint(regionBlock->getTerminator());

      // Generate reductions from info
      llvm::UnreachableInst *tempTerminator = builder.CreateUnreachable();
      builder.SetInsertPoint(tempTerminator);

      llvm::OpenMPIRBuilder::InsertPointTy contInsertPoint =
          ompBuilder->createReductions(builder.saveIP(), allocaIP,
                                       reductionInfos, false);
      if (!contInsertPoint.getBlock()) {
        bodyGenStatus = opInst->emitOpError() << "failed to convert reductions";
        return;
      }

      tempTerminator->eraseFromParent();
      builder.restoreIP(contInsertPoint);
    }
  };

  // TODO: Perform appropriate actions according to the data-sharing
  // attribute (shared, private, firstprivate, ...) of variables.
  // Currently defaults to shared.
  auto privCB = [&](InsertPointTy allocaIP, InsertPointTy codeGenIP,
                    llvm::Value &, llvm::Value &vPtr,
                    llvm::Value *&replacementValue) -> InsertPointTy {
    replacementValue = &vPtr;

    return codeGenIP;
  };

  // TODO: Perform finalization actions for variables. This has to be
  // called for variables which have destructors/finalizers.
  auto finiCB = [&](InsertPointTy codeGenIP) {};

  llvm::Value *ifCond = nullptr;
  if (auto ifExprVar = opInst.getIfExprVar())
    ifCond = moduleTranslation.lookupValue(ifExprVar);
  llvm::Value *numThreads = nullptr;
  if (auto numThreadsVar = opInst.getNumThreadsVar())
    numThreads = moduleTranslation.lookupValue(numThreadsVar);
  auto pbKind = llvm::omp::OMP_PROC_BIND_default;
  if (auto bind = opInst.getProcBindVal())
    pbKind = getProcBindKind(*bind);
  // TODO: Is the Parallel construct cancellable?
  bool isCancellable = false;

  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  builder.restoreIP(
      ompBuilder->createParallel(ompLoc, allocaIP, bodyGenCB, privCB, finiCB,
                                 ifCond, numThreads, pbKind, isCancellable));

  return bodyGenStatus;
}

/// Converts an OpenMP simd loop into LLVM IR using OpenMPIRBuilder.
static LogicalResult
convertOmpSimdLoop(Operation &opInst, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  auto loop = cast<omp::SimdLoopOp>(opInst);

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  // Generator of the canonical loop body.
  // TODO: support error propagation in OpenMPIRBuilder and use it instead of
  // relying on captured variables.
  SmallVector<llvm::CanonicalLoopInfo *> loopInfos;
  SmallVector<llvm::OpenMPIRBuilder::InsertPointTy> bodyInsertPoints;
  LogicalResult bodyGenStatus = success();
  auto bodyGen = [&](llvm::OpenMPIRBuilder::InsertPointTy ip, llvm::Value *iv) {
    // Make sure further conversions know about the induction variable.
    moduleTranslation.mapValue(
        loop.getRegion().front().getArgument(loopInfos.size()), iv);

    // Capture the body insertion point for use in nested loops. BodyIP of the
    // CanonicalLoopInfo always points to the beginning of the entry block of
    // the body.
    bodyInsertPoints.push_back(ip);

    if (loopInfos.size() != loop.getNumLoops() - 1)
      return;

    // Convert the body of the loop.
    builder.restoreIP(ip);
    convertOmpOpRegions(loop.getRegion(), "omp.simdloop.region", builder,
                        moduleTranslation, bodyGenStatus);
  };

  // Delegate actual loop construction to the OpenMP IRBuilder.
  // TODO: this currently assumes SimdLoop is semantically similar to SCF loop,
  // i.e. it has a positive step, uses signed integer semantics. Reconsider
  // this code when SimdLoop clearly supports more cases.
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  for (unsigned i = 0, e = loop.getNumLoops(); i < e; ++i) {
    llvm::Value *lowerBound =
        moduleTranslation.lookupValue(loop.getLowerBound()[i]);
    llvm::Value *upperBound =
        moduleTranslation.lookupValue(loop.getUpperBound()[i]);
    llvm::Value *step = moduleTranslation.lookupValue(loop.getStep()[i]);

    // Make sure loop trip count are emitted in the preheader of the outermost
    // loop at the latest so that they are all available for the new collapsed
    // loop will be created below.
    llvm::OpenMPIRBuilder::LocationDescription loc = ompLoc;
    llvm::OpenMPIRBuilder::InsertPointTy computeIP = ompLoc.IP;
    if (i != 0) {
      loc = llvm::OpenMPIRBuilder::LocationDescription(bodyInsertPoints.back(),
                                                       ompLoc.DL);
      computeIP = loopInfos.front()->getPreheaderIP();
    }
    loopInfos.push_back(ompBuilder->createCanonicalLoop(
        loc, bodyGen, lowerBound, upperBound, step,
        /*IsSigned=*/true, /*Inclusive=*/true, computeIP));

    if (failed(bodyGenStatus))
      return failure();
  }

  // Collapse loops.
  llvm::IRBuilderBase::InsertPoint afterIP = loopInfos.front()->getAfterIP();
  llvm::CanonicalLoopInfo *loopInfo =
      ompBuilder->collapseLoops(ompLoc.DL, loopInfos, {});

  llvm::ConstantInt *simdlen = nullptr;
  if (std::optional<uint64_t> simdlenVar = loop.getSimdlen())
    simdlen = builder.getInt64(simdlenVar.value());

  llvm::ConstantInt *safelen = nullptr;
  if (std::optional<uint64_t> safelenVar = loop.getSafelen())
    safelen = builder.getInt64(safelenVar.value());

  llvm::MapVector<llvm::Value *, llvm::Value *> alignedVars;
  ompBuilder->applySimd(
      loopInfo, alignedVars,
      loop.getIfExpr() ? moduleTranslation.lookupValue(loop.getIfExpr())
                       : nullptr,
      llvm::omp::OrderKind::OMP_ORDER_unknown, simdlen, safelen);

  builder.restoreIP(afterIP);
  return success();
}

/// Convert an Atomic Ordering attribute to llvm::AtomicOrdering.
llvm::AtomicOrdering
convertAtomicOrdering(std::optional<omp::ClauseMemoryOrderKind> ao) {
  if (!ao)
    return llvm::AtomicOrdering::Monotonic; // Default Memory Ordering

  switch (*ao) {
  case omp::ClauseMemoryOrderKind::Seq_cst:
    return llvm::AtomicOrdering::SequentiallyConsistent;
  case omp::ClauseMemoryOrderKind::Acq_rel:
    return llvm::AtomicOrdering::AcquireRelease;
  case omp::ClauseMemoryOrderKind::Acquire:
    return llvm::AtomicOrdering::Acquire;
  case omp::ClauseMemoryOrderKind::Release:
    return llvm::AtomicOrdering::Release;
  case omp::ClauseMemoryOrderKind::Relaxed:
    return llvm::AtomicOrdering::Monotonic;
  }
  llvm_unreachable("Unknown ClauseMemoryOrderKind kind");
}

/// Convert omp.atomic.read operation to LLVM IR.
static LogicalResult
convertOmpAtomicRead(Operation &opInst, llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {

  auto readOp = cast<omp::AtomicReadOp>(opInst);
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  llvm::AtomicOrdering AO = convertAtomicOrdering(readOp.getMemoryOrderVal());
  llvm::Value *x = moduleTranslation.lookupValue(readOp.getX());
  llvm::Value *v = moduleTranslation.lookupValue(readOp.getV());

  llvm::Type *elementType =
      moduleTranslation.convertType(readOp.getElementType());

  llvm::OpenMPIRBuilder::AtomicOpValue V = {v, elementType, false, false};
  llvm::OpenMPIRBuilder::AtomicOpValue X = {x, elementType, false, false};
  builder.restoreIP(ompBuilder->createAtomicRead(ompLoc, X, V, AO));
  return success();
}

/// Converts an omp.atomic.write operation to LLVM IR.
static LogicalResult
convertOmpAtomicWrite(Operation &opInst, llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  auto writeOp = cast<omp::AtomicWriteOp>(opInst);
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::AtomicOrdering ao = convertAtomicOrdering(writeOp.getMemoryOrderVal());
  llvm::Value *expr = moduleTranslation.lookupValue(writeOp.getValue());
  llvm::Value *dest = moduleTranslation.lookupValue(writeOp.getAddress());
  llvm::Type *ty = moduleTranslation.convertType(writeOp.getValue().getType());
  llvm::OpenMPIRBuilder::AtomicOpValue x = {dest, ty, /*isSigned=*/false,
                                            /*isVolatile=*/false};
  builder.restoreIP(ompBuilder->createAtomicWrite(ompLoc, x, expr, ao));
  return success();
}

/// Converts an LLVM dialect binary operation to the corresponding enum value
/// for `atomicrmw` supported binary operation.
llvm::AtomicRMWInst::BinOp convertBinOpToAtomic(Operation &op) {
  return llvm::TypeSwitch<Operation *, llvm::AtomicRMWInst::BinOp>(&op)
      .Case([&](LLVM::AddOp) { return llvm::AtomicRMWInst::BinOp::Add; })
      .Case([&](LLVM::SubOp) { return llvm::AtomicRMWInst::BinOp::Sub; })
      .Case([&](LLVM::AndOp) { return llvm::AtomicRMWInst::BinOp::And; })
      .Case([&](LLVM::OrOp) { return llvm::AtomicRMWInst::BinOp::Or; })
      .Case([&](LLVM::XOrOp) { return llvm::AtomicRMWInst::BinOp::Xor; })
      .Case([&](LLVM::UMaxOp) { return llvm::AtomicRMWInst::BinOp::UMax; })
      .Case([&](LLVM::UMinOp) { return llvm::AtomicRMWInst::BinOp::UMin; })
      .Case([&](LLVM::FAddOp) { return llvm::AtomicRMWInst::BinOp::FAdd; })
      .Case([&](LLVM::FSubOp) { return llvm::AtomicRMWInst::BinOp::FSub; })
      .Default(llvm::AtomicRMWInst::BinOp::BAD_BINOP);
}

/// Converts an OpenMP atomic update operation using OpenMPIRBuilder.
static LogicalResult
convertOmpAtomicUpdate(omp::AtomicUpdateOp &opInst,
                       llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation) {
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  // Convert values and types.
  auto &innerOpList = opInst.getRegion().front().getOperations();
  bool isRegionArgUsed{false}, isXBinopExpr{false};
  llvm::AtomicRMWInst::BinOp binop;
  mlir::Value mlirExpr;
  // Find the binary update operation that uses the region argument
  // and get the expression to update
  for (Operation &innerOp : innerOpList) {
    if (innerOp.getNumOperands() == 2) {
      binop = convertBinOpToAtomic(innerOp);
      if (!llvm::is_contained(innerOp.getOperands(),
                              opInst.getRegion().getArgument(0)))
        continue;
      isRegionArgUsed = true;
      isXBinopExpr = innerOp.getNumOperands() > 0 &&
                     innerOp.getOperand(0) == opInst.getRegion().getArgument(0);
      mlirExpr = (isXBinopExpr ? innerOp.getOperand(1) : innerOp.getOperand(0));
      break;
    }
  }
  if (!isRegionArgUsed)
    return opInst.emitError("no atomic update operation with region argument"
                            " as operand found inside atomic.update region");

  llvm::Value *llvmExpr = moduleTranslation.lookupValue(mlirExpr);
  llvm::Value *llvmX = moduleTranslation.lookupValue(opInst.getX());
  llvm::Type *llvmXElementType = moduleTranslation.convertType(
      opInst.getRegion().getArgument(0).getType());
  llvm::OpenMPIRBuilder::AtomicOpValue llvmAtomicX = {llvmX, llvmXElementType,
                                                      /*isSigned=*/false,
                                                      /*isVolatile=*/false};

  llvm::AtomicOrdering atomicOrdering =
      convertAtomicOrdering(opInst.getMemoryOrderVal());

  // Generate update code.
  LogicalResult updateGenStatus = success();
  auto updateFn = [&opInst, &moduleTranslation, &updateGenStatus](
                      llvm::Value *atomicx,
                      llvm::IRBuilder<> &builder) -> llvm::Value * {
    Block &bb = *opInst.getRegion().begin();
    moduleTranslation.mapValue(*opInst.getRegion().args_begin(), atomicx);
    moduleTranslation.mapBlock(&bb, builder.GetInsertBlock());
    if (failed(moduleTranslation.convertBlock(bb, true, builder))) {
      updateGenStatus = (opInst.emitError()
                         << "unable to convert update operation to llvm IR");
      return nullptr;
    }
    omp::YieldOp yieldop = dyn_cast<omp::YieldOp>(bb.getTerminator());
    assert(yieldop && yieldop.getResults().size() == 1 &&
           "terminator must be omp.yield op and it must have exactly one "
           "argument");
    return moduleTranslation.lookupValue(yieldop.getResults()[0]);
  };

  // Handle ambiguous alloca, if any.
  auto allocaIP = findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(ompBuilder->createAtomicUpdate(
      ompLoc, allocaIP, llvmAtomicX, llvmExpr, atomicOrdering, binop, updateFn,
      isXBinopExpr));
  return updateGenStatus;
}

static LogicalResult
convertOmpAtomicCapture(omp::AtomicCaptureOp atomicCaptureOp,
                        llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation) {
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
  mlir::Value mlirExpr;
  bool isXBinopExpr = false, isPostfixUpdate = false;
  llvm::AtomicRMWInst::BinOp binop = llvm::AtomicRMWInst::BinOp::BAD_BINOP;

  omp::AtomicUpdateOp atomicUpdateOp = atomicCaptureOp.getAtomicUpdateOp();
  omp::AtomicWriteOp atomicWriteOp = atomicCaptureOp.getAtomicWriteOp();

  assert((atomicUpdateOp || atomicWriteOp) &&
         "internal op must be an atomic.update or atomic.write op");

  if (atomicWriteOp) {
    isPostfixUpdate = true;
    mlirExpr = atomicWriteOp.getValue();
  } else {
    isPostfixUpdate = atomicCaptureOp.getSecondOp() ==
                      atomicCaptureOp.getAtomicUpdateOp().getOperation();
    auto &innerOpList = atomicUpdateOp.getRegion().front().getOperations();
    bool isRegionArgUsed{false};
    // Find the binary update operation that uses the region argument
    // and get the expression to update
    for (Operation &innerOp : innerOpList) {
      if (innerOp.getNumOperands() == 2) {
        binop = convertBinOpToAtomic(innerOp);
        if (!llvm::is_contained(innerOp.getOperands(),
                                atomicUpdateOp.getRegion().getArgument(0)))
          continue;
        isRegionArgUsed = true;
        isXBinopExpr =
            innerOp.getNumOperands() > 0 &&
            innerOp.getOperand(0) == atomicUpdateOp.getRegion().getArgument(0);
        mlirExpr =
            (isXBinopExpr ? innerOp.getOperand(1) : innerOp.getOperand(0));
        break;
      }
    }
    if (!isRegionArgUsed)
      return atomicUpdateOp.emitError(
          "no atomic update operation with region argument"
          " as operand found inside atomic.update region");
  }

  llvm::Value *llvmExpr = moduleTranslation.lookupValue(mlirExpr);
  llvm::Value *llvmX =
      moduleTranslation.lookupValue(atomicCaptureOp.getAtomicReadOp().getX());
  llvm::Value *llvmV =
      moduleTranslation.lookupValue(atomicCaptureOp.getAtomicReadOp().getV());
  llvm::Type *llvmXElementType = moduleTranslation.convertType(
      atomicCaptureOp.getAtomicReadOp().getElementType());
  llvm::OpenMPIRBuilder::AtomicOpValue llvmAtomicX = {llvmX, llvmXElementType,
                                                      /*isSigned=*/false,
                                                      /*isVolatile=*/false};
  llvm::OpenMPIRBuilder::AtomicOpValue llvmAtomicV = {llvmV, llvmXElementType,
                                                      /*isSigned=*/false,
                                                      /*isVolatile=*/false};

  llvm::AtomicOrdering atomicOrdering =
      convertAtomicOrdering(atomicCaptureOp.getMemoryOrderVal());

  LogicalResult updateGenStatus = success();
  auto updateFn = [&](llvm::Value *atomicx,
                      llvm::IRBuilder<> &builder) -> llvm::Value * {
    if (atomicWriteOp)
      return moduleTranslation.lookupValue(atomicWriteOp.getValue());
    Block &bb = *atomicUpdateOp.getRegion().begin();
    moduleTranslation.mapValue(*atomicUpdateOp.getRegion().args_begin(),
                               atomicx);
    moduleTranslation.mapBlock(&bb, builder.GetInsertBlock());
    if (failed(moduleTranslation.convertBlock(bb, true, builder))) {
      updateGenStatus = (atomicUpdateOp.emitError()
                         << "unable to convert update operation to llvm IR");
      return nullptr;
    }
    omp::YieldOp yieldop = dyn_cast<omp::YieldOp>(bb.getTerminator());
    assert(yieldop && yieldop.getResults().size() == 1 &&
           "terminator must be omp.yield op and it must have exactly one "
           "argument");
    return moduleTranslation.lookupValue(yieldop.getResults()[0]);
  };

  // Handle ambiguous alloca, if any.
  auto allocaIP = findAllocaInsertPoint(builder, moduleTranslation);
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  builder.restoreIP(ompBuilder->createAtomicCapture(
      ompLoc, allocaIP, llvmAtomicX, llvmAtomicV, llvmExpr, atomicOrdering,
      binop, updateFn, atomicUpdateOp, isPostfixUpdate, isXBinopExpr));
  return updateGenStatus;
}

/// Converts an OpenMP reduction operation using OpenMPIRBuilder. Expects the
/// mapping between reduction variables and their private equivalents to have
/// been stored on the ModuleTranslation stack. Currently only supports
/// reduction within WsLoopOp and ParallelOp, but can be easily extended.
static LogicalResult
convertOmpReductionOp(omp::ReductionOp reductionOp,
                      llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  // Find the declaration that corresponds to the reduction op.
  omp::ReductionDeclareOp declaration;
  Operation *reductionParent = reductionOp->getParentOp();
  if (dyn_cast<omp::ParallelOp>(reductionParent) ||
      dyn_cast<omp::WsLoopOp>(reductionParent)) {
    declaration = findReductionDecl(*reductionParent, reductionOp);
  } else {
    llvm_unreachable("Unhandled reduction container");
  }
  assert(declaration && "could not find reduction declaration");

  // Retrieve the mapping between reduction variables and their private
  // equivalents.
  const DenseMap<Value, llvm::Value *> *reductionVariableMap = nullptr;
  moduleTranslation.stackWalk<OpenMPVarMappingStackFrame>(
      [&](const OpenMPVarMappingStackFrame &frame) {
        if (frame.mapping.contains(reductionOp.getAccumulator())) {
          reductionVariableMap = &frame.mapping;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  assert(reductionVariableMap && "couldn't find private reduction variables");
  // Translate the reduction operation by emitting the body of the corresponding
  // reduction declaration.
  Region &reductionRegion = declaration.getReductionRegion();
  llvm::Value *privateReductionVar =
      reductionVariableMap->lookup(reductionOp.getAccumulator());
  llvm::Value *reductionVal = builder.CreateLoad(
      moduleTranslation.convertType(reductionOp.getOperand().getType()),
      privateReductionVar);

  moduleTranslation.mapValue(reductionRegion.front().getArgument(0),
                             reductionVal);
  moduleTranslation.mapValue(
      reductionRegion.front().getArgument(1),
      moduleTranslation.lookupValue(reductionOp.getOperand()));

  SmallVector<llvm::Value *> phis;
  if (failed(inlineConvertOmpRegions(reductionRegion, "omp.reduction.body",
                                     builder, moduleTranslation, &phis)))
    return failure();
  assert(phis.size() == 1 && "expected one value to be yielded from "
                             "the reduction body declaration region");
  builder.CreateStore(phis[0], privateReductionVar);
  return success();
}

/// Converts an OpenMP Threadprivate operation into LLVM IR using
/// OpenMPIRBuilder.
static LogicalResult
convertOmpThreadprivate(Operation &opInst, llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation) {
  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  auto threadprivateOp = cast<omp::ThreadprivateOp>(opInst);

  Value symAddr = threadprivateOp.getSymAddr();
  auto *symOp = symAddr.getDefiningOp();
  if (!isa<LLVM::AddressOfOp>(symOp))
    return opInst.emitError("Addressing symbol not found");
  LLVM::AddressOfOp addressOfOp = dyn_cast<LLVM::AddressOfOp>(symOp);

  LLVM::GlobalOp global =
      addressOfOp.getGlobal(moduleTranslation.symbolTable());
  llvm::GlobalValue *globalValue = moduleTranslation.lookupGlobal(global);
  llvm::Value *data =
      builder.CreateBitCast(globalValue, builder.getInt8PtrTy());
  llvm::Type *type = globalValue->getValueType();
  llvm::TypeSize typeSize =
      builder.GetInsertBlock()->getModule()->getDataLayout().getTypeStoreSize(
          type);
  llvm::ConstantInt *size = builder.getInt64(typeSize.getFixedValue());
  llvm::StringRef suffix = llvm::StringRef(".cache", 6);
  std::string cacheName = (Twine(global.getSymName()).concat(suffix)).str();
  // Emit runtime function and bitcast its type (i8*) to real data type.
  llvm::Value *callInst =
      moduleTranslation.getOpenMPBuilder()->createCachedThreadPrivate(
          ompLoc, data, size, cacheName);
  llvm::Value *result = builder.CreateBitCast(callInst, globalValue->getType());
  moduleTranslation.mapValue(opInst.getResult(0), result);
  return success();
}

int64_t getSizeInBytes(DataLayout &dl, const mlir::Type &type) {
  if (isa<LLVM::LLVMPointerType>(type))
    return dl.getTypeSize(cast<LLVM::LLVMPointerType>(type).getElementType());
  return dl.getTypeSize(type);
}

// Converts the MLIR enum flag to the LLVM OMPIRBuilder flag
llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind
convertMapCaptureKind(mlir::omp::VariableCaptureKind captureKind) {
  switch (captureKind) {
  case mlir::omp::VariableCaptureKind::ByCopy:
    return llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind::
        OMPTargetVarCaptureByCopy;
    break;
  case mlir::omp::VariableCaptureKind::ByRef:
    return llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind::
        OMPTargetVarCaptureByRef;
    break;

  case mlir::omp::VariableCaptureKind::VLAType:
    return llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind::
        OMPTargetVarCaptureVLAType;
    break;
  case mlir::omp::VariableCaptureKind::This:
    return llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind::
        OMPTargetVarCaptureThis;
    break;
  }
  return llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind::
      OMPTargetVarCaptureByRef;
}

// Get the underlying LLVM type, this will bypass the pointer and
// access the underlying type. Which bypasses llvm's opaque pointers
// to get the underlying type via MLIR.
llvm::Type *getLLVMIRType(mlir::Type inputType,
                          LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Type *type = moduleTranslation.convertType(inputType);
  if (auto pointerType =
          llvm::dyn_cast<mlir::omp::PointerLikeType>(inputType)) {
    if (auto eleType = pointerType.getElementType()) {
      type = moduleTranslation.convertType(eleType);
    }
  }
  return type;
}

llvm::OffloadEntriesInfoManager::OMPTargetDeviceClauseKind
convertToDeviceClauseKind(mlir::omp::DeclareTargetDeviceType deviceClause) {
  switch (deviceClause) {
  case mlir::omp::DeclareTargetDeviceType::host:
    return llvm::OffloadEntriesInfoManager::OMPTargetDeviceClauseHost;
    break;
  case mlir::omp::DeclareTargetDeviceType::nohost:
    return llvm::OffloadEntriesInfoManager::OMPTargetDeviceClauseNoHost;
    break;
  case mlir::omp::DeclareTargetDeviceType::any:
    return llvm::OffloadEntriesInfoManager::OMPTargetDeviceClauseAny;
    break;
  default:
    return llvm::OffloadEntriesInfoManager::OMPTargetDeviceClauseNone;
    break;
  }
}

llvm::OffloadEntriesInfoManager::OMPTargetGlobalVarEntryKind
convertToCaptureClauseKind(
    mlir::omp::DeclareTargetCaptureClause captureClasue) {
  switch (captureClasue) {
  case mlir::omp::DeclareTargetCaptureClause::to:
    return llvm::OffloadEntriesInfoManager::OMPTargetGlobalVarEntryTo;
    break;
  case mlir::omp::DeclareTargetCaptureClause::link:
    return llvm::OffloadEntriesInfoManager::OMPTargetGlobalVarEntryLink;
    break;
  default:
    return llvm::OffloadEntriesInfoManager::OMPTargetGlobalVarEntryNone;
    break;
  }
}

// Returns the reference pointer generated by the lowering of the declare target
// operation in cases where the link clause is used or the to clause is used in
// USM mode.
static llvm::Value *
getRefPtrIfDeclareTarget(mlir::Value const &value,
                         LLVM::ModuleTranslation &moduleTranslation) {
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  // An easier way to do this may just be to keep track of any pointer
  // references and there mapping to their respective operation
  if (isa_and_nonnull<LLVM::AddressOfOp>(value.getDefiningOp())) {
    LLVM::AddressOfOp addressOfOp =
        dyn_cast<LLVM::AddressOfOp>(value.getDefiningOp());
    LLVM::GlobalOp gOp = dyn_cast<LLVM::GlobalOp>(
        addressOfOp->getParentOfType<mlir::ModuleOp>().lookupSymbol(
            addressOfOp.getGlobalName()));

    if (auto declareTargetGlobal =
            llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
                gOp.getOperation())) {

      // In this case, we must utilise the reference pointer generated by the
      // declare target operation, similar to Clang
      if (declareTargetGlobal.isDeclareTarget() &&
          ((declareTargetGlobal.getDeclareTargetCaptureClause() ==
            mlir::omp::DeclareTargetCaptureClause::link) ||
           (declareTargetGlobal.getDeclareTargetCaptureClause() ==
                mlir::omp::DeclareTargetCaptureClause::to &&
            ompBuilder->Config.hasRequiresUnifiedSharedMemory()))) {
        llvm::SmallString<64> suffix;
        {
          llvm::raw_svector_ostream os(suffix);
          if (gOp.getVisibility() == mlir::SymbolTable::Visibility::Private) {
            auto loc = gOp->getLoc()->findInstanceOf<FileLineColLoc>();
            auto fileInfoCallBack = [&loc]() {
              llvm::StringRef filename = loc.getFilename();
              // handle split file from mlir-translate that for some reason
              // manipulates the string to contain other components than
              // filename
              if (filename.contains("within split at")) {
                filename.consume_front("within split at ");
                filename = filename.rsplit(":").first;
              }

              return std::pair<std::string, uint64_t>(filename, loc.getLine());
            };

            os << llvm::format(
                "_%x",
                ompBuilder->getTargetEntryUniqueInfo(fileInfoCallBack).FileID);
          }
          os << "_decl_tgt_ref_ptr";
        }

        if (gOp.getSymName().contains(suffix))
          return moduleTranslation.getLLVMModule()->getNamedValue(
              gOp.getSymName());

        return moduleTranslation.getLLVMModule()->getNamedValue(
            (gOp.getSymName().str() + suffix.str()).str());
      }
    }
  }

  return nullptr;
}

// A small helper structure to contain data gathered
// for map lowering and coalese it into one area and
// avoiding extra computations such as searches in the
// llvm module for lowered mapped varibles or checking
// if something is declare target (and retrieving the
// value).
struct MapData {
  bool isDeclareTarget = false;
  mlir::Operation *mapClause;
  llvm::Value *basePointer;
  llvm::Value *pointer;
  llvm::Value *kernelValue;
  llvm::Type *type;
  uint64_t sizeInBytes;
  llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind captureKind;

  // The vector contains a mapping of MapData -> MembersOf MapData, in most
  // cases for simple mapping with no contained members (e.g. arrays, integers
  // etc.) this vector will be empty (no members to map  as part of the
  // owning object). The vector primarily holds members of
  // structures/classes/derived types (Fortran pointers/alloctables also
  // contain a member of for the pointer data, as the owner of this is
  // a fortran descriptor). As an example the below fortran data type:
  //
  // type :: scalar
  //  integer :: ix = 0
  //  integer,  pointer :: map_ptr(:)
  // end type scalar
  // ...
  // !$omp target map(from:scalar, scalar%map_ptr)
  // ...
  //
  // Would result in map_ptr being an element of the membersOf vector.
  // ix (or other simple data types) at least in this case don't need
  // or undergo specialised mapping, it's implicitly mapped as part of
  // the structure (the pointed to data for map_ptr is the thing that needs
  // to be mapped, which is not part of the structure itself).
  //
  // This format also allows nesting of membersOf, where a membersOf MapData
  // has it's own membersOf and so on and so on.
  //
  // NOTE: has to be a std::vector, llvm::SmallVector's do not appear to like
  // recursive usage of structures.
  std::vector<MapData> membersOf;
};

// Given an mlir::Value we will try to find the corresponding MapData entry
// for it if it exists, recursively going through the MapData we have and
// checking against the clauses VarPtr element (this is the "main" pointer
// component being mapped, the VarPtrPtr is the secondary element indicating
// if the VarPtr has an owner)
static MapData *findMapDataForVariable(llvm::SmallVectorImpl<MapData> &mapData,
                                       Value var) {
  for (MapData &map : mapData) {
    // check top-level MapData
    auto mapOp = mlir::dyn_cast<mlir::omp::MapEntryOp>(map.mapClause);
    if (mapOp.getVarPtr() == var)
      return &map;

    // Check all the members of this MapData, in case of nested
    // dervied-types/structures
    llvm::SmallVector<MapData> members(map.membersOf.begin(),
                                       map.membersOf.end());
    if (MapData *found = findMapDataForVariable(members, var))
      return found;
  }
  return nullptr;
}

void collectMapDataFromMapOperands(llvm::SmallVector<MapData> &mapData,
                                   llvm::SmallVector<Value> &mapOperands,
                                   LLVM::ModuleTranslation &moduleTranslation,
                                   DataLayout &dl) {
  auto getMapData =
      [&](Value val,
          llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind cap,
          Operation *clauseOp) -> MapData {
    MapData map;
    if (llvm::Value *refPtr = getRefPtrIfDeclareTarget(
            val, moduleTranslation)) { // declare target
      map.kernelValue = moduleTranslation.lookupValue(val);
      map.isDeclareTarget = true;
      map.basePointer = refPtr;
      map.pointer = map.kernelValue;
      map.sizeInBytes = getSizeInBytes(dl, val.getType());
    } else { // regular mapped variable
      map.kernelValue = moduleTranslation.lookupValue(val);
      map.isDeclareTarget = false;
      map.basePointer = map.kernelValue;
      map.pointer = map.kernelValue;
      map.sizeInBytes = getSizeInBytes(dl, val.getType());
    }

    map.type = getLLVMIRType(val.getType(), moduleTranslation);
    map.captureKind = cap;
    map.mapClause = clauseOp;

    return map;
  };

  for (mlir::Value mapValue : mapOperands) {
    auto mapOp =
        mlir::dyn_cast<mlir::omp::MapEntryOp>(mapValue.getDefiningOp());

    // It is a mapped member of a larger object
    if (mapOp.getVarPtr() && mapOp.getVarPtrPtr()) {
      // A map clause already exists for the parent, create a
      // MapData and append it to it's membersOf vector, this
      // will now be lowered as a member of the parent.
      if (MapData *parent =
              findMapDataForVariable(mapData, mapOp.getVarPtrPtr())) {
        parent->membersOf.push_back(
            getMapData(mapOp.getVarPtr(),
                       convertMapCaptureKind(mapOp.getMapCaptureType().value_or(
                           mlir::omp::VariableCaptureKind::ByRef)),
                       mapOp));
      } else {
        // The parent has yet to be proccessed whether implicit or
        // explicitly mapped, perhaps a user defined the member mapping
        // earlier, or the larger object was implicitly captured or
        // we have no mapping of the larger structure. So we create an
        // intermediate entry until we can populate it with further data,
        // whether this is the right thing to do for the latter case is up
        // for discussion. For the moment, the only things that need updated
        // when we find the real mapping is the capture type and the clause
        // operation.
        MapData parentDefault = getMapData(
            mapOp.getVarPtrPtr(),
            convertMapCaptureKind(mlir::omp::VariableCaptureKind::ByRef),
            nullptr);
        parentDefault.membersOf.push_back(
            getMapData(mapOp.getVarPtr(),
                       convertMapCaptureKind(mapOp.getMapCaptureType().value_or(
                           mlir::omp::VariableCaptureKind::ByRef)),
                       mapOp));
        mapData.push_back(parentDefault);
      }
    } else {
      // Handles the case where we have created a default MapData already as we
      // encountered a child/member prior to encountering this owning/parent
      // object, essentially just updating relevant fields
      if (MapData *nodeExists =
              findMapDataForVariable(mapData, mapOp.getVarPtr())) {
        nodeExists->mapClause = mapOp;
        nodeExists->captureKind =
            convertMapCaptureKind(mapOp.getMapCaptureType().value_or(
                mlir::omp::VariableCaptureKind::ByRef));
      } else {
        mapData.push_back(
            getMapData(mapOp.getVarPtr(),
                       convertMapCaptureKind(mapOp.getMapCaptureType().value_or(
                           mlir::omp::VariableCaptureKind::ByRef)),
                       mapOp));
      }
    }
  }
}

// TODO: Move this and the Clang version inside of CGOpenMPRuntime.cpp into the
// OMPIRBuilder.cpp
static unsigned getFlagMemberOffset() {
  unsigned Offset = 0;
  for (uint64_t Remain = static_cast<
           std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
           llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_MEMBER_OF);
       !(Remain & 1); Remain = Remain >> 1)
    Offset++;
  return Offset;
}

// TODO: Move this and the Clang version inside of CGOpenMPRuntime.cpp into the
// OMPIRBuilder.cpp
static llvm::omp::OpenMPOffloadMappingFlags getMemberOfFlag(unsigned Position) {
  // Rotate by getFlagMemberOffset() bits.
  return static_cast<llvm::omp::OpenMPOffloadMappingFlags>(
      ((uint64_t)Position + 1) << getFlagMemberOffset());
}

// TODO: Move this and the Clang version inside of CGOpenMPRuntime.cpp into the
// OMPIRBuilder.cpp
static void
setCorrectMemberOfFlag(llvm::omp::OpenMPOffloadMappingFlags &Flags,
                       llvm::omp::OpenMPOffloadMappingFlags MemberOfFlag) {
  // If the entry is PTR_AND_OBJ but has not been marked with the special
  // placeholder value 0xFFFF in the MEMBER_OF field, then it should not be
  // marked as MEMBER_OF.
  if (static_cast<std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
          Flags & llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_PTR_AND_OBJ) &&
      static_cast<std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
          (Flags & llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_MEMBER_OF) !=
          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_MEMBER_OF))
    return;

  // Reset the placeholder value to prepare the flag for the assignment of the
  // proper MEMBER_OF value.
  Flags &= ~llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_MEMBER_OF;
  Flags |= MemberOfFlag;
}

// Fortran pointers and allocatables come with descriptor information alongside
// a pointer to the data itself, currently we map this together as we would a
// regular C/C++ structure with a pointer member that's been specified
// explicitly as a map argument, as the lowered target region from Fortran
// has expectations that it's dealing with this descriptor and the pointer
// or allocatable information to perform runtime checks and calculations.
// A future optimisation pass may allow us to elide this descriptor information
// and simplify the kernel in certain cases where we can prove the bounds are
// known at compile time.
//
// The descriptor information (pointed to by the BoundsOp of the map) allows us
// to take this liberty with the mapping, it contains dynamic information for
// the pointer, such as its runtime size, and type.
//
// TODO: Extend for allocatables N > 1 for allocatables or pointers pointing at
// N-Dimensional data
// TODO: Extend for nested membersOf, this currently only handles one layer of
// nesting and it's only been tested with Fortran's pointers/allocatables/target
// which are effectively a structure with one pointer member and several
// non-complex data members, need to wait on some further Frontend support to
// test user defined types with nested pointers and nested user defined types
// with pointers within
static void processMapWithMembersOf(
    LLVM::ModuleTranslation &moduleTranslation, llvm::IRBuilderBase &builder,
    llvm::OpenMPIRBuilder &ompBuilder, DataLayout &dl,
    llvm::OpenMPIRBuilder::MapInfosTy &combinedInfo, MapData &parentMap) {

  ////////// First Parent Map Segment //////////
  // Map the first segment of our structure (pointer and descriptor), the
  // pointer to the descriptor itself.
  combinedInfo.Types.emplace_back(
      llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TARGET_PARAM);
  combinedInfo.DevicePointers.emplace_back(
      llvm::OpenMPIRBuilder::DeviceInfoTy::None);
  combinedInfo.Names.emplace_back(LLVM::createMappingInformation(
      parentMap.mapClause->getLoc(), ompBuilder));
  combinedInfo.BasePointers.emplace_back(parentMap.basePointer);
  combinedInfo.Pointers.emplace_back(parentMap.pointer);

  // Calculate size of the parent object being mapped based on the
  // addresses at runtime, highAddr - lowAddr = size. This of course
  // doesn't factor in allocated data like pointers, hence the further
  // processing of members specified by users, or in the case of Fortran
  // pointers and allocatables, the mapping of the pointed to data by the
  // descriptor (which itself, is a structure containing runtime information
  // on the dynamically allocated data).
  llvm::Value *lowAddr =
      builder.CreatePointerCast(parentMap.pointer, builder.getPtrTy());
  llvm::Value *highAddr = builder.CreatePointerCast(
      builder.CreateConstGEP1_32(parentMap.type, parentMap.pointer, 1),
      builder.getPtrTy());
  llvm::Value *size = builder.CreateIntCast(
      builder.CreatePtrDiff(builder.getInt8Ty(), highAddr, lowAddr),
      builder.getInt64Ty(),
      /*isSigned=*/false);
  combinedInfo.Sizes.push_back(size);

  ////////// Second Parent Map Segment //////////
  // This creates the initial MEMBER_OF mapping that consists of
  // the parent/top level container (same as above effectively, except with a
  // fixed initial compile time size and seperate maptype which indicates the
  // true mape type (tofrom etc.) and that it is a part of a larger mapping and
  // indicating the link between it and it's members that are also explicitly
  // mapped).
  auto parentClause =
      mlir::dyn_cast<mlir::omp::MapEntryOp>(parentMap.mapClause);
  auto mapFlag =
      llvm::omp::OpenMPOffloadMappingFlags(parentClause.getMapType().value());
  mapFlag &= ~llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TARGET_PARAM;
  llvm::omp::OpenMPOffloadMappingFlags memberOfFlag =
      getMemberOfFlag(combinedInfo.BasePointers.size() - 1);
  setCorrectMemberOfFlag(mapFlag, memberOfFlag);

  combinedInfo.Types.emplace_back(mapFlag);
  combinedInfo.DevicePointers.emplace_back(
      llvm::OpenMPIRBuilder::DeviceInfoTy::None);
  combinedInfo.Names.emplace_back(LLVM::createMappingInformation(
      parentMap.mapClause->getLoc(), ompBuilder));
  combinedInfo.BasePointers.emplace_back(parentMap.basePointer);
  combinedInfo.Pointers.emplace_back(parentMap.pointer);
  combinedInfo.Sizes.emplace_back(builder.getInt64(parentMap.sizeInBytes));

  ////////// Mapping of Members Segment //////////

  for (MapData &mappedMembers : parentMap.membersOf) {
    auto memberClause =
        mlir::dyn_cast<mlir::omp::MapEntryOp>(mappedMembers.mapClause);

    // Same MemberOfFlag to indicate its link with parent and other members of,
    // and we flag that it's part of a pointer and object coupling.
    auto mapFlag =
        llvm::omp::OpenMPOffloadMappingFlags(memberClause.getMapType().value());
    mapFlag &= ~llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TARGET_PARAM;
    setCorrectMemberOfFlag(mapFlag, memberOfFlag);
    mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_PTR_AND_OBJ;
    combinedInfo.Types.emplace_back(mapFlag);
    combinedInfo.DevicePointers.emplace_back(
        llvm::OpenMPIRBuilder::DeviceInfoTy::None);
    combinedInfo.Names.emplace_back(
        LLVM::createMappingInformation(memberClause.getLoc(), ompBuilder));

    // TODO/FIXME: THIS IDX ONLY WORKS FOR POINTER/ALLOCATABLE, NEED TO KNOW
    // WHERE IN ORIGINAL STRUCT/CONTAINER THE MEMBER IS ALLOCATED, extend
    // boundsop to contain and generate structure offset, or is there some way
    // to access this
    auto idx2 =
        std::vector<llvm::Value *>{builder.getInt64(0), builder.getInt64(0)};
    llvm::Value *memberBase = builder.CreateInBoundsGEP(
        parentMap.type, parentMap.pointer, idx2, "member_base");
    combinedInfo.BasePointers.emplace_back(memberBase);

    llvm::Value *loadMember =
        builder.CreateLoad(memberBase->getType(), memberBase);
    auto idx = std::vector<llvm::Value *>{builder.getInt64(0)};
    llvm::Value *memberIdx = builder.CreateInBoundsGEP(
        mappedMembers.type, loadMember, idx, "member_idx");
    combinedInfo.Pointers.emplace_back(memberIdx);

    // NOTE: we must correct off by 1 from an earlier subtraction to the
    // UB that puts it in LLVM-IR index ranges
    auto boundOp = mlir::dyn_cast<mlir::omp::DataBoundsOp>(
        memberClause.getBounds()[0].getDefiningOp());
    llvm::Value *extent = builder.CreateAdd(
        builder.CreateSub(
            moduleTranslation.lookupValue(boundOp.getUpperBound()),
            moduleTranslation.lookupValue(boundOp.getLowerBound())),
        builder.getInt64(1));
    size = builder.CreateMul(
        extent, moduleTranslation.lookupValue(boundOp.getStride()));

    combinedInfo.Sizes.emplace_back(size);
  }
}

static void processMapOp(llvm::IRBuilderBase &builder,
                         LLVM::ModuleTranslation &moduleTranslation,
                         DataLayout &dl,
                         llvm::OpenMPIRBuilder::MapInfosTy &combinedInfo,
                         llvm::SmallVectorImpl<MapData> &mapData,
                         const SmallVector<Value> &devPtrOperands = {},
                         const SmallVector<Value> &devAddrOperands = {}) {
  // Get map clause information.
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  auto fail = [&combinedInfo]() -> void {
    combinedInfo.BasePointers.clear();
    combinedInfo.Pointers.clear();
    combinedInfo.DevicePointers.clear();
    combinedInfo.Sizes.clear();
    combinedInfo.Types.clear();
    combinedInfo.Names.clear();
  };

  for (MapData &mapValue : mapData) {
    if (!mapValue.membersOf.empty()) {
      processMapWithMembersOf(moduleTranslation, builder, *ompBuilder, dl,
                              combinedInfo, mapValue);
      continue;
    }

    auto mapOp = mlir::dyn_cast<mlir::omp::MapEntryOp>(mapValue.mapClause);

    combinedInfo.BasePointers.emplace_back(mapValue.basePointer);
    combinedInfo.Pointers.emplace_back(mapValue.pointer);
    combinedInfo.DevicePointers.emplace_back(
        llvm::OpenMPIRBuilder::DeviceInfoTy::None);
    combinedInfo.Names.emplace_back(
        LLVM::createMappingInformation(mapOp.getLoc(), *ompBuilder));
    combinedInfo.Types.emplace_back(
        llvm::omp::OpenMPOffloadMappingFlags(mapOp.getMapType().value()));
    combinedInfo.Sizes.emplace_back(builder.getInt64(mapValue.sizeInBytes));
  }

  // Device operands are handled seperately and slightly differently to
  // regular maps, at least for the time being
  auto findMapInfo = [&combinedInfo](llvm::Value *val, unsigned &index) {
    index = 0;
    for (llvm::Value *basePtr : combinedInfo.BasePointers) {
      if (basePtr == val)
        return true;
      index++;
    }
    return false;
  };

  auto addDevInfos = [&, fail](auto devOperands, auto devOpType) -> void {
    for (const auto &devOp : devOperands) {
      // TODO: Only LLVMPointerTypes are handled.
      if (!devOp.getType().template isa<LLVM::LLVMPointerType>())
        return fail();

      llvm::Value *mapOpValue = moduleTranslation.lookupValue(devOp);

      // Check if map info is already present for this entry.
      unsigned infoIndex;
      if (findMapInfo(mapOpValue, infoIndex)) {
        combinedInfo.Types[infoIndex] |=
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_RETURN_PARAM;
        combinedInfo.DevicePointers[infoIndex] = devOpType;
      } else {
        combinedInfo.BasePointers.emplace_back(mapOpValue);
        combinedInfo.Pointers.emplace_back(mapOpValue);
        combinedInfo.DevicePointers.emplace_back(devOpType);
        combinedInfo.Names.emplace_back(
            LLVM::createMappingInformation(devOp.getLoc(), *ompBuilder));
        combinedInfo.Types.emplace_back(
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_RETURN_PARAM);
        combinedInfo.Sizes.emplace_back(builder.getInt64(0));
      }
    }
  };

  addDevInfos(devPtrOperands, llvm::OpenMPIRBuilder::DeviceInfoTy::Pointer);
  addDevInfos(devAddrOperands, llvm::OpenMPIRBuilder::DeviceInfoTy::Address);
}

static LogicalResult
convertOmpTargetData(Operation *op, llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Value *ifCond = nullptr;
  int64_t deviceID = llvm::omp::OMP_DEVICEID_UNDEF;
  SmallVector<Value> mapOperands;
  SmallVector<Value> useDevPtrOperands;
  SmallVector<Value> useDevAddrOperands;
  ArrayAttr mapTypes;
  llvm::omp::RuntimeFunction RTLFn;
  DataLayout dl = DataLayout(op->getParentOfType<ModuleOp>());

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  LogicalResult result =
      llvm::TypeSwitch<Operation *, LogicalResult>(op)
          .Case([&](omp::DataOp dataOp) {
            if (auto ifExprVar = dataOp.getIfExpr())
              ifCond = moduleTranslation.lookupValue(ifExprVar);

            if (auto devId = dataOp.getDevice())
              if (auto constOp =
                      dyn_cast<LLVM::ConstantOp>(devId.getDefiningOp()))
                if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                  deviceID = intAttr.getInt();

            mapOperands = dataOp.getMapOperands();
            useDevPtrOperands = dataOp.getUseDevicePtr();
            useDevAddrOperands = dataOp.getUseDeviceAddr();
            return success();
          })
          .Case([&](omp::EnterDataOp enterDataOp) {
            if (enterDataOp.getNowait())
              return failure();

            if (auto ifExprVar = enterDataOp.getIfExpr())
              ifCond = moduleTranslation.lookupValue(ifExprVar);

            if (auto devId = enterDataOp.getDevice())
              if (auto constOp =
                      dyn_cast<LLVM::ConstantOp>(devId.getDefiningOp()))
                if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                  deviceID = intAttr.getInt();
            RTLFn = llvm::omp::OMPRTL___tgt_target_data_begin_mapper;
            mapOperands = enterDataOp.getMapOperands();
            return success();
          })
          .Case([&](omp::ExitDataOp exitDataOp) {
            if (exitDataOp.getNowait())
              return failure();

            if (auto ifExprVar = exitDataOp.getIfExpr())
              ifCond = moduleTranslation.lookupValue(ifExprVar);

            if (auto devId = exitDataOp.getDevice())
              if (auto constOp =
                      dyn_cast<LLVM::ConstantOp>(devId.getDefiningOp()))
                if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                  deviceID = intAttr.getInt();

            RTLFn = llvm::omp::OMPRTL___tgt_target_data_end_mapper;
            mapOperands = exitDataOp.getMapOperands();
            // mapTypes = exitDataOp.getMapTypes();
            return success();
          })
          .Default([&](Operation *op) {
            return op->emitError("unsupported OpenMP operation: ")
                   << op->getName();
          });

  if (failed(result))
    return failure();

  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;

  llvm::SmallVector<MapData> mapData;
  collectMapDataFromMapOperands(mapData, mapOperands, moduleTranslation, dl);

  // Fill up the arrays with all the mapped variables.
  llvm::OpenMPIRBuilder::MapInfosTy combinedInfo;
  auto genMapInfoCB =
      [&](InsertPointTy codeGenIP) -> llvm::OpenMPIRBuilder::MapInfosTy & {
    builder.restoreIP(codeGenIP);
    if (auto DataOp = dyn_cast<omp::DataOp>(op)) {
      processMapOp(builder, moduleTranslation, dl, combinedInfo, mapData,
                   useDevPtrOperands, useDevAddrOperands);
    } else {
      processMapOp(builder, moduleTranslation, dl, combinedInfo, mapData);
    }
    return combinedInfo;
  };

  llvm::OpenMPIRBuilder::TargetDataInfo info(/*RequiresDevicePointerInfo=*/true,
                                             /*SeparateBeginEndCalls=*/true);

  using BodyGenTy = llvm::OpenMPIRBuilder::BodyGenTy;
  LogicalResult bodyGenStatus = success();
  auto bodyGenCB = [&](InsertPointTy codeGenIP, BodyGenTy bodyGenType) {
    assert(isa<omp::DataOp>(op) && "BodyGen requested for non DataOp");
    Region &region = cast<omp::DataOp>(op).getRegion();
    switch (bodyGenType) {
    case BodyGenTy::Priv:
      // Check if any device ptr/addr info is available
      if (!info.DevicePtrInfoMap.empty()) {
        builder.restoreIP(codeGenIP);
        unsigned argIndex = 0;
        for (auto &devPtrOp : useDevPtrOperands) {
          llvm::Value *mapOpValue = moduleTranslation.lookupValue(devPtrOp);
          const auto &arg = region.front().getArgument(argIndex);
          moduleTranslation.mapValue(arg,
                                     info.DevicePtrInfoMap[mapOpValue].second);
          argIndex++;
        }

        for (auto &devAddrOp : useDevAddrOperands) {
          llvm::Value *mapOpValue = moduleTranslation.lookupValue(devAddrOp);
          const auto &arg = region.front().getArgument(argIndex);
          auto *LI = builder.CreateLoad(
              builder.getPtrTy(), info.DevicePtrInfoMap[mapOpValue].second);
          moduleTranslation.mapValue(arg, LI);
          argIndex++;
        }

        bodyGenStatus = inlineConvertOmpRegions(region, "omp.data.region",
                                                builder, moduleTranslation);
      }
      break;
    case BodyGenTy::DupNoPriv:
      break;
    case BodyGenTy::NoPriv:
      // If device info is available then region has already been generated
      if (info.DevicePtrInfoMap.empty()) {
        builder.restoreIP(codeGenIP);
        bodyGenStatus = inlineConvertOmpRegions(region, "omp.data.region",
                                                builder, moduleTranslation);
      }
      break;
    }
    return builder.saveIP();
  };

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  if (isa<omp::DataOp>(op)) {
    builder.restoreIP(ompBuilder->createTargetData(
        ompLoc, allocaIP, builder.saveIP(), builder.getInt64(deviceID), ifCond,
        info, genMapInfoCB, nullptr, bodyGenCB));
  } else {
    builder.restoreIP(ompBuilder->createTargetData(
        ompLoc, allocaIP, builder.saveIP(), builder.getInt64(deviceID), ifCond,
        info, genMapInfoCB, &RTLFn));
  }

return bodyGenStatus;
}

/// Lowers the FlagsAttr which is applied to the module on the device
/// pass when offloading, this attribute contains OpenMP RTL globals that can
/// be passed as flags to the frontend, otherwise they are set to default
LogicalResult convertFlagsAttr(Operation *op, mlir::omp::FlagsAttr attribute,
                               LLVM::ModuleTranslation &moduleTranslation) {
  if (!cast<mlir::ModuleOp>(op))
    return failure();

  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  ompBuilder->createGlobalFlag(
      attribute.getDebugKind() /*LangOpts().OpenMPTargetDebug*/,
      "__omp_rtl_debug_kind");
  ompBuilder->createGlobalFlag(
      attribute
          .getAssumeTeamsOversubscription() /*LangOpts().OpenMPTeamSubscription*/
      ,
      "__omp_rtl_assume_teams_oversubscription");
  ompBuilder->createGlobalFlag(
      attribute
          .getAssumeThreadsOversubscription() /*LangOpts().OpenMPThreadSubscription*/
      ,
      "__omp_rtl_assume_threads_oversubscription");
  ompBuilder->createGlobalFlag(
      attribute.getAssumeNoThreadState() /*LangOpts().OpenMPNoThreadState*/,
      "__omp_rtl_assume_no_thread_state");
  ompBuilder->createGlobalFlag(
      attribute
          .getAssumeNoNestedParallelism() /*LangOpts().OpenMPNoNestedParallelism*/
      ,
      "__omp_rtl_assume_no_nested_parallelism");
  ompBuilder->M.addModuleFlag(llvm::Module::Max, "openmp-device",
                              attribute.getOpenmpDeviceVersion());
  return success();
}

static bool getTargetEntryUniqueInfo(llvm::TargetRegionEntryInfo &targetInfo,
                                     omp::TargetOp targetOp,
                                     llvm::StringRef parentName = "") {
  auto fileLoc = targetOp.getLoc()->findInstanceOf<FileLineColLoc>();

  assert(fileLoc && "No file found from location");
  StringRef fileName = fileLoc.getFilename().getValue();

  llvm::sys::fs::UniqueID id;
  if (auto ec = llvm::sys::fs::getUniqueID(fileName, id)) {
    targetOp.emitError("Unable to get unique ID for file");
    return false;
  }

  uint64_t line = fileLoc.getLine();
  targetInfo = llvm::TargetRegionEntryInfo(parentName, id.getDevice(),
                                           id.getFile(), line);
  return true;
}

static bool targetOpSupported(Operation &opInst) {
  auto targetOp = cast<omp::TargetOp>(opInst);
  if (targetOp.getIfExpr()) {
    opInst.emitError("If clause not yet supported");
    return false;
  }

  if (targetOp.getDevice()) {
    opInst.emitError("Device clause not yet supported");
    return false;
  }

  if (targetOp.getThreadLimit()) {
    opInst.emitError("Thread limit clause not yet supported");
    return false;
  }

  if (targetOp.getNowait()) {
    opInst.emitError("Nowait clause not yet supported");
    return false;
  }

  return true;
}

static void
handleDeclareTargetMapVar(llvm::SmallVectorImpl<MapData> &mapData,
                          LLVM::ModuleTranslation &moduleTranslation,
                          llvm::IRBuilderBase &builder) {
  for (const auto &mapValue : mapData) {
    // In the case of declare target mapped variables, the basePointer is
    // the reference pointer generated by the convertDeclareTargetAttr
    // method. Whereas the kernelValue is the original variable, so for
    // the device we must replace all uses of this original global variable
    // (stored in kernelValue) with the reference pointer (stored in
    // basePointer for declare target mapped variables), as for device the
    // data is mapped into this reference pointer and should be loaded
    // from it, the original variable is discarded. On host both exist and
    // metadata is generated (elsewhere in the convertDeclareTargetAttr)
    // function to link the two variables in the runtime and then both the
    // reference pointer and the pointer are assigned in the kernel argument
    // structure for the host.
    if (mapValue.isDeclareTarget) {
      // The users iterator will get invalidated if we modify an element,
      // so we populate this vector of uses to alter each user on an individual
      // basis to emit it's own load (rather than one load for all).
      llvm::SmallVector<llvm::User *> userVec;
      for (llvm::User *user : mapValue.kernelValue->users())
        userVec.push_back(user);

      for (auto *user : userVec) {
        if (auto *insn = dyn_cast<llvm::Instruction>(user)) {
          auto *load = builder.CreateLoad(mapValue.basePointer->getType(),
                                          mapValue.basePointer);
          load->moveBefore(insn);
          user->replaceUsesOfWith(mapValue.kernelValue, load);
        }
      }
    }
  }
}

// This is a variation on Clang's GenerateOpenMPCapturedVars, which
// generates different operation (e.g. load/store) combinations for
// arguments to the kernel, based on map capture kinds which are then
// utilised in the combinedInfo in place of the original Map value.
static void
createAlteredByCaptureMap(llvm::SmallVectorImpl<MapData> &mapData,
                          LLVM::ModuleTranslation &moduleTranslation,
                          llvm::IRBuilderBase &builder) {
  for (MapData &mapValue : mapData) {
    // if it's declare target, skip it, it's handled seperately.
    if (!mapValue.isDeclareTarget) {
      switch (mapValue.captureKind) {
      case llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind::
          OMPTargetVarCaptureByRef: {
        // Currently handles lowerbound case, but more logic may be required.
        // Clang invokes EmitLValue, which has specialised logic for special
        // Clang types such as user defines, so it is possible we will have
        // to extend this for structures or other complex types. As the general
        // idea is that this function mimics some of the logic from Clang that
        // we require for kernel argument passing from host -> device. The load
        // in from kernel arguments for device is currently handled inside of
        // the OMPIRBuilder.
        //
        // TODO: Handle N-dimensional array range
        // TODO: Update to use dynamic bounds values, not this gathering of
        // constant values to insert in the GEP. This should be possible using
        // the bounds information, can likely use ModuleTranslation to find
        // the relevant LLVM IR value for the LB bound (or any other required
        // piece).
        auto mapOp = mlir::dyn_cast<mlir::omp::MapEntryOp>(mapValue.mapClause);
        if (!mapOp.getBounds().empty()) {
          assert(mapOp.getBounds().size() <= 1 &&
                 "Only 1-D map bounds are currently supported");
          llvm::SetVector<uint64_t> lbs;
          for (auto bound : mapOp.getBounds()) {
            auto boundOp =
                mlir::dyn_cast<mlir::omp::DataBoundsOp>(bound.getDefiningOp());

            if (auto lBound = boundOp.getLowerBound())
              if (auto constOp =
                      dyn_cast<LLVM::ConstantOp>(lBound.getDefiningOp()))
                if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                  lbs.insert(intAttr.getInt());
          }

          if (mapValue.type->isArrayTy() && !lbs.empty()) {
            // NOTE: 2D GEP required for 1D array boundary it seems
            auto arr = std::vector<llvm::Value *>{builder.getInt64(0),
                                                  builder.getInt64(lbs[0])};
            mapValue.pointer = builder.CreateInBoundsGEP(
                mapValue.type, mapValue.pointer, arr, "arrayidx");
          }
        }
      } break;
      case llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind::
          OMPTargetVarCaptureByCopy: {
        llvm::Type *type = mapValue.type;
        llvm::Value *newV = builder.CreateLoad(type, mapValue.pointer);

        if (!type->isPointerTy()) {
          auto curInsert = builder.saveIP();
          builder.restoreIP(findAllocaInsertPoint(builder, moduleTranslation));
          auto *memTempAlloc =
              builder.CreateAlloca(builder.getInt8PtrTy(), nullptr, ".casted");
          builder.restoreIP(curInsert);

          // TODO: Insert an align & scalar convert as Clang does, if
          // neccessary.
          builder.CreateStore(newV, memTempAlloc);
          newV = builder.CreateLoad(builder.getInt8PtrTy(), memTempAlloc);
        }

        mapValue.pointer = newV;
        mapValue.basePointer = newV;
      } break;
      case llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind::
          OMPTargetVarCaptureThis:
      case llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind::
          OMPTargetVarCaptureVLAType:
        mapValue.mapClause->emitOpError("Unhandled capture kind");
        break;
      case llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind::
          OMPTargetVarCaptureNone:
        mapValue.mapClause->emitOpError("MapOp missing capture kind");
        break;
      }
    }
  }
}

// Refactor TODO:
//  * Change the OMPIRBuilder.cpp getArgAccessFromCapture function into
//    a lambda function and move it out of the IRBuilder and have it have
//    an isDevice switch so it functions for host/device
//  * after above change can then simplify the OMPTargetVarCaptureKind to only
//    be an MLIR variable and remove it from the IRBuilder
//  * Move the functionality borrowed from Clang into the IRBuilder and share it
//    across Flang/Clang, meaning the stuff for assiging member_of correctly.
//  * Cut out unneeded data from MapData
//  * Rebase
//  * Change auto's with no type on the right hand to fixed types.
//  * Modify cases where we call getOperator to ->
//  * Modify cases where we call getDefiningOp + a cast after to use
//    dyn_cast_if_present
//  * Adapt the array sectioning to not access the constant value but use the
//    value at runtime similarly to the descriptors for pointers/structures
//  * Handle duplicate map clauses from being processed multiple times
static LogicalResult
convertOmpTarget(Operation &opInst, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  if (!targetOpSupported(opInst))
    return failure();

  auto targetOp = cast<omp::TargetOp>(opInst);
  LogicalResult bodyGenStatus = success();

  using InsertPointTy = llvm::OpenMPIRBuilder::InsertPointTy;
  auto bodyCB = [&](InsertPointTy allocaIP,
                    InsertPointTy codeGenIP) -> InsertPointTy {
    builder.restoreIP(codeGenIP);
    llvm::BasicBlock *exitBlock =
        convertOmpOpRegions(targetOp.getRegion(), "omp.target", builder,
                            moduleTranslation, bodyGenStatus);
    builder.SetInsertPoint(exitBlock);
    return builder.saveIP();
  };

  llvm::OpenMPIRBuilder::LocationDescription ompLoc(builder);

  // Override parent name if early outlining function
  StringRef parentName = opInst.getParentOfType<LLVM::LLVMFuncOp>().getName();
  if (auto earlyOutlineOp = llvm::dyn_cast<mlir::omp::EarlyOutliningInterface>(
          opInst.getParentOfType<LLVM::LLVMFuncOp>().getOperation())) {
    llvm::StringRef outlineParentName = earlyOutlineOp.getParentName();
    parentName = outlineParentName.empty() ? parentName : outlineParentName;
  }

  llvm::TargetRegionEntryInfo entryInfo;
  if (!getTargetEntryUniqueInfo(entryInfo, targetOp, parentName))
    return failure();

  llvm::OpenMPIRBuilder::InsertPointTy allocaIP =
      findAllocaInsertPoint(builder, moduleTranslation);
  DataLayout dl = DataLayout(opInst.getParentOfType<ModuleOp>());

  // TODO:
  // 1) Create a list filter for objects that are part of a struct vs not
  // 2) Change getKernelInfo to work on a single argument perhaps... rather
  //    than a loop, should make it easier to use

  // C++/C classes/structures and Fortran derived-types and
  // pointers/allocatables (which are effectively a structure
  // in LLVM-IR with descriptor information and the data) have
  // there individual pointer members that are targetting
  // dynamic data mapped seperately (information generated for
  // host primarily), however, they are still passed to the
  // kernel as a single input argument, so we filter them out
  // for specialised handling and group them into:
  // Struct -> ListOf<MemberOf>'s to be mapped.

  // NEED TO DO THE ABOVE / AND IF VARPTR+VARPTRPTR SELECT THE VARPTRPTR OR SO
  llvm::SmallVector<MapData> mapData;
  llvm::SmallVector<Value> mapOperands = targetOp.getMapOperands();
  collectMapDataFromMapOperands(mapData, mapOperands, moduleTranslation, dl);

  // We wish to modify some of the methods in which kernel arguments are
  // passed based on their capture type by the target region, this can
  // involve generating new loads and stores, which changes the
  // MLIR value to LLVM value mapping, however, we only wish to do this
  // locally for the current function/target and also avoid altering
  // ModuleTranslation, so we remap the base pointer or pointer stored
  // in the map entries corresponding MapData, which is later accessed
  // by processMapOp and createTarget to help generate the kernel and
  // kernel arg structure. It primarily becomes relevant in cases like
  // bycopy, or byref range'd arrays. In the default case, we simply
  // pass thee pointer byref as both basePointer and pointer.
  if (!moduleTranslation.getOpenMPBuilder()->Config.isTargetDevice())
    createAlteredByCaptureMap(mapData, moduleTranslation, builder);

  llvm::OpenMPIRBuilder::MapInfosTy combinedInfos;
  auto genMapInfoCB = [&](llvm::OpenMPIRBuilder::InsertPointTy codeGenIP)
      -> llvm::OpenMPIRBuilder::MapInfosTy & {
    builder.restoreIP(codeGenIP);
    processMapOp(builder, moduleTranslation, dl, combinedInfos, mapData);
    return combinedInfos;
  };

  auto mapDataForTargetKernelArgGen = [](llvm::SmallVector<MapData> &mapData) {
    llvm::SmallVector<llvm::Value *, 4> values;
    llvm::SmallVector<llvm::Type *, 4> types;
    llvm::SmallVector<llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind,
                      4>
        captureKinds;

    for (auto &data : mapData) {
      // declare target arguments are not passed to kernels as arguments
      if (!data.isDeclareTarget) {
        values.push_back(data.kernelValue);
        types.push_back(data.type);
        captureKinds.push_back(data.captureKind);
      }
    }

    return std::tuple<
        llvm::SmallVector<llvm::Value *, 4>, llvm::SmallVector<llvm::Type *, 4>,
        llvm::SmallVector<
            llvm::OffloadEntriesInfoManager::OMPTargetVarCaptureKind, 4>>(
        values, types, captureKinds);
  };

  auto kernelArgMapData = mapDataForTargetKernelArgGen(mapData);

  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createTarget(
      ompLoc, allocaIP, builder.saveIP(), entryInfo, /*NumTeams*/ -1,
      /*NumThreads*/ 0, std::get<0>(kernelArgMapData),
      std::get<1>(kernelArgMapData), std::get<2>(kernelArgMapData),
      genMapInfoCB, bodyCB));

  if (moduleTranslation.getOpenMPBuilder()->Config.isTargetDevice())
    handleDeclareTargetMapVar(mapData, moduleTranslation, builder);

  return bodyGenStatus;
}

LogicalResult
convertDeclareTargetAttr(Operation *op, mlir::omp::DeclareTargetAttr attribute,
                         LLVM::ModuleTranslation &moduleTranslation) {
  // Amend omp.declare_target by deleting the IR of the outlined functions
  // created for target regions. They cannot be filtered out from MLIR earlier
  // because the omp.target operation inside must be translated to LLVM, but
  // the wrapper functions themselves must not remain at the end of the
  // process. We know that functions where omp.declare_target does not match
  // omp.is_target_device at this stage can only be wrapper functions because
  // those that aren't are removed earlier as an MLIR transformation pass.
  if (FunctionOpInterface funcOp = dyn_cast<FunctionOpInterface>(op)) {
    if (auto offloadMod = dyn_cast<omp::OffloadModuleInterface>(
            op->getParentOfType<ModuleOp>().getOperation())) {
      if (!offloadMod.getIsTargetDevice())
        return success();

      omp::DeclareTargetDeviceType declareType =
          attribute.getDeviceType().getValue();

      if (declareType == omp::DeclareTargetDeviceType::host) {
        llvm::Function *llvmFunc =
            moduleTranslation.lookupFunction(funcOp.getName());
        llvmFunc->dropAllReferences();
        llvmFunc->eraseFromParent();
      }
    }
    return success();
  }

  if (LLVM::GlobalOp gOp = dyn_cast<LLVM::GlobalOp>(op)) {
    llvm::Module *llvmModule = moduleTranslation.getLLVMModule();
    if (auto *gVal = llvmModule->getNamedValue(gOp.getSymName())) {
      llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();
      bool isDeclaration = gOp.isDeclaration();
      bool isExternallyVisible =
          gOp.getVisibility() != mlir::SymbolTable::Visibility::Private;
      auto loc = op->getLoc()->findInstanceOf<FileLineColLoc>();
      llvm::StringRef mangledName = gOp.getSymName();
      auto captureClause =
          convertToCaptureClauseKind(attribute.getCaptureClause().getValue());
      auto deviceClause =
          convertToDeviceClauseKind(attribute.getDeviceType().getValue());
      // unused for MLIR at the moment, required in Clang for book
      // keeping
      std::vector<llvm::GlobalVariable *> generatedRefs;

      std::vector<llvm::Triple> targetTriple;
      auto targetTripleAttr =
          op->getParentOfType<mlir::ModuleOp>().getOperation()->getAttr(
              LLVM::LLVMDialect::getTargetTripleAttrName());
      if (targetTripleAttr)
        targetTriple.emplace_back(
            targetTripleAttr.dyn_cast_or_null<mlir::StringAttr>().data());

      auto fileInfoCallBack = [&loc]() {
        llvm::StringRef filename = loc.getFilename();
        // handle split file from mlir-translate that for some reason
        // manipulates the string to contain other components than
        // filename
        if (filename.contains("within split at")) {
          filename.consume_front("within split at ");
          filename = filename.rsplit(":").first;
        }

        return std::pair<std::string, uint64_t>(filename, loc.getLine());
      };

      ompBuilder->registerTargetGlobalVariable(
          captureClause, deviceClause, isDeclaration, isExternallyVisible,
          ompBuilder->getTargetEntryUniqueInfo(fileInfoCallBack), mangledName,
          generatedRefs, false, targetTriple, nullptr, nullptr, gVal->getType(),
          gVal);

      if (ompBuilder->Config.isTargetDevice() &&
          (attribute.getCaptureClause().getValue() !=
               mlir::omp::DeclareTargetCaptureClause::to ||
           ompBuilder->Config.hasRequiresUnifiedSharedMemory())) {
        ompBuilder->getAddrOfDeclareTargetVar(
            captureClause, deviceClause, isDeclaration, isExternallyVisible,
            ompBuilder->getTargetEntryUniqueInfo(fileInfoCallBack), mangledName,
            generatedRefs, false, targetTriple, gVal->getType(), nullptr,
            nullptr);
        // A global has already been generated by this stage, unlike Clang, so
        // this needs to be specially removed here for device when we're
        // anything but a To clause specified variable with no unified shared
        // memory.
        if (llvm::GlobalValue *llvmVal =
                llvmModule->getNamedValue(mangledName)) {
          llvmVal->removeFromParent();
          llvmVal->dropAllReferences();
        }
      }
    }
  }

  return success();
}

namespace {

/// Implementation of the dialect interface that converts operations belonging
/// to the OpenMP dialect to LLVM IR.
class OpenMPDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final;

  LogicalResult
  amendOperation(Operation *op, NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final;
};

} // namespace

/// Given an OpenMP MLIR attribute, create the corresponding LLVM-IR, runtime
/// calls, or operation amendments
LogicalResult OpenMPDialectLLVMIRTranslationInterface::amendOperation(
    Operation *op, NamedAttribute attribute,
    LLVM::ModuleTranslation &moduleTranslation) const {
  return llvm::TypeSwitch<Attribute, LogicalResult>(attribute.getValue())
      .Case([&](mlir::omp::FlagsAttr rtlAttr) {
        return convertFlagsAttr(op, rtlAttr, moduleTranslation);
      })
      .Case([&](mlir::omp::DeclareTargetAttr dtAttr) {
        return convertDeclareTargetAttr(op, dtAttr, moduleTranslation);
      })
      .Case([&](mlir::omp::VersionAttr versionAttr) {
        llvm::OpenMPIRBuilder *ompBuilder =
            moduleTranslation.getOpenMPBuilder();
        ompBuilder->M.addModuleFlag(llvm::Module::Max, "openmp",
                                    versionAttr.getVersion());
        return success();
      })
      .Case([&](mlir::omp::DeclareTargetAttr declareTargetAttr) {
        return convertDeclareTargetAttr(op, declareTargetAttr,
                                        moduleTranslation);
      })
      .Default([&](Attribute attr) {
        // fall through for omp attributes that do not require lowering and/or
        // have no concrete definition and thus no type to define a case on
        return success();
      });

  return failure();
}

/// Given an OpenMP MLIR operation, create the corresponding LLVM IR
/// (including OpenMP runtime calls).
LogicalResult OpenMPDialectLLVMIRTranslationInterface::convertOperation(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {
  llvm::OpenMPIRBuilder *ompBuilder = moduleTranslation.getOpenMPBuilder();

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](omp::BarrierOp) {
        ompBuilder->createBarrier(builder.saveIP(), llvm::omp::OMPD_barrier);
        return success();
      })
      .Case([&](omp::TaskwaitOp) {
        ompBuilder->createTaskwait(builder.saveIP());
        return success();
      })
      .Case([&](omp::TaskyieldOp) {
        ompBuilder->createTaskyield(builder.saveIP());
        return success();
      })
      .Case([&](omp::FlushOp) {
        // No support in Openmp runtime function (__kmpc_flush) to accept
        // the argument list.
        // OpenMP standard states the following:
        //  "An implementation may implement a flush with a list by ignoring
        //   the list, and treating it the same as a flush without a list."
        //
        // The argument list is discarded so that, flush with a list is treated
        // same as a flush without a list.
        ompBuilder->createFlush(builder.saveIP());
        return success();
      })
      .Case([&](omp::ParallelOp op) {
        return convertOmpParallel(op, builder, moduleTranslation);
      })
      .Case([&](omp::ReductionOp reductionOp) {
        return convertOmpReductionOp(reductionOp, builder, moduleTranslation);
      })
      .Case([&](omp::MasterOp) {
        return convertOmpMaster(*op, builder, moduleTranslation);
      })
      .Case([&](omp::CriticalOp) {
        return convertOmpCritical(*op, builder, moduleTranslation);
      })
      .Case([&](omp::OrderedRegionOp) {
        return convertOmpOrderedRegion(*op, builder, moduleTranslation);
      })
      .Case([&](omp::OrderedOp) {
        return convertOmpOrdered(*op, builder, moduleTranslation);
      })
      .Case([&](omp::WsLoopOp) {
        return convertOmpWsLoop(*op, builder, moduleTranslation);
      })
      .Case([&](omp::SimdLoopOp) {
        return convertOmpSimdLoop(*op, builder, moduleTranslation);
      })
      .Case([&](omp::AtomicReadOp) {
        return convertOmpAtomicRead(*op, builder, moduleTranslation);
      })
      .Case([&](omp::AtomicWriteOp) {
        return convertOmpAtomicWrite(*op, builder, moduleTranslation);
      })
      .Case([&](omp::AtomicUpdateOp op) {
        return convertOmpAtomicUpdate(op, builder, moduleTranslation);
      })
      .Case([&](omp::AtomicCaptureOp op) {
        return convertOmpAtomicCapture(op, builder, moduleTranslation);
      })
      .Case([&](omp::SectionsOp) {
        return convertOmpSections(*op, builder, moduleTranslation);
      })
      .Case([&](omp::SingleOp op) {
        return convertOmpSingle(op, builder, moduleTranslation);
      })
      .Case([&](omp::TaskOp op) {
        return convertOmpTaskOp(op, builder, moduleTranslation);
      })
      .Case([&](omp::TaskGroupOp op) {
        return convertOmpTaskgroupOp(op, builder, moduleTranslation);
      })
      .Case<omp::YieldOp, omp::TerminatorOp, omp::ReductionDeclareOp,
            omp::CriticalDeclareOp>([](auto op) {
        // `yield` and `terminator` can be just omitted. The block structure
        // was created in the region that handles their parent operation.
        // `reduction.declare` will be used by reductions and is not
        // converted directly, skip it.
        // `critical.declare` is only used to declare names of critical
        // sections which will be used by `critical` ops and hence can be
        // ignored for lowering. The OpenMP IRBuilder will create unique
        // name for critical section names.
        return success();
      })
      .Case([&](omp::ThreadprivateOp) {
        return convertOmpThreadprivate(*op, builder, moduleTranslation);
      })
      .Case<omp::DataOp, omp::EnterDataOp, omp::ExitDataOp>([&](auto op) {
        return convertOmpTargetData(op, builder, moduleTranslation);
      })
      .Case([&](omp::TargetOp) {
        return convertOmpTarget(*op, builder, moduleTranslation);
      })
      .Case([&](omp::MapEntryOp) {
        // handled in respective container operators e.g. TargetOp.
        return success();
      })
      .Case([&](omp::DataBoundsOp) {
        // handled in respective container operator e.g. MapEntryOp.
        return success();
      })
      .Default([&](Operation *inst) {
        return inst->emitError("unsupported OpenMP operation: ")
               << inst->getName();
      });
}

void mlir::registerOpenMPDialectTranslation(DialectRegistry &registry) {
  registry.insert<omp::OpenMPDialect>();
  registry.addExtension(+[](MLIRContext *ctx, omp::OpenMPDialect *dialect) {
    dialect->addInterfaces<OpenMPDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerOpenMPDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerOpenMPDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
