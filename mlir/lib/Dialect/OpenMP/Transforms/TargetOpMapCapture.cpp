//===- TargetOpMapCapture.cpp - Capture Implicits in TargetOp Map Clause --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/IR/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include <cstdint>

namespace mlir {
namespace {

#define GEN_PASS_DEF_TARGETOPMAPCAPTUREPASS
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"

struct TargetOpMapCapturePass
    : public impl::TargetOpMapCapturePassBase<
          TargetOpMapCapturePass> {

  using TargetOpMapCapturePassBase<
      TargetOpMapCapturePass>::TargetOpMapCapturePassBase;

  void runOnOperation() override {
    auto module = getOperation();

    module.walk([&](mlir::omp::TargetOp tarOp) {
      llvm::SetVector<Value> operandSet;
      getUsedValuesDefinedAbove(tarOp.getRegion(), operandSet);

      llvm::SmallVector<mlir::Value> usedButNotCaptured;
      for (auto v : operandSet) {
        bool insertable = true;
        for (auto mapOp : tarOp.getMapOperands())
          if (auto mapEntry =
                  mlir::dyn_cast<mlir::omp::MapEntryOp>(mapOp.getDefiningOp()))
            if (v == mapEntry.getVarPtr())
              insertable = false;

        if (insertable)
          usedButNotCaptured.push_back(v);
      }

      // NOTE: Ponter-case, unused currently as it is a WIP.
      // llvm::omp::OpenMPOffloadMappingFlags captureByThis =
      //     llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO |
      //     llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;

      llvm::omp::OpenMPOffloadMappingFlags literalCapture =
          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_LITERAL;

      llvm::omp::OpenMPOffloadMappingFlags mapTypeBits;

      mlir::OpBuilder builder(tarOp);
      llvm::SmallVector<mlir::Value> sv;
      // Mimicing Map Type Generation code from CGOpenMPRuntime.cpp in Clang's
      // generateDefaultMapInfo, this is an initial
      for (auto &var : usedButNotCaptured) {
        // TODO/NOTE: Currently doesn't handle VarPtrPtr (Struct Elements), can
        // look at OpenMP Lower's createMapEntryOp for inspiration
        mlir::omp::MapEntryOp op = builder.create<mlir::omp::MapEntryOp>(
            tarOp->getLoc(), var.getType(), var);

        if (var.getDefiningOp())
          op.setNameAttr(builder.getStringAttr(
              var.getDefiningOp()->getName().getStringRef()));
        op.setImplicit(true);

        // TODO: Case for pointers/non-literals
        mapTypeBits = literalCapture;

        // All captures are target_param
        mapTypeBits |=
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TARGET_PARAM;

        // TODO: not all captures are implicit, but it is the default
        // handling this needs to be extended to handle the non-default
        mapTypeBits |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;

        op.setMapType(
            static_cast<
                std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                mapTypeBits));
        op.setMapCaptureType(mlir::omp::VariableCaptureKind::ByCopy);

        // Default is an empty bounds for implicit capture, for the moment.
        // We can likely at most generate the max lb and ub for arrays here
        // from array ref information and give a default stride, but that
        // may be the full extent of what's possible at the moment. Ideally
        // we handle implicit captures at the PFT lowering level as well, but
        // yet to work out how to do this.
        llvm::SmallVector<mlir::Value> bounds;
        op->setAttr(mlir::omp::MapEntryOp::getOperandSegmentSizeAttr(),
                    builder.getDenseI32ArrayAttr(
                        {1, 0, static_cast<int32_t>(bounds.size())}));

        sv.push_back(op);
      }

      tarOp.getMapOperandsMutable().append(sv);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::omp::createTargetOpMapCapturePass() {
  return std::make_unique<TargetOpMapCapturePass>();
}

} // namespace mlir

