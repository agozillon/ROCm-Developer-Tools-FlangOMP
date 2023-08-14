#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/IR/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"

namespace fir {
#define GEN_PASS_DEF_OMPEARLYOUTLININGPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class OMPEarlyOutliningPass
    : public fir::impl::OMPEarlyOutliningPassBase<OMPEarlyOutliningPass> {

  std::string getOutlinedFnName(llvm::StringRef parentName, unsigned count) {
    return std::string(parentName) + "_omp_outline_" + std::to_string(count);
  }

  mlir::func::FuncOp outlineTargetOp(mlir::OpBuilder &builder,
                                     mlir::omp::TargetOp &targetOp,
                                     mlir::func::FuncOp &parentFunc,
                                     unsigned count) {
    // NOTE: once implicit captures are handled appropriately in the initial
    // PFT lowering if it is possible, we can remove the usage of
    // getUsedValuesDefinedAbove and instead just iterate over the target op's
    // operands (or just the map arguments) and perhaps refactor this function
    // a little.
    // Collect inputs
    llvm::SetVector<mlir::Value> inputs;
    for (auto operand : targetOp.getOperation()->getOperands())
      if (!mlir::isa<mlir::omp::MapEntryOp>(operand.getDefiningOp()))
        inputs.insert(operand);
    
    mlir::Region &targetRegion = targetOp.getRegion();
    mlir::getUsedValuesDefinedAbove(targetRegion, inputs);

    llvm::SetVector<mlir::Value> declareTargetAddrOfOld;
    llvm::SetVector<mlir::Value> declareTargetAddrOfNew;
    llvm::SetVector<mlir::Operation *> declareTargetclone;
    for (auto input : inputs) {
      if (input.getDefiningOp()) {
        if (fir::AddrOfOp addressOfOp =
                mlir::dyn_cast<fir::AddrOfOp>(input.getDefiningOp())) {
          if (fir::GlobalOp gOp = mlir::dyn_cast<fir::GlobalOp>(
                  addressOfOp->getParentOfType<mlir::ModuleOp>().lookupSymbol(
                      addressOfOp.getSymbol()))) {
            if (auto declareTargetGlobal =
                    llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
                        gOp.getOperation())) {
              if (declareTargetGlobal.isDeclareTarget()) {
                declareTargetAddrOfOld.insert(input);
                inputs.remove(input);
                declareTargetclone.insert(addressOfOp);
              }
            }
          }
        }
      }
    }

      // Create new function and initialize
      mlir::FunctionType funcType = builder.getFunctionType(
          mlir::TypeRange(inputs.getArrayRef()), mlir::TypeRange());
      std::string parentName(parentFunc.getName());
      std::string funcName = getOutlinedFnName(parentName, count);
      auto loc = targetOp.getLoc();
      mlir::func::FuncOp newFunc =
          mlir::func::FuncOp::create(loc, funcName, funcType);
      mlir::Block *entryBlock = newFunc.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);
      mlir::ValueRange newInputs = entryBlock->getArguments();

      // Set the declare target information, the outlined function
      // is always a host function.
      if (auto parentDTOp = llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
              parentFunc.getOperation()))
        if (auto newDTOp = llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
                newFunc.getOperation()))
          newDTOp.setDeclareTarget(mlir::omp::DeclareTargetDeviceType::host,
                                   parentDTOp.getDeclareTargetCaptureClause());

      // Set the early outlining interface parent name
      if (auto earlyOutlineOp =
              llvm::dyn_cast<mlir::omp::EarlyOutliningInterface>(
                  newFunc.getOperation()))
        earlyOutlineOp.setParentName(parentName);

      
      // The assumption is that we're using addressOfOp to indicate the correct
      // global that the map with a declare target argument corresponds to and
      // this addressOfOp has only a single return value, which is currently
      // the case
      for (auto &clone : declareTargetclone)
        declareTargetAddrOfNew.insert(builder.clone(*clone)->getResult(0));
      
      llvm::SetVector<mlir::Value> mapOld;
      llvm::SetVector<mlir::Value> mapNew;

      for (auto oper : targetOp.getOperation()->getOperands()) {
        if (auto mapEntry =
                mlir::dyn_cast<mlir::omp::MapEntryOp>(oper.getDefiningOp())) {
          mlir::IRMapping mapEntryMap;
          // NOTE: do we need more than one bounds per map entry here? There is
          // only a single var_ptr it seems
          for (auto bound : mapEntry.getBounds()) {
            if (auto mapEntryBound = mlir::dyn_cast<mlir::omp::DataBoundsOp>(
                    bound.getDefiningOp())) {
              mlir::IRMapping boundMap;
              if (mapEntryBound.getUpperBound() &&
                  mapEntryBound.getUpperBound().getDefiningOp())
                boundMap.map(
                    mapEntryBound.getUpperBound(),
                    builder.clone(*mapEntryBound.getUpperBound().getDefiningOp())
                        ->getResult(0));
              if (mapEntryBound.getLowerBound() &&
                  mapEntryBound.getLowerBound().getDefiningOp())
                boundMap.map(
                    mapEntryBound.getLowerBound(),
                    builder.clone(*mapEntryBound.getLowerBound().getDefiningOp())
                        ->getResult(0));
              if (mapEntryBound.getStride() &&
                  mapEntryBound.getStride().getDefiningOp())
                boundMap.map(
                    mapEntryBound.getStride(),
                    builder.clone(*mapEntryBound.getStride().getDefiningOp())
                        ->getResult(0));
              if (mapEntryBound.getStartIdx() &&
                  mapEntryBound.getStartIdx().getDefiningOp())
                boundMap.map(
                    mapEntryBound.getStartIdx(),
                    builder.clone(*mapEntryBound.getStartIdx().getDefiningOp())
                        ->getResult(0));
              mapEntryMap.map(
                  bound, builder.clone(*mapEntryBound, boundMap)->getResult(0));
            }
          }

            for (auto inArg : llvm::zip(inputs, newInputs))
              if (mapEntry.getVarPtr() == std::get<0>(inArg))
              mapEntryMap.map(mapEntry.getVarPtr(), std::get<1>(inArg));
     

            mapOld.insert(mapEntry);
            mapNew.insert(builder.clone(*mapEntry.getOperation(), mapEntryMap)->getResult(0));       
          }
      }

      // Create input map from inputs to function parameters.
      mlir::IRMapping valueMap;
      for (auto inArg : llvm::zip(inputs, newInputs))
        valueMap.map(std::get<0>(inArg), std::get<1>(inArg));

      for (auto inArg :
           llvm::zip(declareTargetAddrOfOld, declareTargetAddrOfNew))
        valueMap.map(std::get<0>(inArg), std::get<1>(inArg));

      for (auto inArg :
           llvm::zip(mapOld, mapNew))
        valueMap.map(std::get<0>(inArg), std::get<1>(inArg));
      
      // Clone the target op into the new function
      builder.clone(*(targetOp.getOperation()), valueMap);

      // Create return op
      builder.create<mlir::func::ReturnOp>(loc);

      return newFunc;
    }

  // Returns true if a target region was found int the function.
  bool outlineTargetOps(mlir::OpBuilder &builder,
                        mlir::func::FuncOp &functionOp,
                        mlir::ModuleOp &moduleOp,
                        llvm::SmallVectorImpl<mlir::func::FuncOp> &newFuncs) {
    unsigned count = 0;
    for (auto TargetOp : functionOp.getOps<mlir::omp::TargetOp>()) {
      mlir::func::FuncOp outlinedFunc =
          outlineTargetOp(builder, TargetOp, functionOp, count);
      newFuncs.push_back(outlinedFunc);
      count++;
    }
    return count > 0;
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::MLIRContext *context = &getContext();
    mlir::OpBuilder builder(context);
    llvm::SmallVector<mlir::func::FuncOp> newFuncs;

    for (auto functionOp :
         llvm::make_early_inc_range(moduleOp.getOps<mlir::func::FuncOp>())) {
      bool outlined = outlineTargetOps(builder, functionOp, moduleOp, newFuncs);
      
      if (outlined)
        functionOp.erase();
    }

    for (auto newFunc : newFuncs)
      moduleOp.push_back(newFunc);

    moduleOp.dump();

    llvm::errs() << "escaped? \n";
  }
};

} // namespace

namespace fir {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOMPEarlyOutliningPass() {
  return std::make_unique<OMPEarlyOutliningPass>();
}
} // namespace fir
