#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace {
enum ForType { Generic, Gemm, FA };

class InferStride {
  scf::ForOp forOp;
  SmallVector<tt::LoadOp> gLoadOps;
  SmallVector<tt::DotOp> dotOps;

public:
  InferStride(scf::ForOp forOp) : forOp(forOp) {}
  ForType inferForType();
  void inferGemmStride();

private:
  tt::LoadOp getLoadOp(Operation *op);
  Value getInputArg(Value v);
};

ForType InferStride::inferForType() {
  OpBuilder builder(forOp);
  MLIRContext *ctx = forOp.getContext();
  Location loc = forOp.getLoc();

  forOp->walk([&](Operation *op) {
    if (auto maybeGemmDot = dyn_cast<tt::DotOp>(op))
      if (maybeGemmDot.getType().getRank() == 2)
        dotOps.push_back(maybeGemmDot);
  });

  if (dotOps.size() == 1)
    return ForType::Gemm;
  else
    return ForType::Generic;
}

tt::LoadOp InferStride::getLoadOp(Operation *op) {
  if (auto loadOp = dyn_cast<tt::LoadOp>(op))
    return loadOp;
  while (auto src = op->getOperand(0)) {
    if (auto maybeLoadOp = src.getDefiningOp()) {
      if (auto loadOp = dyn_cast<tt::LoadOp>(maybeLoadOp))
        return loadOp;
      op = maybeLoadOp;
    } else {
      break;
    }
  }
  return nullptr;
}

Value InferStride::getInputArg(Value v) {
  while (auto srcOp = v.getDefiningOp()) {
    if (auto addPtrOp = dyn_cast<tt::AddPtrOp>(srcOp))
      v = addPtrOp.getPtr();
    else if (auto splatOp = dyn_cast<tt::SplatOp>(srcOp))
      v = splatOp.getSrc();
    else if (auto bcOp = dyn_cast<tt::BroadcastOp>(srcOp))
      v = bcOp.getSrc();
    else
      return nullptr;
  }
  if (auto funcArg = dyn_cast<BlockArgument>(v)) {
    return v;
  } else
    return nullptr;
}

void InferStride::inferGemmStride() {
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  auto matA = dotOps[0].getA();
  auto matB = dotOps[0].getB();
  auto loadA = getLoadOp(matA.getDefiningOp());
  auto loadB = getLoadOp(matB.getDefiningOp());
  if (loadA == nullptr || loadB == nullptr)
    return;
  SmallVector<arith::ConstantIntOp> incrementA;
  auto ptrA = loadA.getPtr();
  Value forArg = forOp.getTiedLoopInit(dyn_cast<BlockArgument>(ptrA))->get();
  Value inputArg = getInputArg(forArg);
  if (inputArg == nullptr)
    return;
  builder.setInsertionPointAfter(forOp->getPrevNode());
  for (auto user : ptrA.getUsers()) {
    if (auto addPtrOp = dyn_cast<tt::AddPtrOp>(user)) {
      // todo check if parent is forOp!!!!!!!!!!

      auto v = addPtrOp.getOffset();
      if (auto constIncrTensorOp =
              dyn_cast<arith::ConstantOp>(v.getDefiningOp())) {
        // todo check isSplat !!!!!!!!!!!!!!

        auto incrValue =
            mlir::cast<DenseElementsAttr>(constIncrTensorOp.getValue())
                .getSplatValue<uint32_t>();
        incrementA.push_back(
            builder.create<arith::ConstantIntOp>(loc, incrValue, 32));
      }
    }
  }

  // incrementA[0].dump();
  auto ub = forOp.getUpperBound();
  auto inferredStrideA = builder.create<arith::MulIOp>(loc, ub, incrementA[0]);
  auto strideAOp =
      builder.create<tt::amdgpu::SetStrideOp>(loc,inputArg.getType(), inputArg, inferredStrideA);
}

class TritonAMDGPUInferStridePass
    : public TritonAMDGPUInferStrideBase<TritonAMDGPUInferStridePass> {
public:
  TritonAMDGPUInferStridePass() = default;
  void runOnOperation() override {
    ModuleOp m = getOperation();
    for (auto funcOp : m.getOps<tt::FuncOp>()) {
      funcOp.walk([&](scf::ForOp forOp) {
        InferStride inferringOp(forOp);
        switch (inferringOp.inferForType()) {
        case ForType::Gemm:
          inferringOp.inferGemmStride();
          break;
        default:
          break;
        }
      });
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUInferStridePass() {
  return std::make_unique<TritonAMDGPUInferStridePass>();
}
