#ifndef TRITONGPU_ATTRDEFS
#define TRITONGPU_ATTRDEFS

include "tritoncc/dialect/TritonGPU/Dialect.td"

include "mlir/IR/AttrTypeBase.td"

def TritonGPU_AttrTrait : AttrInterface<"TritonGPU_AttrTrait"> {
  let cppNamespace = "::mlir::_tritoncc";

  let methods = [
    InterfaceMethod<"Return total element size per thread.",
        "unsigned",
        "getTotalElemsPerThread",
        (ins "llvm::ArrayRef<int64_t>":$tensorShape,
            "Type":$eltTy)>,

    InterfaceMethod<"Return element size per thread in each dimension.",
        "llvm::SmallVector<unsigned>",
        "getElemsPerThread",
        (ins "llvm::ArrayRef<int64_t>":$tensorShape,
            "Type":$eltTy)>,
  ];
}

class TritonGPU_Attr<string name, string attrMnemonic, list<Trait> traits = [],
    Dialect dialect = TritonGPU_Dialect,
    string baseCppClass = "::mlir::Attribute">
  : AttrDef<dialect, name, !listconcat([TritonGPU_AttrTrait], traits), baseCppClass> {
  let attrName = "triton.gpu." # attrMnemonic;

  code extraBaseClassDeclaration = [{
    unsigned getTotalElemsPerThread(llvm::ArrayRef<int64_t> shape, mlir::Type eltTy) const;
    llvm::SmallVector<unsigned> getElemsPerThread(llvm::ArrayRef<int64_t> shape, mlir::Type eltTy) const;
  }];
}

def DistributedEncodingTrait : AttrInterface<"DistributedEncodingTrait"> {
  let cppNamespace = "::mlir::_tritoncc";

  let methods = [
    // Interface for the meta information about the multiple thread hierarchy.
    InterfaceMethod<"Get the shape of the warps per CTA.",
        "llvm::SmallVector<unsigned>",
        "getWarpsPerCTA">,

    InterfaceMethod<"Each CTA processes 1/CTASplitNum of the tensor.",
        "llvm::SmallVector<unsigned>",
        "getCTASplitNum">,

    InterfaceMethod<"Get the order of the CTAs per CGA. The fastest-changing axis first",
        "llvm::SmallVector<unsigned>",
        "getCTAOrder">,

    InterfaceMethod<"Get the shape of the CTAs per CGA.",
        "llvm::SmallVector<unsigned>",
        "getCTAsPerCGA">,

    InterfaceMethod<"Get the shape of the values per thread.",
        "llvm::SmallVector<unsigned>",
        "getSizePerThread">,

    InterfaceMethod<"Get the shape of the threads per warp",
        "llvm::SmallVector<unsigned>",
        "getThreadsPerWarp">,

    InterfaceMethod<"Gets the shape of the encoding's tile, e.g. sizePerThread * threadsPerWarp * warpsPerCTA",
        "llvm::SmallVector<unsigned>",
        "getShapePerCTATile",
        (ins "llvm::ArrayRef<int64_t>":$tensorShape)>,
  ];
}

class DistributedEncoding<string name, string attrMnemonic, list<Trait> traits = [], Dialect dialect = TritonGPU_Dialect> : TritonGPU_Attr<name, attrMnemonic, !listconcat([DistributedEncodingTrait], traits), dialect> {
  code extraDistributedDeclaration = extraBaseClassDeclaration # [{
    llvm::SmallVector<unsigned> getWarpsPerCTA() const;
    llvm::SmallVector<unsigned> getSizePerThread() const;
    llvm::SmallVector<unsigned> getThreadsPerWarp() const;
    llvm::SmallVector<unsigned> getCTASplitNum() const;
    llvm::SmallVector<unsigned> getCTAOrder() const;
    llvm::SmallVector<unsigned> getCTAsPerCGA() const;
    llvm::SmallVector<unsigned> getShapePerCTATile(llvm::ArrayRef<int64_t> tensorShape = llvm::ArrayRef<int64_t>()) const;
  }];
}

def CTALayoutAttr : TritonGPU_Attr<"CTALayout", "cta_layout"> {
  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$CTAsPerCGA,
    ArrayRefParameter<"unsigned">:$CTASplitNum,
    ArrayRefParameter<"unsigned">:$CTAOrder
  );
  let extraClassDeclaration = [{
    unsigned getTotalElemsPerThread(llvm::ArrayRef<int64_t> shape, mlir::Type eltTy) const {
      assert(false && "getTotalElemsPerThread");
    }
    llvm::SmallVector<unsigned> getElemsPerThread(llvm::ArrayRef<int64_t> shape, mlir::Type eltTy) const {
      assert(false && "getElemsPerThread");
    }
  }];
}

// TODO: MMAv1 and MMAv2 should be two instances of the same class
def MmaEncodingTrait : AttrInterface<"MmaEncodingTrait"> {
  let cppNamespace = "::mlir::_tritoncc";
  let methods = [
  ];
}

def NvidiaMmaEncodingAttr : DistributedEncoding<"NvidiaMmaEncoding", "nvidia_mma_encoding", [MmaEncodingTrait]> {
  let parameters = (
    ins
    "unsigned":$versionMajor,
    "unsigned":$versionMinor,
    ArrayRefParameter<"unsigned">:$warpsPerCTA__,
    "CTALayoutAttr":$CTALayout,
    ArrayRefParameter<"unsigned">:$instrShape
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{
    bool isVolta() const;
    bool isAmpere() const;
    bool isHopper() const;
  }];
}

def DotOperandEncodingAttr : DistributedEncoding<"DotOperandEncoding", "dot_operand_encoding"> {
  let parameters = (
    ins
    "unsigned":$opIdx,
    "Attribute":$parent,
    "unsigned":$kWidth
  );
  let extraClassDeclaration = extraDistributedDeclaration;
}

def SliceEncodingAttr : DistributedEncoding<"SliceEncoding", "slice_encoding"> {
  let mnemonic = "slice";

  let parameters = (
    ins
    "unsigned":$dim,
    // TODO: constraint here to only take distributed encodings
    "Attribute":$parent
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{
    template<class T>
    llvm::SmallVector<T> paddedShape(llvm::ArrayRef<T> shape) const;
  }];

  let hasCustomAssemblyFormat = 1;
}

def BlockedEncodingAttr : DistributedEncoding<"BlockedEncoding", "blocked_encoding"> {
  let mnemonic = "blocked";

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$sizePerThread__,
    ArrayRefParameter<"unsigned">:$threadsPerWarp__,
    ArrayRefParameter<"unsigned">:$warpsPerCTA__,
    ArrayRefParameter<"unsigned">:$order, // the fastest-changing axis first

    // CTALayout is optional in the textual IR. If omitted, we infer it to be a
    // single CTA (so CTAsPerCGA = [1, ..., 1], CTASplitNum = [1, ..., 1[,
    // CTAOrder=[n, n - 1, ..., 0]).
    "CTALayoutAttr":$CTALayout
  );

  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
        "ArrayRef<unsigned>":$sizePerThread,
        "ArrayRef<unsigned>":$order,
        "unsigned":$numWarps,
        "unsigned":$numThreadsPerWarp,
        "CTALayoutAttr":$CTALayout), [{
      unsigned rank = sizePerThread.size();
      llvm::SmallVector<unsigned, 4> threadsPerWarp(rank);
      llvm::SmallVector<unsigned, 4> warpsPerCTA(rank);
      llvm::SmallVector<int64_t> shapePerCTA = tritoncc::getShapePerCTA(CTALayout.getCTASplitNum(), shape);

      unsigned remainingLanes = numThreadsPerWarp;
      unsigned remainingThreads = numWarps * numThreadsPerWarp;
      unsigned remainingWarps = numWarps;
      unsigned prevLanes = 1;
      unsigned prevWarps = 1;

      // starting from the contiguous dimension
      for (unsigned d = 0; d < rank - 1; ++d) {
        unsigned i = order[d];
        unsigned threadsPerCTA = std::clamp<unsigned>(remainingThreads, 1, shapePerCTA[i] / sizePerThread[i]);
        threadsPerWarp[i] = std::clamp<unsigned>(threadsPerCTA, 1, remainingLanes);
        warpsPerCTA[i] = std::clamp<unsigned>(threadsPerCTA / threadsPerWarp[i], 1, remainingWarps);
        remainingWarps /= warpsPerCTA[i];
        remainingLanes /= threadsPerWarp[i];
        remainingThreads /= threadsPerCTA;
        prevLanes *= threadsPerWarp[i];
        prevWarps *= warpsPerCTA[i];
      }

      // Expand the last dimension to fill the remaining lanes and warps
      threadsPerWarp[order[rank - 1]] = numThreadsPerWarp / prevLanes;
      warpsPerCTA[order[rank - 1]] = numWarps / prevWarps;

      return $_get(context, sizePerThread, threadsPerWarp, warpsPerCTA, order, CTALayout);
    }]>,
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
        "ArrayRef<unsigned>":$sizePerThread,
        "ArrayRef<unsigned>":$order,
        "unsigned":$numWarps,
        "unsigned":$numThreadsPerWarp,
        "unsigned":$numCTAs), [{
      unsigned rank = sizePerThread.size();
      llvm::SmallVector<unsigned, 4> CTAsPerCGA(rank);
      llvm::SmallVector<unsigned, 4> CTASplitNum(rank);
      llvm::ArrayRef<unsigned> CTAOrder = order;

      unsigned remainingCTAs = numCTAs;

      // starting from the most strided dimension
      for (int d = rank - 1; d >= 0; --d) {
        unsigned i = order[d];
        CTAsPerCGA[i] = std::clamp<unsigned>(remainingCTAs, 1, shape[i] / sizePerThread[i]);
        CTASplitNum[i] = CTAsPerCGA[i];
        remainingCTAs /= CTAsPerCGA[i];
      }

      CTAsPerCGA[rank - 1] *= remainingCTAs; // wrap at CTA level

      CTALayoutAttr CTALayout = CTALayoutAttr::get(context, CTAsPerCGA, CTASplitNum, CTAOrder);
      return get(context, shape, sizePerThread, order, numWarps, numThreadsPerWarp, CTALayout);
    }]>
  ];

  let extraClassDeclaration = extraDistributedDeclaration # [{
    SliceEncodingAttr squeeze(int axis);
  }];

  let hasCustomAssemblyFormat = 1;
}

def SharedEncodingAttr : TritonGPU_Attr<"SharedEncoding", "shared_encoding"> {
  let parameters = (
    ins
    "unsigned":$vec,
    "unsigned":$perPhase,
    "unsigned":$maxPhas,
    ArrayRefParameter<"unsigned">:$order,
    "CTALayoutAttr":$CTALayout,
    "bool":$hasLeadingOffset
  );
  let extraClassDeclaration = extraBaseClassDeclaration;
}

def AMDMfmaEncodingAttr : DistributedEncoding<"AMDMfmaEncoding", "amd_mfma_encoding", [MmaEncodingTrait]> {
  // let mnemonic = "amd_mfma";

  let parameters = (
    ins
    "unsigned": $versionMajor,
    "unsigned": $versionMinor,
    ArrayRefParameter<"unsigned">:$warpsPerCTA__,
    "unsigned":$MDim,
    "unsigned":$NDim,
    "bool":$isTransposed,
    "CTALayoutAttr":$CTALayout
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{
  }];
}

#endif
