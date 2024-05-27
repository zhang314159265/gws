#include <iostream>
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "toy/AST.h"
#include "toy/Dump.h"
#include "toy/Lexer.h"
#include "toy/Parser.h"
#include "toy/Dialect.h"
#include "toy/MLIRGen.h"
#include "toy/Pass.h"

using namespace toy;
namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(
    cl::Positional,
    cl::desc("<input toy file>"),
    cl::init("-"),
    cl::value_desc("filename"));

namespace {
enum Action {
  None,
  DumpAST,
  DumpMLIR,
  DumpMLIRLLVM,
  DumpLLVMIR,
  RunJIT
};
enum InputType { Toy, MLIR };
}

static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Decided the kind of input desired"),
    cl::values(clEnumValN(Toy, "toy", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir", "load the input file as an MLIR file")));

static cl::opt<enum Action>
    emitAction("emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
        "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(clEnumValN(RunJIT, "jit",
        "JIT the code and run it by involing the main function")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
    return 5;
  }
  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST) {
    return 1;
  }

  dump(*moduleAST);
  return 0;
}

int loadMLIR(llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
    mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Handle '.toy' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).ends_with(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST) {
      return 6;
    }
    module = mlirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  // Otherwise, the input is '.mlir'
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  llvm::SourceMgr sourceMgr; 
  if (int error = loadMLIR(sourceMgr, context, module)) {
    return error;
  }

  mlir::PassManager pm(module.get()->getName());

  if (enableOpt) {
    {
      // Inline all functions into main and then delete them.
      pm.addPass(mlir::createInlinerPass());
  
      mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
      optPM.addPass(toy::createShapeInferencePass());
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());
  
    }
    {
      pm.addPass(toy::createLowerToAffinePass());

      // Add a few cleanups post lowering.
      mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());

      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }
  }

  bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

  if (isLoweringToLLVM) {
    pm.addPass(toy::createLowerToLLVMPass());
    // This is necessary to have line tables emitted and basic
    // debugger working. In the future we will add proper debug information
    // emission directly from our frontend.
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }

  if (mlir::failed(pm.run(*module))) {
    return 4;
  }

  return 0;
}

// LLVM dialect to LLVM IR
int dumpLLVMIR(mlir::ModuleOp module) {
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Faield to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Configure the LLVM module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(
      llvmModule.get(),
      tmOrError.get().get());

  // Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

int runJit(mlir::ModuleOp module) {
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "faield to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Inoke the JIT-compiled function
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }
  return 0;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
  // std::cerr << "inputFilename is " << inputFilename << std::endl;
  
  if (emitAction == Action::DumpAST) {
    return dumpAST();
  }

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module)) {
    return error;
  }

  bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
  if (isOutputingMLIR) {
    module->dump();
    return 0;
  }

  if (emitAction == Action::DumpLLVMIR) {
    return dumpLLVMIR(*module);
  }

  if (emitAction == Action::RunJIT) {
    return runJit(*module);
  }

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return 0;
}
