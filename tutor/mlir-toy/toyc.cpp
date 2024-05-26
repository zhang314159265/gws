#include <iostream>
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"

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
enum Action { None, DumpAST, DumpMLIR };
enum InputType { Toy, MLIR };
}

static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Decided the kind of input desired"),
    cl::values(clEnumValN(Toy, "toy", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir", "load the input file as an MLIR file")));

static cl::opt<enum Action>
    emitAction("emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

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

int dumpMLIR() {
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr sourceMgr; 
  if (int error = loadMLIR(sourceMgr, context, module)) {
    return error;
  }

  if (enableOpt) {
    {
      mlir::PassManager pm(module.get()->getName());
  
      // Inline all functions into main and then delete them.
      pm.addPass(mlir::createInlinerPass());
  
      mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
      optPM.addPass(toy::createShapeInferencePass());
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());
  
      if (mlir::failed(pm.run(*module))) {
        return 4;
      }
    }
    {
      mlir::PassManager pm(module.get()->getName());
      pm.addPass(toy::createLowerToAffinePass());

      // Add a few cleanups post lowering.
      mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());

      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());

      if (mlir::failed(pm.run(*module))) {
        return 4;
      }
    }
  }

  module->dump();
  return 0;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
  // std::cerr << "inputFilename is " << inputFilename << std::endl;
  
  switch (emitAction) {
  case Action::DumpAST:
    return dumpAST();
  case Action::DumpMLIR:
    return dumpMLIR();
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }
  std::cerr << "bye" << std::endl;
  return 0;
}
