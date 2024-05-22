#include <iostream>
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/StringRef.h"

#include "toy/AST.h"
#include "toy/Dump.h"
#include "toy/Lexer.h"
#include "toy/Parser.h"

using namespace toy;
namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(
    cl::Positional,
    cl::desc("<input toy file>"),
    cl::init("-"),
    cl::value_desc("filename"));

namespace {
enum Action { None, DumpAST };
}

static cl::opt<enum Action>
    emitAction("emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")));

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

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
  std::cerr << "inputFilename is " << inputFilename << std::endl;
  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST) {
    return 1;
  }
  
  switch (emitAction) {
  case Action::DumpAST:
    dump(*moduleAST);
    return 0;
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }
  std::cerr << "bye" << std::endl;
  return 0;
}
