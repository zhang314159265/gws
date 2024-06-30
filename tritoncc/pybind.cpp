#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "tritoncc/LocationOpBuilder.h"
#include "tritoncc/dialect_util.h"

namespace py = pybind11;
using MyOpBuilder = tritoncc::LocationOpBuilder;

void init_triton_ir(py::module &&m) {
  using ret = py::return_value_policy;

  py::class_<mlir::MLIRContext>(m, "context", py::module_local()).def(py::init<>());

  py::class_<mlir::OpState>(m, "OpState", py::module_local())
      ;

  py::class_<mlir::Block>(m, "block", py::module_local())
      ;

  const auto& operation_to_str = [](mlir::Operation &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      };

  py::class_<mlir::Operation, std::unique_ptr<mlir::Operation, py::nodelete>>(
      m, "operation", py::module_local())
      .def("__str__", operation_to_str)
      .def("__repr__", operation_to_str)
      .def("get_containing_module", [](mlir::Operation &op) {
        return op.getParentOfType<mlir::ModuleOp>();
      })
      ;
 
  const auto& value_to_str = [](mlir::Value &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      };
 
  py::class_<mlir::Value>(m, "value", py::module_local())
      .def("get_type", &mlir::Value::getType)
      .def("type", &mlir::Value::getType)
      .def("__str__", value_to_str)
      .def("__repr__", value_to_str)
      .def("get_defining_op", [](mlir::Value &self) {
        return self.getDefiningOp();
      })
      ;

  py::class_<mlir::BlockArgument, mlir::Value>(m, "block_argument", py::module_local());

  py::class_<mlir::_tritoncc::FuncOp, mlir::OpState>(m, "function", py::module_local())
      .def(
          "add_entry_block",
          [](mlir::_tritoncc::FuncOp &self) -> mlir::Block * {
            return self.addEntryBlock();
          }, ret::reference)
      .def("args",
          [](mlir::_tritoncc::FuncOp &self, unsigned idx) -> mlir::BlockArgument {
            if (idx >= self.getNumArguments()) {
              throw py::index_error(
                  "Function argument index out of range");
            }
            return self.getArgument(idx);
          })
      .def(
          "num_args",
          [](mlir::_tritoncc::FuncOp &self) -> int {
            return self.getNumArguments();
          })
      .def("finalize",
          [](mlir::_tritoncc::FuncOp &self) -> void {
            // Do nothing for now
          })
    ;

  py::enum_<mlir::_tritoncc::CacheModifier>(m, "CACHE_MODIFIER", py::module_local())
      .value("NONE", mlir::_tritoncc::CacheModifier::NONE)
      .export_values()
      ;
  py::enum_<mlir::_tritoncc::EvictionPolicy>(m, "EVICTION_POLICY", py::module_local())
      .value("NORMAL", mlir::_tritoncc::EvictionPolicy::NORMAL)
      .export_values()
      ;

  py::class_<MyOpBuilder>(m, "builder", py::module_local(),
          py::dynamic_attr())
      .def(py::init<mlir::MLIRContext *>())
      .def("create_module",
          [](MyOpBuilder &self) -> mlir::ModuleOp {
            return self.create<mlir::ModuleOp>();
          })
      .def("get_or_insert_function",
          [](MyOpBuilder &self, mlir::ModuleOp &module, std::string &funcName,
              mlir::Type &funcType, std::string &visibility,
              bool noinline) -> mlir::_tritoncc::FuncOp {
            if (mlir::Operation *funcOperation = module.lookupSymbol(funcName)) {
              return llvm::dyn_cast<mlir::_tritoncc::FuncOp>(funcOperation);
            }
            if (auto funcTy = funcType.dyn_cast<mlir::FunctionType>()) {
              llvm::SmallVector<mlir::NamedAttribute> attrs = {
                mlir::NamedAttribute(
                  self.getBuilder().getStringAttr("sym_visibility"),
                  self.getBuilder().getStringAttr(visibility)),
                mlir::NamedAttribute(self.getBuilder().getStringAttr("noinline"),
                  self.getBuilder().getBoolAttr(noinline))
              };
              return self.create<mlir::_tritoncc::FuncOp>(funcName, funcTy, attrs);
            }
            throw std::invalid_argument("invalid function type");
          })
      .def("get_float_ty",
          [](MyOpBuilder &self) -> mlir::Type {
            return self.getBuilder().getF32Type();
          })
      .def("get_ptr_ty",
          [](MyOpBuilder &self, mlir::Type &type, int addrSpace) -> mlir::Type {
            return mlir::_tritoncc::PointerType::get(type, addrSpace);
          })
      .def("get_int32_ty",
          [](MyOpBuilder &self) -> mlir::Type {
            return self.getBuilder().getI32Type();
          })
      .def("get_block_ty",
          [](MyOpBuilder &self, mlir::Type &elementType,
              std::vector<int64_t> &shape) -> mlir::Type {
            return mlir::RankedTensorType::get(shape, elementType);
          })
      .def("get_function_ty",
          [](MyOpBuilder &self, std::vector<mlir::Type> inTypes,
              std::vector<mlir::Type> outTypes) -> mlir::Type {
            return self.getBuilder().getFunctionType(inTypes, outTypes);
          })
      .def("get_int32",
          [](MyOpBuilder &self, int64_t v) -> mlir::Value {
            return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
              v, self.getBuilder().getI32Type()));
          })
      .def("set_insertion_point_to_start",
          [](MyOpBuilder &self, mlir::Block &block) -> void {
            self.setInsertionPointToStart(&block);
          })
      .def("get_insertion_block",
          [](MyOpBuilder &self) -> mlir::Block * {
            return self.getBuilder().getInsertionBlock();
          },
          ret::reference)
      .def("ret",
          [](MyOpBuilder &self, std::vector<mlir::Value> &vals) -> mlir::OpState {
            return self.create<mlir::_tritoncc::ReturnOp>(vals);
          })
      .def("create_masked_load",
          [](MyOpBuilder &self, mlir::Value &ptrs, mlir::Value &mask,
             std::optional<mlir::Value> &other, mlir::_tritoncc::CacheModifier cacheModifier,
             mlir::_tritoncc::EvictionPolicy evictionPolicy, bool isVolatile) -> mlir::Value {
             return self.create<mlir::_tritoncc::LoadOp>(ptrs, mask, other.value_or(mlir::Value()),
               cacheModifier, evictionPolicy, isVolatile);
           })
      .def("create_masked_store",
          [](MyOpBuilder &self, mlir::Value &ptrs, mlir::Value &val, mlir::Value &mask,
              mlir::_tritoncc::CacheModifier cacheModifier,
              mlir::_tritoncc::EvictionPolicy evictionPolicy) -> void {
            self.create<mlir::_tritoncc::StoreOp>(ptrs, val, mask, cacheModifier,
                evictionPolicy);
          })
      .def("create_icmpULT",
          [](MyOpBuilder &self, mlir::Value &lhs, mlir::Value &rhs) -> mlir::Value {
            return self.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ult, lhs, rhs);
          })
      .def("create_addptr",
          [](MyOpBuilder &self, mlir::Value &ptr, mlir::Value &offset) -> mlir::Value {
            return self.create<mlir::_tritoncc::AddPtrOp>(ptr.getType(), ptr, offset);
          })
      .def("create_add",
          [](MyOpBuilder &self, mlir::Value &lhs, mlir::Value &rhs) -> mlir::Value {
            return self.create<mlir::arith::AddIOp>(lhs, rhs);
          })
      .def("create_fadd",
          [](MyOpBuilder &self, mlir::Value &lhs, mlir::Value &rhs) -> mlir::Value {
            return self.create<mlir::arith::AddFOp>(lhs, rhs);
          })
      .def("create_fsub",
          [](MyOpBuilder &self, mlir::Value &lhs, mlir::Value &rhs) -> mlir::Value {
            return self.create<mlir::arith::SubFOp>(lhs, rhs);
          })
      .def("create_make_range",
          [](MyOpBuilder &self, int start, int end) -> mlir::Value {
            auto retType = mlir::RankedTensorType::get(
              {end - start}, self.getBuilder().getI32Type());
            return self.create<mlir::_tritoncc::MakeRangeOp>(retType, start, end);
          })
      .def("create_get_program_id",
          [](MyOpBuilder &self, int axis) -> mlir::Value {
            return self.create<mlir::_tritoncc::GetProgramIdOp>(
              self.getBuilder().getI32Type(),
              mlir::_tritoncc::ProgramIDDimAttr::get(self.getBuilder().getContext(),
                mlir::_tritoncc::ProgramIDDim(axis)));
          })
      .def("create_mul",
          [](MyOpBuilder &self, mlir::Value &lhs, mlir::Value &rhs) -> mlir::Value {
            return self.create<mlir::arith::MulIOp>(lhs, rhs);
          })
      .def("create_splat",
          [](MyOpBuilder &self, mlir::Value &arg,
              std::vector<int64_t> &shape) -> mlir::Value {
            auto argType = arg.getType();
            auto ret = self.createOrFold<mlir::_tritoncc::SplatOp>(
              mlir::RankedTensorType::get(shape, argType), arg);
            return ret;
          })
      ;

  const auto& type_to_str = [](mlir::Type &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      };
  py::class_<mlir::Type>(m, "type", py::module_local())
      .def("__str__", type_to_str)
      .def("__repr__", type_to_str)
      .def("scalar", [](mlir::Type &self) {
        if (auto tensor_type = self.dyn_cast<mlir::RankedTensorType>()) {
          return tensor_type.getElementType();
        } else {
          return self;
        }
      })
      .def("is_ptr", [](mlir::Type &self) {
        return self.isa<mlir::_tritoncc::PointerType>();
      })
      .def("element_ty", [](mlir::Type &self) {
        auto ptr_ty = self.dyn_cast<mlir::_tritoncc::PointerType>();
        assert(ptr_ty);
        return ptr_ty.getPointeeType();
      })
      .def("is_floating", [](mlir::Type &self) {
        return !self.isSignedInteger() && !self.isSignlessInteger() && self.isIntOrFloat();
      })
      .def("is_bool", [](mlir::Type &self) {
        return self.isInteger(1);
      })
      .def("is_int", [](mlir::Type &self) {
        return self.isSignedInteger() || self.isSignlessInteger();
      })
      .def("is_int_signed", [](mlir::Type &self) {
        return self.isSignedInteger();
      })
      .def("is_int_signless", [](mlir::Type &self) {
        return self.isSignlessInteger();
      })
      .def("is_block", [](mlir::Type &self) {
        return self.isa<mlir::RankedTensorType>();
      })
      .def("get_block_shape", [](mlir::Type &self) -> std::vector<int64_t> {
        auto block_ty = self.dyn_cast<mlir::RankedTensorType>();
        return block_ty.getShape().vec();
      })
    ;

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    tritoncc::loadDialects(context);
  });

  const auto& module_to_str = [](mlir::ModuleOp &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      };

  py::class_<mlir::ModuleOp, mlir::OpState>(m, "module", py::module_local(),
      py::dynamic_attr())
      .def("__str__", module_to_str)
      .def("__repr__", module_to_str)
      .def("push_back",
          [](mlir::ModuleOp &self, mlir::_tritoncc::FuncOp &funcOp) -> void {
            self.push_back(funcOp);
          })
      ;
}

PYBIND11_MODULE(_C, m) {
  m.doc() = "Python binings";
  init_triton_ir(m.def_submodule("ir"));
}
