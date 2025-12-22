#include <pybind11/pybind11.h>

int run(void);

PYBIND11_MODULE(bind, m) {
  m.def("run", run);
}
