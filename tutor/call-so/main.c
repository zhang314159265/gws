#include <stdio.h>
#include <dlfcn.h>
#include <assert.h>

int main(void) {
  int a = 2, b = 3;
  void *handle = dlopen("add.so", RTLD_NOW);
  void *sym = dlsym(handle, "add");
  printf("sym is %p\n", sym);
  assert(sym);

  int (*fun)(int a, int b) = sym;
  int result = fun(a, b);
  printf("add %d and %d, get %d\n", a, b, result);
  printf("bye\n");
}
