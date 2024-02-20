#pragma once

#include <fstream>
#include <iostream>
#include <cstdlib>

namespace tritoncc {

std::string makeCubin(std::string& ptxCode, Option& opt) {
  // example ptxas command
  // triton/ptxas -lineinfo -v --gpu-name=sm_90a /tmp/tmpq_mc2q9k.ptx -o /tmp/tmpq_mc2q9k.ptx.o 2> /tmp/tmpgb9rb2qg.log
  
  { // create ptx file
    std::ofstream out_ptx;
    out_ptx.open("/tmp/tritoncc.ptx");
    out_ptx << ptxCode;
    out_ptx.close();

    std::cerr << "Written ptx code to /tmp/tritoncc.ptx" << std::endl;
  }
  { // run the ptx command
    std::string cmd = "/home/shunting/ws/triton/python/triton/backends/nvidia/bin/ptxas -lineinfo -v --gpu-name=sm_90a /tmp/tritoncc.ptx -o /tmp/tritoncc.cubin 2> /tmp/tritoncc.ptxas.log";
    int status_code = system(cmd.c_str());
    assert(status_code == 0 && "fail to run the ptxas command");
    std::cerr << "Write the cubin file to /tmp/tritoncc.cubin" << std::endl;
  }
  std::string cubinBytes;
  { // read the cubin file
    std::ifstream input("/tmp/tritoncc.cubin", std::ios::binary);
    cubinBytes = std::string(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>()
    );
  }
  std::cerr << "Got " << cubinBytes.size() << " bytes in the cubin file" << std::endl;
  return cubinBytes;
}

}
