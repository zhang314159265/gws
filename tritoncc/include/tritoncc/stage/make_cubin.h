#pragma once

namespace tritoncc {

std::string make_cubin(std::string &ptxCode, Option &opt) {
  { // create PTX file
    std::ofstream out_ptx;
    out_ptx.open("/tmp/tritoncc.ptx");
    out_ptx << ptxCode;
    out_ptx.close();

    std::cerr << "Written the ptx code to /tmp/tritoncc.ptx" << std::endl;
  }
  { // run the ptxas command
    // we use the system ptxas here. An alternative is to use the ptxas under
    // triton/third_party/nvidia/backend/bin/ptxas
    std::string cmd = "ptxas -lineinfo -v --gpu-name=sm_90a /tmp/tritoncc.ptx -o /tmp/tritoncc.cubin 2> /tmp/tritoncc.ptxas.log";
    int status_code = system(cmd.c_str());
    assert(status_code == 0 && "fail to run the ptxas command");
    std::cerr << "Written the cubin file to /tmp/tritoncc.cubin" << std::endl;
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
