struct Options {
  bool error = false;
  bool reference_check = true;
  bool causal = false;

  int head_number = 12;
  int batch_size = 16;
  int head_size = 64;
  int head_size_v = 64;
  int seq_length = 1024;
  int seq_length_kv = 1024;
  int iterations = 20;

  float alpha0 = 1.0f / sqrt(64);
  float alpha1 = 1.0f;
  float beta = 0;

  std::vector<cutlass::gemm::GemmCoord> problem_sizes0;
  std::vector<cutlass::gemm::GemmCoord> problem_sizes1;

  void randomize_problems() {

    int problem_count = head_number * batch_size;

    problem_sizes0.reserve(problem_count);
    problem_sizes1.reserve(problem_count);

    for (int i = 0; i < batch_size; ++i) {
      // problems belonging to the same batch share the same seq len
      int m = seq_length;
      int mkv = seq_length_kv;
      int k0 = head_size;
      int k1 = head_size_v;

      for (int j = 0; j < head_number; ++j) {
        cutlass::gemm::GemmCoord problem0(m, mkv, k0);
        cutlass::gemm::GemmCoord problem1(m, k1, mkv);
        problem_sizes0.push_back(problem0);
        problem_sizes1.push_back(problem1);
      }
    }
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fops = int64_t();

    for (size_t i = 0; i < problem_sizes0.size(); ++i) {
      auto const& problem0 = problem_sizes0[i];
      auto const& problem1 = problem_sizes1[i];
      for (int row = 0; row < problem0.m(); ++row) {
        int num_cols0 = problem0.n();
        if (causal) {
          num_cols0 = std::min(row + 1, num_cols0);
        }
        // P <- Q . K_t
        fops += 2 * num_cols0 * problem0.k();
        // P <- exp(P - max(P))
        fops += 2 * num_cols0;
        // S <- sum(P)
        fops += num_cols0 - 1;
        // O <- P . V
        fops += 2 * num_cols0 * problem1.n();
        // O <- O / S
        fops += num_cols0 * problem1.n();
      }
    }

    return double(fops) / double(1.0e9) / runtime_s;
  }
};

/// Helper to initialize a tensor view
template <typename Element>
void initialize_tensor_(
  Element *ptr,
  size_t capacity, 
  uint32_t seed) {
    Element scope_max, scope_min;
    scope_max = 8;
    scope_min = -8;

    cutlass::reference::device::BlockFillRandomUniform(
      ptr, capacity, seed, scope_max, scope_min, 0);
}

