
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "third_party/arm_compute/arm_compute/graph.h"
#include "third_party/arm_compute/arm_compute/runtime/Tensor.h"
#include "third_party/arm_compute/arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"

using namespace arm_compute;

int main(int argc, char** argv) {
  arm_compute::Tensor a1 { }, b { }, c1 { };
  size_t M1 = 20;
  size_t K = 3000;
  size_t N = 1000;

  a1.allocator()->init(TensorInfo(TensorShape(K, M1), 1, DataType::S8));
  b.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::S8));
  c1.allocator()->init(TensorInfo(TensorShape(N, M1), 1, DataType::S32));

  // Import memory
  uint8_t* dataA1 = new uint8_t[M1*K];
  uint8_t* dataB = new uint8_t[N*K];
  int32_t* dataC1 = new int32_t[N*M1];

  a1.allocator()->import_memory(dataA1, M1*K);
  b.allocator()->import_memory(dataB, N*K);
  c1.allocator()->import_memory(dataC1, N*M1);

  for (int i = 0; i < M1; ++i) {
    for (int j = 0; j < K; ++j) {
      dataA1[i*K+j] = i;
    }
  }

  // Fill random value.
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      dataB[i*N+j] = M1+i;
    }
  }

  // Create gemm function here, since B is the same, so we create only one object.
  arm_compute::NEGEMMLowpMatrixMultiplyCore gemm_core;
  size_t M2 = 30;
  uint8_t* dataA2 = new uint8_t[M2*K];
  int32_t* dataC2 = new int32_t[N*M2];

  for (int i = 0; i < M2; ++i) {
    for (int j = 0; j < K; ++j) {
      dataA2[i*K+j] = i;
    }
  }

  // a1 is different from a2, BUT B is the same.
  arm_compute::Tensor a2 { }, c2 { };
  a2.allocator()->init(TensorInfo(TensorShape(K, M2), 1, DataType::S8));
  c2.allocator()->init(TensorInfo(TensorShape(N, M2), 1, DataType::S32));

  a2.allocator()->import_memory(dataA2, M2*K);
  c2.allocator()->import_memory(dataC2, N*M2);

  // Call gemm
  arm_compute::GEMMInfo info(false, false, true);
  for (int i = 0; i < 10000000; ++i) {
    // Apply gemm for tensor a1
    gemm_core.configure(&a1, &b, nullptr, &c1, info);
    gemm_core.run();

    //  Apply gemm for tensor a2
    gemm_core.configure(&a2, &b, nullptr, &c2, info);
    gemm_core.run();
  }

  return 0;
}



