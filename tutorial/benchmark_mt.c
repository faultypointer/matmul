// clang-17 -O2 -mno-avx512f -march=native -fopenmp benchmark_mt.c -o benchmark_mt.out && ./benchmark_mt.out
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MEM_ALIGN 64

#define MR 16
#define NR 6

#define NTHREADS 8
#define MC MR* NTHREADS * 1
#define NC NR* NTHREADS * 80
#define KC 1000

#ifndef NITER
#define NITER 500
#endif

#ifndef MINSIZE
#define MINSIZE 200
#endif

#ifndef MAXSIZE
#define MAXSIZE 2000
#endif

#ifndef NPTS
#define NPTS 50
#endif

#define min(x, y) ((x) < (y) ? (x) : (y))

static float blockA_packed[MC * KC] __attribute__((aligned(MEM_ALIGN)));
static float blockB_packed[NC * KC] __attribute__((aligned(MEM_ALIGN)));

void pack_panelB(float* B, float* blockB_packed, const int nr, const int kc, const int K) {
  for (int p = 0; p < kc; p++) {
    for (int j = 0; j < nr; j++) {
      *blockB_packed++ = B[j * K + p];
    }
    for (int j = nr; j < NR; j++) {
      *blockB_packed++ = 0;
    }
  }
}

void pack_blockB(float* B, float* blockB_packed, const int nc, const int kc, const int K) {
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
  for (int j = 0; j < nc; j += NR) {
    const int nr = min(NR, nc - j);
    pack_panelB(&B[j * K], &blockB_packed[j * kc], nr, kc, K);
  }
}

void pack_panelA(float* A, float* blockA_packed, const int mr, const int kc, const int M) {
  for (int p = 0; p < kc; p++) {
    for (int i = 0; i < mr; i++) {
      *blockA_packed++ = A[p * M + i];
    }
    for (int i = mr; i < MR; i++) {
      *blockA_packed++ = 0;
    }
  }
}

void pack_blockA(float* A, float* blockA_packed, const int mc, const int kc, const int M) {
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
  for (int i = 0; i < mc; i += MR) {
    const int mr = min(MR, mc - i);
    pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, M);
  }
}

void kernel_16x6(float* blockA_packed, float* blockB_packed, float* C, const int m, const int n, const int k,
                 const int M) {

  __m256 C_buffer[2][6];
  __m256 b_packFloat8;
  __m256 a0_packFloat8;
  __m256 a1_packFloat8;
  __m256i masks[2];

  if (m != MR) {
    const unsigned int bit_mask = 65535;
    masks[0] = _mm256_setr_epi32(bit_mask << (m + 15), bit_mask << (m + 14), bit_mask << (m + 13), bit_mask << (m + 12),
                                 bit_mask << (m + 11), bit_mask << (m + 10), bit_mask << (m + 9), bit_mask << (m + 8));
    masks[1] = _mm256_setr_epi32(bit_mask << (m + 7), bit_mask << (m + 6), bit_mask << (m + 5), bit_mask << (m + 4),
                                 bit_mask << (m + 3), bit_mask << (m + 2), bit_mask << (m + 1), bit_mask << m);

    for (int j = 0; j < n; j++) {
      C_buffer[0][j] = _mm256_maskload_ps(&C[j * M], masks[0]);
      C_buffer[1][j] = _mm256_maskload_ps(&C[j * M + 8], masks[1]);
    }
  } else {
    for (int j = 0; j < n; j++) {
      C_buffer[0][j] = _mm256_loadu_ps(&C[j * M]);
      C_buffer[1][j] = _mm256_loadu_ps(&C[j * M + 8]);
    }
  }
  for (int p = 0; p < k; p++) {

    a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
    a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
    C_buffer[0][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][0]);
    C_buffer[1][0] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][0]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
    C_buffer[0][1] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][1]);
    C_buffer[1][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][1]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
    C_buffer[0][2] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][2]);
    C_buffer[1][2] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][2]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
    C_buffer[0][3] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][3]);
    C_buffer[1][3] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][3]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
    C_buffer[0][4] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][4]);
    C_buffer[1][4] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][4]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
    C_buffer[0][5] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_buffer[0][5]);
    C_buffer[1][5] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_buffer[1][5]);

    blockA_packed += MR;
    blockB_packed += NR;
  }
  if (m != MR) {
    for (int j = 0; j < n; j++) {
      _mm256_maskstore_ps(&C[j * M], masks[0], C_buffer[0][j]);
      _mm256_maskstore_ps(&C[j * M + 8], masks[1], C_buffer[1][j]);
    }
  } else {
    for (int j = 0; j < n; j++) {
      _mm256_storeu_ps(&C[j * M], C_buffer[0][j]);
      _mm256_storeu_ps(&C[j * M + 8], C_buffer[1][j]);
    }
  }
}

void matmul_parallel(float* A, float* B, float* C, const int M, const int N, const int K) {
  for (int j = 0; j < N; j += NC) {
    const int nc = min(NC, N - j);
    for (int p = 0; p < K; p += KC) {
      const int kc = min(KC, K - p);
      pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);
      for (int i = 0; i < M; i += MC) {
        const int mc = min(MC, M - i);
        pack_blockA(&A[p * M + i], blockA_packed, mc, kc, M);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
        for (int jr = 0; jr < nc; jr += NR) {
          const int nr = min(NR, nc - jr);
          for (int ir = 0; ir < mc; ir += MR) {
            const int mr = min(MR, mc - ir);
            kernel_16x6(&blockA_packed[ir * kc], &blockB_packed[jr * kc], &C[(j + jr) * M + (i + ir)], mr, nr, kc, M);
          }
        }
      }
    }
  }
}

void init_rand(float* mat, const int M, const int N) {
  for (int i = 0; i < M * N; i++) {
    *mat++ = rand() / (float)RAND_MAX;
  }
}

void init_const(float* mat, const float value, const int M, const int N) {
  for (int i = 0; i < M * N; i++) {
    *mat++ = value;
  }
}

uint64_t timer() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

int main() {
  int avg_gflops[NPTS];
  int min_gflops[NPTS];
  int max_gflops[NPTS];

  // float* blockA_packed = (float*)_mm_malloc(MC * KC * sizeof(float), MEM_ALIGN);
  // float* blockB_packed = (float*)_mm_malloc(NC * KC * sizeof(float), MEM_ALIGN);
  // init_const(blockA_packed, 0.0, MC, KC);
  // init_const(blockB_packed, 0.0, NC, KC);

  int mat_sizes[NPTS];
  double size_increment = (double)(MAXSIZE - MINSIZE) / (NPTS - 1);
  for (int i = 0; i < NPTS - 1; i++) {
    mat_sizes[i] = MINSIZE + i * size_increment;
  }
  mat_sizes[NPTS - 1] = MAXSIZE;

  // Warmup
  float* A = (float*)_mm_malloc(MAXSIZE * MAXSIZE * sizeof(float), MEM_ALIGN);
  float* B = (float*)_mm_malloc(MAXSIZE * MAXSIZE * sizeof(float), MEM_ALIGN);
  float* C = (float*)_mm_malloc(MAXSIZE * MAXSIZE * sizeof(float), MEM_ALIGN);
  for (int j = 0; j < 20; j++) {
    init_const(C, 0.0, MAXSIZE, MAXSIZE);
    matmul_parallel(A, B, C, MAXSIZE, MAXSIZE, MAXSIZE);
  }
  _mm_free(A);
  _mm_free(B);
  _mm_free(C);

  for (int i = 0; i < NPTS; i++) {
    int mat_size = mat_sizes[i];
    float* A = (float*)_mm_malloc(mat_size * mat_size * sizeof(float), MEM_ALIGN);
    float* B = (float*)_mm_malloc(mat_size * mat_size * sizeof(float), MEM_ALIGN);
    float* C = (float*)_mm_malloc(mat_size * mat_size * sizeof(float), MEM_ALIGN);

    init_rand(A, mat_size, mat_size);
    init_rand(B, mat_size, mat_size);

    double FLOP = 2 * (double)mat_size * mat_size * mat_size;
    double avg_exec_time = 0;
    double max_exec_time = 0;
    double min_exec_time = 1e69;

    for (int j = 0; j < NITER; j++) {
      init_const(C, 0.0, mat_size, mat_size);
      uint64_t start = timer();
      matmul_parallel(A, B, C, mat_size, mat_size, mat_size);
      uint64_t end = timer();
      float exec_time = (end - start) * 1e-9;
      max_exec_time = exec_time > max_exec_time ? exec_time : max_exec_time;
      min_exec_time = exec_time < min_exec_time ? exec_time : min_exec_time;
      avg_exec_time += exec_time;
    }

    avg_exec_time /= NITER;
    avg_gflops[i] = (int)(FLOP / avg_exec_time / 1e9);
    max_gflops[i] = (int)(FLOP / min_exec_time / 1e9);
    min_gflops[i] = (int)(FLOP / max_exec_time / 1e9);

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
  }

  FILE* fptr;
  const char* filename = "benchmark_c.txt";
  fptr = fopen(filename, "w");
  if (fptr == NULL) {
    printf("Error opening the file %s", filename);
    return -1;
  }
  for (int i = 0; i < NPTS; i++) {
    fprintf(fptr, "%i %i %i %i\n", mat_sizes[i], min_gflops[i], max_gflops[i], avg_gflops[i]);
  }
  fclose(fptr);
  return 0;
}