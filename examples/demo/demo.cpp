#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include "ggml-cpu.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>


int main(void) {
    // initialize data of matrices to perform matrix multiplication
    const int rows_A = 4, cols_A = 2;
    float matrix_A[rows_A * cols_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };
    const int rows_B = 3, cols_B = 2;
    float matrix_B[rows_B * cols_B] = {
        10, 5,
        9, 9,
        5, 4
    };

    // 1. Initialize backend 
    ggml_backend_t backend = NULL; 

#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    backend = ggml_backend_cuda_init(0); // init device 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif 

    // if there aren't GPU Backends fallback to CPU backend
    if (!backend) {
        printf("Init cpu backend");
        backend = ggml_backend_cpu_init();
    }

    // Calculate the size needed to allocate 
    size_t ctx_size = 0;
    ctx_size += 2 * ggml_tensor_overhead(); // tensors 

    printf("ctx size: %ld", ctx_size);
    // no need to allocate anything else! 

    // 2. Allocate `ggml_context` to store tensor data 
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size, 
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    };

    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
    struct ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);


    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(tensor_a, matrix_A, 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, matrix_B, 0, ggml_nbytes(tensor_b));


    struct ggml_cgraph * gf = NULL;
    struct ggml_context * ctx_graph = NULL; 

    {
        // create a temp context to build the graph 
        struct ggml_init_params params0 = {
            /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
        };

        ctx_graph = ggml_init(params0);
        gf = ggml_new_graph(ctx_graph);

        // result = a * b^T
        // pay attention = ggml_mat_mul(A, B) ==> B will be tranposed internally 
        // the result is tranposed 
        struct ggml_tensor * result0 = ggml_mul_mat(ctx_graph, tensor_a, tensor_b);

        // add "result" tensor and all of its dependencies to the cgraph 
        ggml_build_forward_expand(gf, result0); 

    }


    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    int n_threads = 1;

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

    ggml_backend_graph_compute(backend, gf);

    // 10. Retrieve result 
    // in this example output tensor is always the last tensor in the graph 
    struct ggml_tensor * result = ggml_graph_node(gf, -1);
    float * result_data = (float*)malloc(ggml_nbytes(result));

    ggml_backend_tensor_get(result, result_data, 0, ggml_nbytes(result));
    printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++) {
            printf(" %.2f", result_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");
    free(result_data);

    ggml_free(ctx_graph);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return 0;
}
