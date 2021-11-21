#include "mesh_simplify.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

struct GlobalConstants {
    vertex_t *V;
    int *F;
    int *H;
    int v;
    int f;
}

__constant__ GlobalConstantants renderParams;

int *error_matrix;
__constant__ int num_error_params = 10;


// Helper functions for vector operations

__device__ inline vertex_t add(vertex_t v1, vertex_t v2) {
    vertex_t r;
    r.x = v1.x + v2.x;
    r.y = v1.y + v2.y;
    r.z = v1.z + v2.z;
    return r;
}

__device__ inline vertex_t subtract(vertex_t v1, vertex_t v2) {
    vertex_t r;
    r.x = v1.x - v2.x;
    r.y = v1.y - v2.y;
    r.z = v1.z - v2.z;
    return r;
}


// Generates the initial quadric error matrix
__global__ void kernel_init_qerror() {
    int index = blockIdx.x * blockDim.x + threadId.x;
    vertex_t v1 = renderParams.V[renderParams.F[3 * index]];
    vertex_t v2 = renderParams.V[renderParams.F[3 * index + 1]];
    vertex_t v2 = renderParams.V[renderParams.F[3 * index + 2]];

    // Build plane equation
    vertex_t side1 = subtract(v1, v3);
    vertex_t side2 = subtract(v2, v3);

    int a = side1.y * side2.z - side1.z * side2.y;
    int b = side1.z * side2.x - side1.x * side2.z;
    int c = side1.x * side2.y - side1.y * side2.x;
    int d = -a * v1.x - b * v1.y - c * v1.z;

    // Add to the quadric error matrix
    for (int i = 0; i < 3; i++) {
        vertex_index = renderParams.F[3 * index + i];
        int *matrix_position = error_matrix + num_error_params * vertex_index;
        atomicAdd(matrix_position, a * a);
        atomicAdd(matrix_position + 1, a * b);
        atomicAdd(matrix_position + 2, a * c);
        atomicAdd(matrix_position + 3, a * d);
        atomicAdd(matrix_position + 4, b * b);
        atomicAdd(matrix_position + 5, b * c);
        atomicAdd(matrix_position + 6, b * d);
        atomicAdd(matrix_position + 7, c * c);
        atomicAdd(matrix_position + 8, c * d);
        atomicAdd(matrix_position + 9, d * d);
    }
}
