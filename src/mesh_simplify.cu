#include "mesh_simplify.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>


struct GlobalConstants {
    int vertex_cnt;
    int halfedge_cnt;
    int triangle_cnt;

    float3* vertices;
    int4* halfedges;
};

__constant__ GlobalConstants cuConstParams;

__global__ void kernelTest() {

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    cuConstParams.vertices[idx_x].x = idx_x;
    cuConstParams.vertices[idx_x].y = idx_x;
    cuConstParams.vertices[idx_x].z = idx_x;
}


// Helper functions for vector operations

/*
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
*/

void setup(mesh_t mesh) {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaSimplification\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("NVIDIA GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }

    float3* vertex_buffer;
    int4* halfedge_buffer;

    // Allocate space for the mesh
    cudaMalloc(&vertex_buffer, sizeof(float3) * mesh.vertex_cnt);
    cudaMalloc(&halfedge_buffer, sizeof(int4) * mesh.halfedge_cnt);

    // Copy the 3 x N and 4 x N arrays into vector types
    float3* verts = (float3*)calloc(mesh.vertex_cnt, sizeof(float3));
    int4* halfedge = (int4*)calloc(mesh.halfedge_cnt, sizeof(int4));

    for(int i = 0; i < mesh.vertex_cnt; i++) {
        verts[i].x = mesh.vertices[i * 3 + 0];
        verts[i].y = mesh.vertices[i * 3 + 1];
        verts[i].z = mesh.vertices[i * 3 + 2];
    }

    for(int i = 0; i < mesh.halfedge_cnt; i++) {
        halfedge[i].x = mesh.halfedges[i * 4 + 0];
        halfedge[i].y = mesh.halfedges[i * 4 + 1];
        halfedge[i].z = mesh.halfedges[i * 4 + 2];
        halfedge[i].w = mesh.halfedges[i * 4 + 3];
    }

    // Move the vector arrays to the device
    cudaMemcpy(vertex_buffer, verts, sizeof(float3) * mesh.vertex_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(halfedge_buffer, halfedge, sizeof(int4) * mesh.halfedge_cnt, cudaMemcpyHostToDevice);

    // Free the local arrays since we don't need them anymore
    free(verts);
    free(halfedge);

    // Pass all these parameters to the GPU
    GlobalConstants params;
    params.vertex_cnt    = mesh.vertex_cnt;
    params.halfedge_cnt  = mesh.halfedge_cnt;
    params.triangle_cnt  = mesh.triangle_cnt;
    params.vertices      = vertex_buffer;
    params.halfedges     = halfedge_buffer;

    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));
}

void simplify(mesh_t mesh) {
    
    // 256 threads per block is a healthy number
    int box_size = 256;
    dim3 blockDim(box_size);
    dim3 gridDim((mesh.vertex_cnt  + blockDim.x - 1) / blockDim.x);

    kernelTest<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();


}

void get_results() {
    mesh_t mesh;

    
    // Pass all these parameters to the GPU
    GlobalConstants params;
    cudaMemcpyFromSymbol(&params, cuConstParams, sizeof(GlobalConstants));

    mesh.vertex_cnt = params.vertex_cnt;
    mesh.halfedge_cnt = params.halfedge_cnt;
    mesh.triangle_cnt = params.triangle_cnt;

    // Copy the 3 x N and 4 x N arrays into vector types
    float3* verts = (float3*)calloc(mesh.vertex_cnt, sizeof(float3));
    int4* halfedges = (int4*)calloc(mesh.halfedge_cnt, sizeof(int4));

    // Copy the data back from the GPU
    cudaMemcpy(verts, params.vertices, sizeof(float3) * mesh.vertex_cnt, cudaMemcpyDeviceToHost);
    cudaMemcpy(halfedges, params.halfedges, sizeof(int4) * mesh.halfedge_cnt, cudaMemcpyDeviceToHost);


    for(int i = 0; i < mesh.vertex_cnt; i++) {
        printf("%f %f %f\n", 
                verts[i].x,
                verts[i].y,
                verts[i].z);
    }

    
    for(int i = 0; i < mesh.halfedge_cnt; i++) {
        printf("%d %d %d %d\n", 
                halfedges[i].x,
                halfedges[i].y,
                halfedges[i].z,
                halfedges[i].w);
    }
}

