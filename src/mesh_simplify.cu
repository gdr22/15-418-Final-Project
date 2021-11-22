#include "mesh_simplify.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "helper_math.h"


struct GlobalConstants {
    int vertex_cnt;
    int halfedge_cnt;
    int triangle_cnt;

    // 3D vertex positions for all verts in the mesh
    float3* vertices;

    // Halfedges in the mesh, stored as:
    //
    // x - vertex (head)
    // y - triangle
    // z - next
    // w - twin
    int4* halfedges;

    // For each triangle, the index to a halfedge which has this vertex as the head
    int* tri_halfedges;

    // The plane equations for each triangle
    float4* tri_planes;

    // The quadric error matrix coefficients for each vertex (stored in vertex major order)
    float* Qv;
};

__constant__ GlobalConstants cuConstParams;


/* Computes the plane equations for each of the triangles in the mesh, and stores them
 * in global memory so that we can reuse the values to compute the Q matrices for
 * all vertices
 */
__global__ void compute_triangle_quadrics() {

    // Get the triangle we are operating over
    int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Get the half_edges associated with this triangle
    int half_edge1 = cuConstParams.tri_halfedges[tri_idx];
    int half_edge2 = cuConstParams.halfedges[half_edge1].z;
    int half_edge3 = cuConstParams.halfedges[half_edge2].z;

    // Get the vertices of this halfedge
    int v1 = cuConstParams.halfedges[half_edge1].x;
    int v2 = cuConstParams.halfedges[half_edge2].x;
    int v3 = cuConstParams.halfedges[half_edge3].x;

    // Get the vertex positions
    float3 p1 = cuConstParams.vertices[v1];
    float3 p2 = cuConstParams.vertices[v2];
    float3 p3 = cuConstParams.vertices[v3];


    // Build plane equation
    float3 side1 = p1 - p3;
    float3 side2 = p2 - p3;

    float a = side1.y * side2.z - side1.z * side2.y;
    float b = side1.z * side2.x - side1.x * side2.z;
    float c = side1.x * side2.y - side1.y * side2.x;
    float d = -(a * p1.x) - (b * p1.y) - (c * p1.z);

    // Store the planes into memory (we normalize over d)
    cuConstParams.tri_planes[tri_idx].x = a / d;
    cuConstParams.tri_planes[tri_idx].y = b / d;
    cuConstParams.tri_planes[tri_idx].z = c / d;
    cuConstParams.tri_planes[tri_idx].w = d / d;
}


// Helper functions for vector operations
/*
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
    int* tri_halfedge_buffer;

    float4* tri_planes;

    // Allocate space for the mesh
    cudaMalloc(&vertex_buffer, sizeof(float3) * mesh.vertex_cnt);
    cudaMalloc(&halfedge_buffer, sizeof(int4) * mesh.halfedge_cnt);
    cudaMalloc(&tri_halfedge_buffer, sizeof(int) * mesh.triangle_cnt);
    cudaMalloc(&tri_planes, sizeof(float4) * mesh.triangle_cnt);

    // Copy the 3 x N and 4 x N arrays into vector types
    float3* verts = (float3*)calloc(mesh.vertex_cnt, sizeof(float3));
    int4* halfedge = (int4*)calloc(mesh.halfedge_cnt, sizeof(int4));
    int* tri_halfedges = (int*)calloc(mesh.triangle_cnt, sizeof(int));

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

        // Set the associated halfedge for whatever triangle this halfedge touches
        tri_halfedges[halfedge[i].y] = i;
    }

    // Just make sure this array is populated properly
    for(int i = 0; i < mesh.triangle_cnt; i++) {
        if(tri_halfedges[i] == 0) {
            printf("Triangle %d has no associated halfedge!\n");
        }
    }

    // Move the vector arrays to the device
    cudaMemcpy(vertex_buffer, verts, sizeof(float3) * mesh.vertex_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(halfedge_buffer, halfedge, sizeof(int4) * mesh.halfedge_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(tri_halfedge_buffer, tri_halfedges, sizeof(int) * mesh.triangle_cnt, cudaMemcpyHostToDevice);

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
    params.tri_halfedges = tri_halfedge_buffer;
    params.tri_planes    = tri_planes;

    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));
}

void simplify(mesh_t mesh) {
    
    // 256 threads per block is a healthy number
    int box_size = 256;
    dim3 blockDim(box_size);
    dim3 gridDim((mesh.triangle_cnt  + blockDim.x - 1) / blockDim.x);

    compute_triangle_quadrics<<<gridDim, blockDim>>>();
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

    /*
    for(int i = 0; i < mesh.halfedge_cnt; i++) {
        printf("%d %d %d %d\n", 
                halfedges[i].x,
                halfedges[i].y,
                halfedges[i].z,
                halfedges[i].w);
    }
    */
}

