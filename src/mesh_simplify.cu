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

    // For each triangle, the index to a halfedge which points to this triangle
    int* tri_halfedges;

    // For each vertex, the index to a halfedge which has this vertex as the head
    int* vert_halfedges;

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


/* Computes the Q matrix coefficients for all vertices in the mesh
 * using the previously computed plane equations
 */
__global__ void compute_vertex_quadrics() {

    // Get the triangle we are operating over
    int vert_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Get the halfedge of this vertex
    int base_halfedge = cuConstParams.vert_halfedges[vert_idx];

    // Initialize the 9 coefficients of the Q matrix to zero
    float q1 = 0.f;
    float q2 = 0.f;
    float q3 = 0.f;
    float q4 = 0.f;
    float q5 = 0.f;
    float q6 = 0.f;
    float q7 = 0.f;
    float q8 = 0.f;
    float q9 = 0.f;

    // Loop around all faces incident on this vertex
    int halfedge = base_halfedge;
    do {
        int triangle = cuConstParams.halfedges[halfedge].y;

        float4 p = cuConstParams.tri_planes[triangle];

        // Compute the matrix product p^T p
        // Remembering that this matrix is symmetric, we 
        // only need to compute and store half of it

        q1 += p.x * p.x;
        q2 += p.x * p.y;
        q3 += p.x * p.z;
        q4 += p.x * p.w;
        
        q5 += p.y * p.y;
        q6 += p.y * p.z;
        q7 += p.y * p.w;

        q8 += p.z * p.z;
        q9 += p.z * p.w;

        // Because we normalize the planes over d, we know
        // d^2 = 1, so no need to compute / save it


        // Step to the next face
        int twin = cuConstParams.halfedges[halfedge].w;
        int next = cuConstParams.halfedges[twin].z;
        halfedge = next;

    // Repeat until we land on our starting halfedge
    } while(halfedge != base_halfedge);

    // Store the Q matrix coefficients for this vertex in global memory
    /*
    cuConstParams.Qv[vert_idx * 9 + 0] = q1;
    cuConstParams.Qv[vert_idx * 9 + 1] = q2;
    cuConstParams.Qv[vert_idx * 9 + 2] = q3;
    cuConstParams.Qv[vert_idx * 9 + 3] = q4;

    cuConstParams.Qv[vert_idx * 9 + 4] = q5;
    cuConstParams.Qv[vert_idx * 9 + 5] = q6;
    cuConstParams.Qv[vert_idx * 9 + 6] = q7;

    cuConstParams.Qv[vert_idx * 9 + 7] = q8;
    cuConstParams.Qv[vert_idx * 9 + 8] = q9;
    */
}


/* Given a mesh, set up GPU state then push the mesh onto the GPU */
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
    int* vert_halfedge_buffer;

    float4* tri_planes;
    float* Qv;

    // Allocate space for the mesh
    cudaMalloc(&vertex_buffer, sizeof(float3) * mesh.vertex_cnt);
    cudaMalloc(&halfedge_buffer, sizeof(int4) * mesh.halfedge_cnt);
    cudaMalloc(&tri_halfedge_buffer, sizeof(int) * mesh.triangle_cnt);
    cudaMalloc(&vert_halfedge_buffer, sizeof(int) * mesh.vertex_cnt);
    cudaMalloc(&tri_planes, sizeof(float4) * mesh.triangle_cnt);
    cudaMalloc(&Qv, sizeof(float) * 9 * mesh.vertex_cnt);

    // Copy the 3 x N and 4 x N arrays into vector types
    float3* verts = (float3*)calloc(mesh.vertex_cnt, sizeof(float3));
    int4* halfedge = (int4*)calloc(mesh.halfedge_cnt, sizeof(int4));
    
    int* tri_halfedges = (int*)calloc(mesh.triangle_cnt, sizeof(int));
    int* vert_halfedges = (int*)calloc(mesh.vertex_cnt, sizeof(int));

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

        
        // Set the associated halfedge for whatever vertex is the head of this halfedge
        vert_halfedges[halfedge[i].x] = i;

        // Set the associated halfedge for whatever triangle this halfedge touches
        tri_halfedges[halfedge[i].y] = i;
    }

    // Just make sure this array is populated properly
    for(int i = 0; i < mesh.triangle_cnt; i++) {
        if(tri_halfedges[i] == 0) {
            printf("Triangle %d has no associated halfedge!\n", i);
        }
    }

    // Move the vector arrays to the device
    cudaMemcpy(vertex_buffer, verts, sizeof(float3) * mesh.vertex_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(halfedge_buffer, halfedge, sizeof(int4) * mesh.halfedge_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(tri_halfedge_buffer, tri_halfedges, sizeof(int) * mesh.triangle_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(vert_halfedge_buffer, vert_halfedges, sizeof(int) * mesh.vertex_cnt, cudaMemcpyHostToDevice);

    // Free the local arrays since we don't need them anymore
    free(verts);
    free(halfedge);

    // Pass all these parameters to the GPU
    GlobalConstants params;
    params.vertex_cnt     = mesh.vertex_cnt;
    params.halfedge_cnt   = mesh.halfedge_cnt;
    params.triangle_cnt   = mesh.triangle_cnt;
    params.vertices       = vertex_buffer;
    params.halfedges      = halfedge_buffer;

    params.tri_halfedges  = tri_halfedge_buffer;
    params.vert_halfedges = vert_halfedge_buffer;
    
    params.tri_planes     = tri_planes;
    params.Qv             = Qv;

    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));
}

/* Simplify the mesh stored on the GPU */
void simplify(mesh_t mesh) {
    // Step 1.1 - Compute plane equations for all triangles
    {
        // 256 threads per block is a healthy number
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.triangle_cnt  + blockDim.x - 1) / blockDim.x);

        compute_triangle_quadrics<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
    }

    // Step 1.2 - Compute quadric matrix coefficients for all vertices
    {
        // 256 threads per block is a healthy number
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.vertex_cnt  + blockDim.x - 1) / blockDim.x);

        compute_vertex_quadrics<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
    }
}

/* Read mesh data back from the GPU and print it to stdout */
void get_results() {
    mesh_t mesh;


    printf("Getting results\n");
    
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

