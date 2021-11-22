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

    // Maximum error we tolerate for collapsing a vertex
    float error_threshold;

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

    // Array of tree edges for each vertex
    int* Vcol;
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

/* Evaluates the quadric error at position p with respect to vertex v
 * Calculated by taking (p^T)Qp */
__device__ float quadric_error(int v, float3 p) {
    int qidx = 9 * v;
    float q1 = cuConstParams.Qv[qidx];     // Q11
    float q2 = cuConstParams.Qv[qidx + 1]; // Q12, Q21
    float q3 = cuConstParams.Qv[qidx + 2]; // Q13, Q31
    float q4 = cuConstParams.Qv[qidx + 3]; // Q14, Q41
    float q5 = cuConstParams.Qv[qidx + 4]; // Q22
    float q6 = cuConstParams.Qv[qidx + 5]; // Q23, Q32
    float q7 = cuConstParams.Qv[qidx + 6]; // Q24, Q42
    float q8 = cuConstParams.Qv[qidx + 7]; // Q33
    float q9 = cuConstParams.Qv[qidx + 9]; // Q34, Q43

    // Qp
    float p1 = q1 * p.x + q2 * p.y + q3 * p.z + q4;
    float p2 = q2 * p.x + q5 * p.y + q6 * p.z + q7;
    float p3 = q3 * p.x + q6 * p.y + q8 * p.z + q8;
    float p4 = q4 * p.x + q7 * p.y + q9 * p.z + 1;

    // p^T(Qp)
    return p1 * p.x + p2 * p.y + p3 * p.z + p4;
}

/* This kernel creates embedded trees. This is done by picking
 * the minimum weight halfedge coming out of every vertex.
 * */
__global__ void build_trees() {
    int v_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_halfedge = cuConstParams.vert_halfedges[v_idx];
    float3 v = cuConstParams.vertices[v_idx];

    // Collapsing error cannot be greater than the error threshold
    float min_error = cuConstParams.error_threshold;
    int min_halfedge = -1;
    int halfedge = base_halfedge;

    do {
        int twin = cuConstParams.halfedges[halfedge].w;
        int h_idx = cuConstParams.halfedges[twin].x;
        float3 h = cuConstParams.vertices[h_idx];

        float collapsing_error =
            (quadric_error(v_idx, h) + quadric_error(h_idx, v)) / 2.f;

        // Pick the halfedge with the lowest collapsing error
        if (collapsing_error <= min_error) {
            min_error = collapsing_error;
            min_halfedge = halfedge;
        }
        halfedge = cuConstParams.halfedges[twin].z;

    } while (halfedge != base_halfedge);

    cuConstParams.Vcol[v_idx] = min_halfedge;
}


/* We verify our embedded trees to ensure that we do not have a case where
 * a halfedge and its twin are both selected by adjacent vertices. When
 * this happens we keep the halfedge for the vertex with the lower error.
 * */
__global__ void verify_trees() {
    int v_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int v_halfedge = cuConstParams.Vcol[v_idx];
    float3 v = cuConstParams.vertices[v_idx];

    int twin = cuConstParams.halfedges[v_halfedge].w;
    int h_idx = cuConstParams.halfedges[twin].x;
    int h_halfedge = cuConstParams.Vcol[h_idx];

    // Consider halfedges for which their twin has also been chosen
    if (h_halfedge != twin) return;

    float3 h = cuConstParams.vertices[h_idx];
    float h_error = quadric_error(v_idx, h);
    float v_error = quadric_error(h_idx, v);

    /* Remove the tree edge pertaining to the vertex with a
     * larger collapsing error */
    bool remove_edge =
        (v_error > h_error) || ((v_error == h_error) && (v_idx > h_idx));

    if (remove_edge) cuConstParams.Vcol[v_idx] = -1;
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
    int* Vcol;

    // Allocate space for the mesh
    cudaMalloc(&vertex_buffer, sizeof(float3) * mesh.vertex_cnt);
    cudaMalloc(&halfedge_buffer, sizeof(int4) * mesh.halfedge_cnt);
    cudaMalloc(&tri_halfedge_buffer, sizeof(int) * mesh.triangle_cnt);
    cudaMalloc(&vert_halfedge_buffer, sizeof(int) * mesh.vertex_cnt);
    cudaMalloc(&tri_planes, sizeof(float4) * mesh.triangle_cnt);
    cudaMalloc(&Qv, sizeof(float) * 9 * mesh.vertex_cnt);
    cudaMalloc(&Vcol, sizeof(int) * mesh.vertex_cnt);

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
    params.Vcol           = Vcol;

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
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.vertex_cnt  + blockDim.x - 1) / blockDim.x);

        compute_vertex_quadrics<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
    }

    // Step 2.1 - Compute embedded tree
    {
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.vertex_cnt + blockDim.x - 1) / blockDim.x);

        build_trees<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
    }

    // Step 2.2 - Verify embedded tree
    {
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.vertex_cnt + blockDim.x - 1) / blockDim.x);

        verify_trees<<<gridDim, blockDim>>>();
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

