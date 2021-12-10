#include "mesh_simplify.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "helper_math.h"
#include "CycleTimer.h"

// Accessor macros for the fields of a halfedge
#define VERT(e) (cuConstParams.halfedges[e].x)
#define TRI(e) (cuConstParams.halfedges[e].y)
#define NEXT(e) (cuConstParams.halfedges[e].z)
#define TWIN(e) (cuConstParams.halfedges[e].w)

#define ERROR_THRESHOLD 1.f
#define MAX_BLOCK_SIZE 1024
#define SCAN_BLOCK_DIM MAX_BLOCK_SIZE

#include "exclusiveScan.cu_inl"

struct GlobalConstants {
    int vertex_cnt;
    int halfedge_cnt;
    int triangle_cnt;

    int *new_halfedge_cnt;
    int *new_vertex_cnt;

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

    // Halfedge collapse lookup table
    int* Meg;

    // Vertex collapse lookup table
    int* Veg;
};

__constant__ GlobalConstants cuConstParams;


/* Computes the plane equations for each of the triangles in the mesh, and stores them
 * in global memory so that we can reuse the values to compute the Q matrices for
 * all vertices
 */
__global__ void compute_triangle_quadrics() {

    // Get the triangle we are operating over
    int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tri_idx >= cuConstParams.triangle_cnt) return;

    // Get the half_edges associated with this triangle
    int half_edge1 = cuConstParams.tri_halfedges[tri_idx];
    int half_edge2 = NEXT(half_edge1);
    int half_edge3 = NEXT(half_edge2);

    // Get the vertices of this halfedge
    int v1 = VERT(half_edge1);
    int v2 = VERT(half_edge2);
    int v3 = VERT(half_edge3);

    // Get the vertex positions
    float3 p1 = cuConstParams.vertices[v1];
    float3 p2 = cuConstParams.vertices[v2];
    float3 p3 = cuConstParams.vertices[v3];


    // Build plane equation
    float3 side1 = p2 - p1;
    float3 side2 = p1 - p3;

    float3 normal = normalize(cross(side1, side2));
    float d = -dot(normal, p1);

    cuConstParams.tri_planes[tri_idx].x = normal.x;
    cuConstParams.tri_planes[tri_idx].y = normal.y;
    cuConstParams.tri_planes[tri_idx].z = normal.z;
    cuConstParams.tri_planes[tri_idx].w = d;
}


/* Computes the Q matrix coefficients for all vertices in the mesh
 * using the previously computed plane equations
 */
__global__ void compute_vertex_quadrics() {

    // Get the triangle we are operating over
    int vert_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(vert_idx >= cuConstParams.vertex_cnt) return;

    // Get the halfedge of this vertex
    int base_halfedge = cuConstParams.vert_halfedges[vert_idx];

    // Initialize the 9 coefficients of the Q matrix to zero
    float q1  = 0.f;
    float q2  = 0.f;
    float q3  = 0.f;
    float q4  = 0.f;
    float q5  = 0.f;
    float q6  = 0.f;
    float q7  = 0.f;
    float q8  = 0.f;
    float q9  = 0.f;
    float q10 = 0.f;

    // Loop around all faces incident on this vertex
    int halfedge = base_halfedge;
    do {
        int triangle = TRI(halfedge);

        float4 p = cuConstParams.tri_planes[triangle];

        // Compute the matrix product p^T p
        // Remembering that this matrix is symmetric, we
        // only need to compute and store half of it

        q1  += p.x * p.x;
        q2  += p.x * p.y;
        q3  += p.x * p.z;
        q4  += p.x * p.w;
 
        q5  += p.y * p.y;
        q6  += p.y * p.z;
        q7  += p.y * p.w;
 
        q8  += p.z * p.z;
        q9  += p.z * p.w;

        q10 += p.w * p.w;

        // Because we normalize the planes over d, we know
        // d^2 = 1, so no need to compute / save it

        // Step to the next face
        int twin = TWIN(halfedge);

        if(twin < 0) break;

        int next = NEXT(twin);
        halfedge = next;

    // Repeat until we land on our starting halfedge or a boundary
    } while(halfedge == base_halfedge);

    // Store the Q matrix coefficients for this vertex in global memory

    cuConstParams.Qv[vert_idx * 10 + 0] = q1;
    cuConstParams.Qv[vert_idx * 10 + 1] = q2;
    cuConstParams.Qv[vert_idx * 10 + 2] = q3;
    cuConstParams.Qv[vert_idx * 10 + 3] = q4;

    cuConstParams.Qv[vert_idx * 10 + 4] = q5;
    cuConstParams.Qv[vert_idx * 10 + 5] = q6;
    cuConstParams.Qv[vert_idx * 10 + 6] = q7;

    cuConstParams.Qv[vert_idx * 10 + 7] = q8;
    cuConstParams.Qv[vert_idx * 10 + 8] = q9;

    cuConstParams.Qv[vert_idx * 10 + 9] = q10;
}

/* Evaluates the quadric error at position p with respect to vertex v
 * Calculated by taking (p^T)Qp */
__device__ float quadric_error(int v1, int v2, float3 p) {
    int q1idx = 10 * v1;
    int q2idx = 10 * v2;

    float q1  = cuConstParams.Qv[q1idx];     // Q11
    float q2  = cuConstParams.Qv[q1idx + 1]; // Q12, Q21
    float q3  = cuConstParams.Qv[q1idx + 2]; // Q13, Q31
    float q4  = cuConstParams.Qv[q1idx + 3]; // Q14, Q41
    float q5  = cuConstParams.Qv[q1idx + 4]; // Q22
    float q6  = cuConstParams.Qv[q1idx + 5]; // Q23, Q32
    float q7  = cuConstParams.Qv[q1idx + 6]; // Q24, Q42
    float q8  = cuConstParams.Qv[q1idx + 7]; // Q33
    float q9  = cuConstParams.Qv[q1idx + 8]; // Q34, Q43
    float q10 = cuConstParams.Qv[q1idx + 9]; // Q44

    q1  += cuConstParams.Qv[q2idx];     // Q11
    q2  += cuConstParams.Qv[q2idx + 1]; // Q12, Q21
    q3  += cuConstParams.Qv[q2idx + 2]; // Q13, Q31
    q4  += cuConstParams.Qv[q2idx + 3]; // Q14, Q41
    q5  += cuConstParams.Qv[q2idx + 4]; // Q22
    q6  += cuConstParams.Qv[q2idx + 5]; // Q23, Q32
    q7  += cuConstParams.Qv[q2idx + 6]; // Q24, Q42
    q8  += cuConstParams.Qv[q2idx + 7]; // Q33
    q9  += cuConstParams.Qv[q2idx + 8]; // Q34, Q43
    q10 += cuConstParams.Qv[q2idx + 9]; // Q44

    // Average the two Q matrices
    q1  /= 2.0f;
    q2  /= 2.0f;
    q3  /= 2.0f;
    q4  /= 2.0f;
    q5  /= 2.0f;
    q6  /= 2.0f;
    q7  /= 2.0f;
    q8  /= 2.0f;
    q9  /= 2.0f;
    q10 /= 2.0f;

    // Qp
    float p1 = q1 * p.x + q2 * p.y + q3 * p.z + q4;
    float p2 = q2 * p.x + q5 * p.y + q6 * p.z + q7;
    float p3 = q3 * p.x + q6 * p.y + q8 * p.z + q9;
    float p4 = q4 * p.x + q7 * p.y + q9 * p.z + q10;

    // p^T(Qp)
    return p1 * p.x + p2 * p.y + p3 * p.z + p4;
}

/* This kernel creates embedded trees. This is done by picking
 * the minimum weight halfedge coming out of every vertex.
 * */
__global__ void build_trees() {
    int vert_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(vert_idx >= cuConstParams.vertex_cnt) return;

    int base_halfedge = cuConstParams.vert_halfedges[vert_idx];
    float3 v = cuConstParams.vertices[vert_idx];

    // Collapsing error cannot be greater than the error threshold
    float min_error = cuConstParams.error_threshold;
    int halfedge = base_halfedge;
    int min_halfedge = -1;

    do {
        int twin = TWIN(halfedge);

        // Ignore any vertices on boundaries (they can potentially cause cycles)
        if(twin < 0) {
            min_halfedge = -1;
            break;
        }

        int h_idx = VERT(NEXT(halfedge));
        float3 h = cuConstParams.vertices[h_idx];

        // Find the midpoint between the two verts
        float3 v_bar = (v + h) / 2.f;

        float collapsing_error = quadric_error(vert_idx, h_idx, v_bar);

        // Pick the halfedge with the lowest collapsing error
        if (collapsing_error < min_error) {
            min_error = collapsing_error;
            min_halfedge = halfedge;
        }
        // If two edges have equal weight, give up for now to avoid cycles
        else if(collapsing_error <= min_error)
        {
            min_halfedge = -1;
            break;
        }   

        halfedge = NEXT(twin);

    } while (halfedge != base_halfedge);

    cuConstParams.Vcol[vert_idx] = min_halfedge;
}


/* We verify our embedded trees to ensure that we do not have a case where
 * a halfedge and its twin are both selected by adjacent vertices. When
 * this happens we keep the halfedge for the vertex with the lower error.
 * */
__global__ void verify_trees() {
    int vert_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(vert_idx >= cuConstParams.vertex_cnt) return;

    int v_halfedge = cuConstParams.Vcol[vert_idx];
    float3 v = cuConstParams.vertices[vert_idx];

    if (v_halfedge < 0) return;

    int twin = TWIN(v_halfedge);
    int h_idx = VERT(NEXT(v_halfedge));
    int h_halfedge = cuConstParams.Vcol[h_idx];

    // Consider halfedges for which their twin has also been chosen
    if (h_halfedge != twin) return;

    
    float3 h = cuConstParams.vertices[h_idx];
    float h_error = quadric_error(vert_idx, h_idx, h);
    float v_error = quadric_error(vert_idx, h_idx, v);

    /* Remove the tree edge pertaining to the vertex with a
     * larger collapsing error */
    bool remove_edge =
        (v_error > h_error) || ((v_error == h_error) && (vert_idx > h_idx));

    if (remove_edge) cuConstParams.Vcol[vert_idx] = -1;
}

/* Go through Vcol and compute the replacement mapping for edges being removed
 * from the mesh
 * */
__global__ void collapse_edges() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= cuConstParams.vertex_cnt) return;

    int halfedge = cuConstParams.Vcol[idx];

    // Ignore edges which have already been collapsed
    //if(cuConstParams.Meg[halfedge] != halfedge) return;

    // Ignore vertices with no collapsed edges
    if(halfedge < 0) return;

    int twin = TWIN(halfedge);

    int e1 = NEXT(halfedge);
    int e2 = NEXT(e1);

    // Mark this edge as contracted
    cuConstParams.Meg[halfedge] = -1;
    // And stitch the two adjacent twins together
    cuConstParams.Meg[e1] = TWIN(e2);
    cuConstParams.Meg[e2] = TWIN(e1);

    // If we aren't on a boundary, contract this edge too
    if(twin >= 0) {
        int e3 = NEXT(twin);
        int e4 = NEXT(e3);

        // Mark this edge as contracted
        cuConstParams.Meg[twin] = -1;
        // And stitch the two adjacent twins together
        cuConstParams.Meg[e3] = TWIN(e4);
        cuConstParams.Meg[e4] = TWIN(e3);
    }

    // List the vertex as collapsed into the tail
    int head = idx;
    int tail = VERT(NEXT(halfedge));
    cuConstParams.Veg[idx] = tail;
}

/* Compress the halfedge replacement map to all final values
 * */
__global__ void compress_meg() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= cuConstParams.halfedge_cnt) return;

    // Start at our current halfedge
    int x = idx;

    // Traverse the list until we reach an uncollapsed halfedge
    while(x >= 0 && x != cuConstParams.Meg[x]) {
        x = cuConstParams.Meg[x];
    }

    // Point the initial halfedge to this result
    cuConstParams.Meg[idx] = x;
}

/* Compress the vertex replacement map to all final values
 * */
__global__ void compress_veg() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= cuConstParams.vertex_cnt) return;

    // Start at our current halfedge
    int x = idx;
    int cnt = 0;

    // Traverse the list until we reach an uncollapsed halfedge
    while(x >= 0 && x != cuConstParams.Veg[x] && cnt < 100) {
        x = cuConstParams.Veg[x];
        cnt++;
    }

    // Point the initial halfedge to this result
    cuConstParams.Veg[idx] = x;
}


/* Apply the halfedge and vertex replacement mappings to the mesh
 * */
__global__ void update_halfedges() {
    int halfedge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(halfedge_idx >= cuConstParams.halfedge_cnt) return;

    int twin = TWIN(halfedge_idx);
    int next = NEXT(halfedge_idx);

    // If halfedge is non-null, look up what replaces it
    twin = twin < 0 ? -1 : cuConstParams.Meg[twin];
    next = next < 0 ? -1 : cuConstParams.Meg[next];

    // Remove collapsed vertex
    cuConstParams.halfedges[halfedge_idx].z = next;
    cuConstParams.halfedges[halfedge_idx].w = twin;
    
    int vert = VERT(halfedge_idx);
    vert = cuConstParams.Veg[vert];

    cuConstParams.halfedges[halfedge_idx].x = vert;
}


/* Use exclusive scan to remove all deleted halfedges from the mesh
 * */
__global__ void relabel_halfedges() {
    __shared__ uint flags[MAX_BLOCK_SIZE];
    __shared__ uint running[MAX_BLOCK_SIZE];
    __shared__ uint scratch[2 * MAX_BLOCK_SIZE];
    
    int base = 0;
    int halfedge_cnt = cuConstParams.halfedge_cnt;

    // Iterate over all halfedges in the mesh until all have been remapped
    int iters = (cuConstParams.halfedge_cnt + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    for(int i = 0; i < iters; i++) {
        int halfedge_idx = i * MAX_BLOCK_SIZE + threadIdx.x;

        // Mark this halfedge only if no other 
        if(halfedge_idx < halfedge_cnt) {
            flags[threadIdx.x] = halfedge_idx == cuConstParams.Meg[halfedge_idx] ? 1 : 0;
        }
        else {
            flags[threadIdx.x] = 0;
        }
        __syncthreads();

        // Count up the index for each active halfedge here
        sharedMemExclusiveScan(threadIdx.x, flags, running, scratch, MAX_BLOCK_SIZE);

        __syncthreads();
        int new_idx = base + running[threadIdx.x];

        // The number of halfedges we have accumulated so far
        base += running[MAX_BLOCK_SIZE - 1] + flags[MAX_BLOCK_SIZE - 1];

        cuConstParams.Meg[halfedge_idx] = new_idx;

        int4 he;

        // Grab the halfedge we are moving
        if(flags[threadIdx.x] > 0 && halfedge_idx < halfedge_cnt)
            he = cuConstParams.halfedges[halfedge_idx];
        
        // If this halfedge should be kept, move it to its new position
        if(flags[threadIdx.x] > 0 && halfedge_idx < halfedge_cnt)
            cuConstParams.halfedges[new_idx] = he;


        __syncthreads();
    }

    __syncthreads();

    // Update the halfedge count
    if(threadIdx.x == 0) 
        *(cuConstParams.new_halfedge_cnt) = base;
}


/* Use exclusive scan to remove all deleted vertices from the mesh
 * */
__global__ void relabel_vertices() {
    __shared__ uint flags[MAX_BLOCK_SIZE];
    __shared__ uint running[MAX_BLOCK_SIZE];
    __shared__ uint scratch[2 * MAX_BLOCK_SIZE];
    
    int base = 0;
    int vertex_cnt = cuConstParams.vertex_cnt;

    // Iterate over all vertices in the mesh until all have been remapped
    int iters = (cuConstParams.vertex_cnt + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    for(int i = 0; i < iters; i++) {
        int vertex_idx = i * MAX_BLOCK_SIZE + threadIdx.x;

        // Mark this vertex only if no other 
        if(vertex_idx < vertex_cnt) {
            flags[threadIdx.x] = vertex_idx == cuConstParams.Veg[vertex_idx] ? 1 : 0;
        }
        else {
            flags[threadIdx.x] = 0;
        }
        __syncthreads();

        // Count up the index for each active vertex here
        sharedMemExclusiveScan(threadIdx.x, flags, running, scratch, MAX_BLOCK_SIZE);

        __syncthreads();
        int new_idx = base + running[threadIdx.x];

        cuConstParams.Veg[vertex_idx] = new_idx;

        // If this vertex should be kept, move it to its new position
        if(flags[threadIdx.x] > 0 && vertex_idx < vertex_cnt)
            cuConstParams.vertices[new_idx] = cuConstParams.vertices[vertex_idx];

        // The number of vertices we have accumulated so far
        base += running[MAX_BLOCK_SIZE - 1] + flags[MAX_BLOCK_SIZE - 1];

        __syncthreads();
    }

    __syncthreads();

    // Update the vertex count
    if(threadIdx.x == 0) 
        *(cuConstParams.new_vertex_cnt) = base;
}



/* We refine vertex positions by solving the system of equations
 * Q3v = l3 where Q3 is the top-left 3x3 submatrix of Q[v] and
 * l3 is a vector made of the first 3 entries of 4th column of Q[v].
 * We solve the linear equation using the conjugate gradient method. 
 * Code is based on: http://www.cplusplus.com/forum/general/222617/*/
__global__ void refine_vertices() {
    int v_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (v_idx >= cuConstParams.vertex_cnt) return;

    // Ignore vertices that have been collapsed
    if(cuConstParams.Veg[v_idx] != v_idx) return;

    int qidx = 10 * v_idx;
    float q1 = cuConstParams.Qv[qidx];     // Q11
    float q2 = cuConstParams.Qv[qidx + 1]; // Q12, Q21
    float q3 = cuConstParams.Qv[qidx + 2]; // Q13, Q31
    float q4 = cuConstParams.Qv[qidx + 4]; // Q22
    float q5 = cuConstParams.Qv[qidx + 5]; // Q23, Q32
    float q6 = cuConstParams.Qv[qidx + 7]; // Q33

    float3 l;
    l.x = -cuConstParams.Qv[qidx + 3];     // Q14
    l.y = -cuConstParams.Qv[qidx + 6];     // Q24
    l.z = -cuConstParams.Qv[qidx + 8];     // Q34

    float3 row1 = make_float3(q1, q2, q3);
    float3 row2 = make_float3(q2, q4, q5);
    float3 row3 = make_float3(q3, q5, q6);

    float tolerance = 1.0e-5;
    float near_zero = 1.0e-5;
    int n = 11;
    float3 newv = make_float3(0.f, 0.f, 0.f);
    float3 r = l;
    float3 p = r;

    // We solve our system iteratively with conjugate gradient descent
    for (int k = 0; k < n; k++) {
        float3 rold = r;
        float3 ap = make_float3(dot(row1, p), dot(row2, p), dot(row3, p));

        float alpha = dot(r, r) / fmaxf(dot(p, ap), near_zero);
        newv = newv + alpha * p;
        r = r - alpha * ap;

        if (length(r) < tolerance) break;

        float beta = dot(r, r) / fmaxf(dot(rold, rold), near_zero);
        p = r + beta * p;
    }

    // Only make change if refinement is below error threshold
    if (length(cuConstParams.vertices[v_idx] - newv) <= .001f)
        cuConstParams.vertices[v_idx] = newv;
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
    int* Meg;
    int* Veg;

    int* new_halfedge_cnt;
    int* new_vertex_cnt;

    // Allocate space for the mesh
    cudaMalloc(&vertex_buffer, sizeof(float3) * mesh.vertex_cnt);
    cudaMalloc(&halfedge_buffer, sizeof(int4) * mesh.halfedge_cnt);
    cudaMalloc(&tri_halfedge_buffer, sizeof(int) * mesh.triangle_cnt);
    cudaMalloc(&vert_halfedge_buffer, sizeof(int) * mesh.vertex_cnt);
    cudaMalloc(&tri_planes, sizeof(float4) * mesh.triangle_cnt);
    cudaMalloc(&Qv, sizeof(float) * 10 * mesh.vertex_cnt);
    cudaMalloc(&Vcol, sizeof(int) * mesh.vertex_cnt);
    cudaMalloc(&Meg, sizeof(int) * mesh.halfedge_cnt);
    cudaMalloc(&Veg, sizeof(int) * mesh.vertex_cnt);
    cudaMalloc(&new_halfedge_cnt, sizeof(int));
    cudaMalloc(&new_vertex_cnt, sizeof(int));

    // Copy the 3 x N and 4 x N arrays into vector types
    float3* verts = (float3*)calloc(mesh.vertex_cnt, sizeof(float3));
    int4* halfedge = (int4*)calloc(mesh.halfedge_cnt, sizeof(int4));

    int* tri_halfedges = (int*)calloc(mesh.triangle_cnt, sizeof(int));
    int* vert_halfedges = (int*)calloc(mesh.halfedge_cnt, sizeof(int));

    int* Meg_init = (int*)calloc(mesh.halfedge_cnt, sizeof(int));
    int* Veg_init = (int*)calloc(mesh.halfedge_cnt, sizeof(int));

    for(int i = 0; i < mesh.vertex_cnt; i++) {
        verts[i].x = mesh.vertices[i * 3 + 0];
        verts[i].y = mesh.vertices[i * 3 + 1];
        verts[i].z = mesh.vertices[i * 3 + 2];

        Veg_init[i] = i;
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

        // Set the edge collapse map to be the identity function
        Meg_init[i] = i;
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
    cudaMemcpy(Meg, Meg_init, sizeof(int) * mesh.halfedge_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(Veg, Veg_init, sizeof(int) * mesh.vertex_cnt, cudaMemcpyHostToDevice);

    // Free the local arrays since we don't need them anymore
    free(verts);
    free(halfedge);

    // Pass all these parameters to the GPU
    GlobalConstants params;
    params.vertex_cnt     = mesh.vertex_cnt;
    params.halfedge_cnt   = mesh.halfedge_cnt;
    params.triangle_cnt   = mesh.triangle_cnt;

    params.new_halfedge_cnt = new_halfedge_cnt;
    params.new_vertex_cnt   = new_vertex_cnt;

    params.vertices       = vertex_buffer;
    params.halfedges      = halfedge_buffer;

    params.tri_halfedges  = tri_halfedge_buffer;
    params.vert_halfedges = vert_halfedge_buffer;

    params.tri_planes     = tri_planes;
    params.Qv             = Qv;
    params.Vcol           = Vcol;

    params.Meg            = Meg;
    params.Veg            = Veg;
    params.error_threshold = ERROR_THRESHOLD;

    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));
}

/* Simplify the mesh stored on the GPU */
void simplify(mesh_t mesh) {
    double total_time = 0.f;

    // Step 1.1 - Compute plane equations for all triangles
    {
        double startTime = CycleTimer::currentSeconds();
        
        // 256 threads per block is a healthy number
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.triangle_cnt  + blockDim.x - 1) / blockDim.x);

        compute_triangle_quadrics<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        
        double endTime = CycleTimer::currentSeconds();
        total_time += endTime - startTime;
        printf("Compute plane equations:        %.03fms\n", (endTime - startTime) * 1000);
    }

    // Step 1.2 - Compute quadric matrix coefficients for all vertices
    {
        double startTime = CycleTimer::currentSeconds();
        
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.vertex_cnt  + blockDim.x - 1) / blockDim.x);

        compute_vertex_quadrics<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        
        double endTime = CycleTimer::currentSeconds();
        total_time += endTime - startTime;
        printf("Compute vertex quadrics:        %.03fms\n", (endTime - startTime) * 1000);
    }

    // Step 2.1 - Compute embedded tree
    {
        double startTime = CycleTimer::currentSeconds();
        
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.vertex_cnt + blockDim.x - 1) / blockDim.x);

        build_trees<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        
        double endTime = CycleTimer::currentSeconds();
        total_time += endTime - startTime;
        printf("Compute embedded trees:         %.03fms\n", (endTime - startTime) * 1000);
    }

    // Step 2.2 - Verify embedded tree
    {
        double startTime = CycleTimer::currentSeconds();
        
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.vertex_cnt + blockDim.x - 1) / blockDim.x);

        verify_trees<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        
        double endTime = CycleTimer::currentSeconds();
        total_time += endTime - startTime;
        printf("Verify embedded trees:          %.03fms\n", (endTime - startTime) * 1000);
    }
    
    // Step 5.1 - Apply halfedge collapses and store values into the Meg array
    {
        double startTime = CycleTimer::currentSeconds();
        
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.vertex_cnt  + blockDim.x - 1) / blockDim.x);

        collapse_edges<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        
        double endTime = CycleTimer::currentSeconds();
        total_time += endTime - startTime;
        printf("Apply halfedge collapses:       %.03fms\n", (endTime - startTime) * 1000);
    }

    // Step 5.1 - Apply halfedge collapses and store values into the Meg array
    {
        double startTime = CycleTimer::currentSeconds();
        
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.halfedge_cnt  + blockDim.x - 1) / blockDim.x);

        compress_meg<<<gridDim, blockDim>>>();
        compress_veg<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        
        double endTime = CycleTimer::currentSeconds();
        total_time += endTime - startTime;
        printf("Compress halfedge/vertex maps:  %.03fms\n", (endTime - startTime) * 1000);
    }
    
    // Step 5.2 - Push updates from Meg into the halfedge array
    {
        double startTime = CycleTimer::currentSeconds();
        
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.halfedge_cnt  + blockDim.x - 1) / blockDim.x);

        update_halfedges<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        
        double endTime = CycleTimer::currentSeconds();
        total_time += endTime - startTime;
        printf("Apply halfedge map:             %.03fms\n", (endTime - startTime) * 1000);
    }

    // Step 6 - Refine vertex positions
    {
        double startTime = CycleTimer::currentSeconds();
        
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.vertex_cnt + blockDim.x - 1) / blockDim.x);

        refine_vertices<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        
        double endTime = CycleTimer::currentSeconds();
        total_time += endTime - startTime;
        printf("Refine vertex positions:        %.03fms\n", (endTime - startTime) * 1000);
    }

    // Step 7.1 - Make the mesh contiguous again (for sake of efficiency)
    {
        double startTime = CycleTimer::currentSeconds();
        
        int box_size = MAX_BLOCK_SIZE;
        dim3 blockDim(box_size);
        dim3 gridDim(1);

        relabel_halfedges<<<gridDim, blockDim>>>();
        relabel_vertices<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        
        double endTime = CycleTimer::currentSeconds();
        total_time += endTime - startTime;
        printf("Relabel halfedges and vertices: %.03fms\n", (endTime - startTime) * 1000);
    }
    
    // Step 7.2 - Push updates from the contiguous Meg into the halfedge array
    {
        double startTime = CycleTimer::currentSeconds();
        
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.halfedge_cnt  + blockDim.x - 1) / blockDim.x);

        update_halfedges<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        
        double endTime = CycleTimer::currentSeconds();
        total_time += endTime - startTime;
        printf("Apply halfedge map:             %.03fms\n", (endTime - startTime) * 1000);
    }

    printf("\nTotal time: %.03fms\n\n", total_time * 1000);
}

/* Read mesh data back from the GPU and print it to stdout */
mesh_t get_results() {
    mesh_t mesh;


    printf("Getting results\n");

    // Pass all these parameters to the GPU
    GlobalConstants params;
    cudaMemcpyFromSymbol(&params, cuConstParams, sizeof(GlobalConstants));

    mesh.vertex_cnt = params.vertex_cnt;
    mesh.halfedge_cnt = params.halfedge_cnt;
    mesh.triangle_cnt = params.triangle_cnt;

    cudaMemcpy(&mesh.halfedge_cnt, params.new_halfedge_cnt, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&mesh.vertex_cnt, params.new_vertex_cnt, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Verts: %d\tHalfedges: %d\tTriangles: %d\n",
            mesh.vertex_cnt, mesh.halfedge_cnt, mesh.triangle_cnt);

    // Copy the 3 x N and 4 x N arrays into vector types
    float3* verts = (float3*)calloc(mesh.vertex_cnt, sizeof(float3));
    int4* halfedges = (int4*)calloc(mesh.halfedge_cnt, sizeof(int4));

    // Copy the data back from the GPU
    cudaMemcpy(verts, params.vertices, sizeof(float3) * mesh.vertex_cnt, cudaMemcpyDeviceToHost);
    cudaMemcpy(halfedges, params.halfedges, sizeof(int4) * mesh.halfedge_cnt, cudaMemcpyDeviceToHost);

    mesh.vertices = (float*)calloc(mesh.vertex_cnt * 3, sizeof(float));
    mesh.halfedges = (int*)calloc(mesh.halfedge_cnt * 4, sizeof(int));

    // Copy vertices
    for(int i = 0; i < mesh.vertex_cnt; i++) {
        mesh.vertices[3 * i + 0] = verts[i].x;
        mesh.vertices[3 * i + 1] = verts[i].y;
        mesh.vertices[3 * i + 2] = verts[i].z;
    }

    // Copy halfedges
    for(int i = 0; i < mesh.halfedge_cnt; i++) {
        mesh.halfedges[4 * i + 0] = halfedges[i].x;
        mesh.halfedges[4 * i + 1] = halfedges[i].y;
        mesh.halfedges[4 * i + 2] = halfedges[i].z;
        mesh.halfedges[4 * i + 3] = halfedges[i].w;
    }


    return mesh;
}

