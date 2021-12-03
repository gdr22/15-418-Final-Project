#include "mesh_simplify.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "helper_math.h"

// Accessor macros for the fields of a halfedge
#define VERT(e) (cuConstParams.halfedges[e].x)
#define TRI(e) (cuConstParams.halfedges[e].y)
#define NEXT(e) (cuConstParams.halfedges[e].z)
#define TWIN(e) (cuConstParams.halfedges[e].w)

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

    // Edges to collapse
    int* Ecol;
    int collapse_cnt;

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
    if(vert_idx >= cuConstParams.vertex_cnt) return;

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
        int triangle = TRI(halfedge);

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
        int twin = TWIN(halfedge);

        if(twin == -1) break;

        int next = NEXT(twin);
        halfedge = next;

    // Repeat until we land on our starting halfedge or a boundary
    } while(halfedge == base_halfedge);

    // Store the Q matrix coefficients for this vertex in global memory

    cuConstParams.Qv[vert_idx * 9 + 0] = q1;
    cuConstParams.Qv[vert_idx * 9 + 1] = q2;
    cuConstParams.Qv[vert_idx * 9 + 2] = q3;
    cuConstParams.Qv[vert_idx * 9 + 3] = q4;

    cuConstParams.Qv[vert_idx * 9 + 4] = q5;
    cuConstParams.Qv[vert_idx * 9 + 5] = q6;
    cuConstParams.Qv[vert_idx * 9 + 6] = q7;

    cuConstParams.Qv[vert_idx * 9 + 7] = q8;
    cuConstParams.Qv[vert_idx * 9 + 8] = q9;
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
    float q9 = cuConstParams.Qv[qidx + 8]; // Q34, Q43

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
    int vert_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(vert_idx >= cuConstParams.vertex_cnt) return;

    int base_halfedge = cuConstParams.vert_halfedges[vert_idx];
    float3 v = cuConstParams.vertices[vert_idx];

    // Collapsing error cannot be greater than the error threshold
    float min_error = cuConstParams.error_threshold;
    int min_halfedge = -1;
    int halfedge = base_halfedge;

    do {
        int twin = TWIN(halfedge);

        if(twin == -1) break;

        int h_idx = VERT(twin);
        float3 h = cuConstParams.vertices[h_idx];

        float collapsing_error =
            (quadric_error(vert_idx, h) + quadric_error(h_idx, v)) / 2.f;

        // Pick the halfedge with the lowest collapsing error
        if (collapsing_error <= min_error) {
            min_error = collapsing_error;
            min_halfedge = halfedge;
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

    if (v_halfedge == -1) return;

    int twin = cuConstParams.halfedges[v_halfedge].w;
    int h_idx = cuConstParams.halfedges[twin].x;
    int h_halfedge = cuConstParams.Vcol[h_idx];

    // Consider halfedges for which their twin has also been chosen
    if (h_halfedge != twin) return;

    float3 h = cuConstParams.vertices[h_idx];
    float h_error = quadric_error(vert_idx, h);
    float v_error = quadric_error(h_idx, v);

    /* Remove the tree edge pertaining to the vertex with a
     * larger collapsing error */
    bool remove_edge =
        (v_error > h_error) || ((v_error == h_error) && (vert_idx > h_idx));

    if (remove_edge) cuConstParams.Vcol[vert_idx] = -1;
}

/* Go through Ecol and compute the replacement mapping for edges being removed
 * from the mesh
 * */
__global__ void collapse_edges() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= cuConstParams.halfedge_cnt) return;

    if(idx % 100 != 1) return;

    int halfedge = idx;
    int twin = TWIN(halfedge);

    int e1 = NEXT(halfedge);
    int e2 = NEXT(e1);

    // Mark this edge as contracted
    cuConstParams.Meg[halfedge] = -1;
    // And stitch the two adjacent twins together
    cuConstParams.Meg[e1] = TWIN(e2);
    cuConstParams.Meg[e2] = TWIN(e1);

    // If we aren't on a boundary, contract this edge too
    if(twin != -1) {
        int e3 = NEXT(twin);
        int e4 = NEXT(e3);

        // Mark this edge as contracted
        cuConstParams.Meg[twin] = -1;
        // And stitch the two adjacent twins together
        cuConstParams.Meg[e3] = TWIN(e4);
        cuConstParams.Meg[e4] = TWIN(e3);
    }

    // List the tail vertex as collapsed into the head
    int tail = VERT(NEXT(halfedge));
    cuConstParams.Veg[tail] = VERT(halfedge);
}

/* Apply the halfedge and vertex replacement mappings to the mesh
 * */
__global__ void update_halfedges() {
    int halfedge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(halfedge_idx >= cuConstParams.halfedge_cnt) return;

    int twin = TWIN(halfedge_idx);

    // If halfedge is non-null, look up what it collapses to
    twin = twin == -1 ? -1 : cuConstParams.Meg[twin];

    // Remove collapsed vertex
    cuConstParams.halfedges[halfedge_idx].w = twin;

    int vert = VERT(halfedge_idx);
    vert = cuConstParams.Veg[vert];

    cuConstParams.halfedges[halfedge_idx].x = vert;
}

/* We refine vertex positions by solving the system of equations
 * Q3v = l3 where Q3 is the top-left 3x3 submatrix of Q[v] and
 * l3 is a vector made of the first 3 entries of 4th column of Q[v]*/
__global__ void refine_vertices() {
    int vidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vidx >= cuConstParams.vertex_cnt) return;

    float3 l;

    int qidx = 9 * vidx;
    float q1 = cuConstParams.Qv[qidx];     // Q11
    float q2 = cuConstParams.Qv[qidx + 1]; // Q12, Q21
    float q3 = cuConstParams.Qv[qidx + 2]; // Q13, Q31
    l.x = -cuConstParams.Qv[qidx + 3];     // Q14
    float q4 = cuConstParams.Qv[qidx + 4]; // Q22
    float q5 = cuConstParams.Qv[qidx + 5]; // Q23, Q32
    l.y = -cuConstParams.Qv[qidx + 6];     // Q24
    float q6 = cuConstParams.Qv[qidx + 7]; // Q33
    l.z = -cuConstParams.Qv[qidx + 8];     // Q34

    float3 row1 = make_float3(q1, q2, q3);
    float3 row2 = make_float3(q2, q4, q5);
    float3 row3 = make_float3(q3, q5, q6);

    float tolerance = 1.0e-5;
    float near_zero = 1.0e-5;
    int n = 3;
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
    if (length(cuConstParams.vertices[vidx] - newv) <= cuConstParams.error_threshold)
        cuConstParams.vertices[vidx] = newv;
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

    // Allocate space for the mesh
    cudaMalloc(&vertex_buffer, sizeof(float3) * mesh.vertex_cnt);
    cudaMalloc(&halfedge_buffer, sizeof(int4) * mesh.halfedge_cnt);
    cudaMalloc(&tri_halfedge_buffer, sizeof(int) * mesh.triangle_cnt);
    cudaMalloc(&vert_halfedge_buffer, sizeof(int) * mesh.vertex_cnt);
    cudaMalloc(&tri_planes, sizeof(float4) * mesh.triangle_cnt);
    cudaMalloc(&Qv, sizeof(float) * 9 * mesh.vertex_cnt);
    cudaMalloc(&Vcol, sizeof(int) * mesh.vertex_cnt);
    cudaMalloc(&Meg, sizeof(int) * mesh.halfedge_cnt);
    cudaMalloc(&Veg, sizeof(int) * mesh.vertex_cnt);

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
    params.vertices       = vertex_buffer;
    params.halfedges      = halfedge_buffer;

    params.tri_halfedges  = tri_halfedge_buffer;
    params.vert_halfedges = vert_halfedge_buffer;

    params.tri_planes     = tri_planes;
    params.Qv             = Qv;
    params.Vcol           = Vcol;

    params.Meg            = Meg;
    params.Veg            = Veg;



    // Manually add edges to test
    params.collapse_cnt = mesh.halfedge_cnt;









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


    /*
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
    */

    // Step 5.1 - Apply halfedge collapses and store values into the Meg array
    {
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.halfedge_cnt  + blockDim.x - 1) / blockDim.x);

        collapse_edges<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
    }

    // Step 5.2 - Push updates from Meg into the halfedge array
    {
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.halfedge_cnt  + blockDim.x - 1) / blockDim.x);

        update_halfedges<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
    }

    // Step 6 - Refine vertex positions
    {
        int box_size = 256;
        dim3 blockDim(box_size);
        dim3 gridDim((mesh.vertex_cnt + blockDim.x - 1) / blockDim.x);

        refine_vertices<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
    }

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

    printf("Verts: %d\tHalfedges: %d\tTriangles: %d\n",
            mesh.vertex_cnt, mesh.halfedge_cnt, mesh.triangle_cnt);

    // Copy the 3 x N and 4 x N arrays into vector types
    float3* verts = (float3*)calloc(mesh.vertex_cnt, sizeof(float3));
    int4* halfedges = (int4*)calloc(mesh.halfedge_cnt, sizeof(int4));
    float* Qv = (float*)calloc(mesh.vertex_cnt * 9, sizeof(float));

    // Copy the data back from the GPU
    cudaMemcpy(verts, params.vertices, sizeof(float3) * mesh.vertex_cnt, cudaMemcpyDeviceToHost);
    cudaMemcpy(halfedges, params.halfedges, sizeof(int4) * mesh.halfedge_cnt, cudaMemcpyDeviceToHost);
    cudaMemcpy(Qv, params.Qv, sizeof(float) * mesh.vertex_cnt * 9, cudaMemcpyDeviceToHost);

    mesh.vertices = (float*)calloc(mesh.vertex_cnt * 3, sizeof(float));
    mesh.halfedges = (int*)calloc(mesh.halfedge_cnt * 4, sizeof(int));



    int* Meg = (int*)calloc(mesh.halfedge_cnt, sizeof(int));
    cudaMemcpy(Meg, params.Meg, sizeof(int) * mesh.halfedge_cnt, cudaMemcpyDeviceToHost);

    int* he_idx = (int*)calloc(mesh.halfedge_cnt, sizeof(int));
    int new_idx = 0;
    for(int i = 0; i < mesh.halfedge_cnt; i++) {
        int removed = Meg[i] == i ? 1 : 0;
        he_idx[i] = new_idx;

        //printf("%02d: %d %d\n", i, removed, new_idx);
        new_idx += removed;
    }

    int4* collapsed_halfedges = (int4*)calloc(new_idx, sizeof(int4));

    for(int i = 0; i < mesh.halfedge_cnt; i++) {
        if(Meg[i] == i ? 1 : 0) {
            int4 halfedge = halfedges[i];

            halfedge.z = he_idx[halfedge.z];
            halfedge.w = he_idx[halfedge.w];

            collapsed_halfedges[he_idx[i]] = halfedge;
        }
    }

    mesh.halfedge_cnt = new_idx;
    halfedges = collapsed_halfedges;





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

