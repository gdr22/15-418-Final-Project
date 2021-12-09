# 15-418-Final-Project

## Mesh Simplification via Embedded Tree Collapsing Implemented in CUDA

Nicolas Mignucci, George Ralph

## Progress (12/9/2021)

We have written code to execute all necessary of the edge collapsing (steps 1, 2, 5, and 6 in Lee and Kyung's paper). In step 2 we generate trees of edges, in step 5 we collapse edges to the roots of the trees, and in step 6 we refine vertex positions. We also added code which removes the halfedges and vertices deleted by the previous steps from their corresponding arrays using an exclusive scan. With this we have a functional version of edge collapsing. Comparing it to the sequential implementation we observe a 398x speedup on the small mesh (34,834 vertices and 69,451 triangles) and a 1179x speedup for a very large mesh (4,999,996 vertices and 10,000,000 triangles). Our implementation produces reasonable output meshes, with slight, infrequent errors in shading due to uneven triangulation and points of normal inversion. 

(More information can be found in Final Project Report PDF document in the ./doc directory)

## Progress (11/22/2021)

Currently, we have written code to create a half edge mesh from a Stanford format (.ply) mesh and store this information in a human readable file. We are also able to load the half-edge mesh from this file and onto the GPU in the format specified by the Lee-Kyung paper. We have also implemented the first step of the algorithm, which requires computation of plane equations over all triangles, followed by computation of the quartic error matrix coefficients for all vertices. This code tests our halfedge implementation. By observation of the code written so far, which makes extensive use of gather operations due to the halfedge mesh structure, our final program is likely to be memory bound.

(More information can be found in the Final Project Milestone PDF document in the ./doc directory)
