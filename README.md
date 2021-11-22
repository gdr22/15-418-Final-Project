# 15-418-Final-Project

## Mesh Simplification via Embedded Tree Collapsing Implemented in CUDA

Nicolas Mignucci, George Ralph

## Progress (11/22/2021)

Currently, we have written code to create a half edge mesh from a Stanford format (.ply) mesh and store this information in a human readable file. We are also able to load the half-edge mesh from this file and onto the GPU in the format specified by the Lee-Kyung paper. We have also implemented the first step of the algorithm, which requires computation of plane equations over all triangles, followed by computation of the quartic error matrix coefficients for all vertices. This code tests our halfedge implementation. By observation of the code written so far, which makes extensive use of gather operations due to the halfedge mesh structure, our final program is likely to be memory bound.

(More information can be found in the Final Project Milestone PDF document in the ./doc directory)