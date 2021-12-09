#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include "mesh_simplify.h"

//void saxpyCuda(int N, float alpha, float* x, float* y, float* result);
//void printCudaInfo();

mesh_t load_mesh(const char *filename) {
    printf("Loading mesh %s\n", filename);

    FILE *input = fopen(filename, "r");

    mesh_t mesh;

    // Read in the size of this file
    fscanf(input, "%d %d\n", &mesh.vertex_cnt, &mesh.halfedge_cnt);

    mesh.vertices  = (float*)calloc(mesh.vertex_cnt, sizeof(float) * 3);
    mesh.halfedges = (int*)calloc(mesh.halfedge_cnt, sizeof(int) * 4);

    // Read the vertex information from file
    for (int i = 0; i < mesh.vertex_cnt; i++) {
        float x, y, z;
        fscanf(input, "%f %f %f\n", &x, &y, &z);

        mesh.vertices[i * 3 + 0] = x;
        mesh.vertices[i * 3 + 1] = y;
        mesh.vertices[i * 3 + 2] = z;
    }

    mesh.triangle_cnt = 0;

    // Read the halfedge information from file
    for (int i = 0; i < mesh.halfedge_cnt; i++) {
        int vertex, triangle, next, twin;
        fscanf(input, "%d %d %d %d\n", &vertex, &triangle, &next, &twin);

        mesh.halfedges[i * 4 + 0] = vertex;
        mesh.halfedges[i * 4 + 1] = triangle;
        mesh.halfedges[i * 4 + 2] = next;
        mesh.halfedges[i * 4 + 3] = twin;

        mesh.triangle_cnt = std::max(mesh.triangle_cnt, triangle + 1);
    }

    printf("Verts: %d\tHalfedges %d\tTriangles: %d\n", 
           mesh.vertex_cnt, mesh.halfedge_cnt, mesh.triangle_cnt);

    fclose(input);

    return mesh;
}

void write_mesh(const char *filename, mesh_t mesh) {
    printf("Writing mesh %s\n", filename);

    FILE *output = fopen(filename, "w");

    // Write size headers
    fprintf(output, "%d %d\n", mesh.vertex_cnt, mesh.halfedge_cnt);

    // Write vertices
    for(int i = 0; i < mesh.vertex_cnt; i++) {
        fprintf(output, "%f %f %f\n", 
                mesh.vertices[3 * i + 0],
                mesh.vertices[3 * i + 1],
                mesh.vertices[3 * i + 2]);
    }

    // Write halfedges
    for(int i = 0; i < mesh.halfedge_cnt; i++) {
        fprintf(output, "%d %d %d %d\n", 
                mesh.halfedges[4 * i + 0],
                mesh.halfedges[4 * i + 1],
                mesh.halfedges[4 * i + 2],
                mesh.halfedges[4 * i + 3]);
    }

    fclose(output);
}



int main(int argc, char** argv)
{

    char in_path[512];
    char out_path[512];

    sprintf(in_path, "../data/%s_halfedge.txt", argv[1]);
    sprintf(out_path, "../data/%s_halfedge_reduced.txt", argv[1]);

    mesh_t mesh = load_mesh(in_path);

    setup(mesh);

    simplify(mesh);

    mesh_t out_mesh = get_results();

    write_mesh(out_path, out_mesh);

    //printCudaInfo();

    return 0;
}
