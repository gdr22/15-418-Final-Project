typedef struct {
    float x;
    float y;
    float z;
} vertex_t;

typedef struct
{
    int vertex_cnt;
    int halfedge_cnt;
    int triangle_cnt;

    //float3* vertices;
    //int4* halfedges;
    float* vertices;
    int* halfedges;
} mesh_t;
