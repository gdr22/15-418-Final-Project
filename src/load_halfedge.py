import numpy as np
import open3d as o3d

data = "../data/bunny.ply"

mesh = o3d.io.read_triangle_mesh(data, print_progress = True)

halfedge_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)

verts = np.asarray(halfedge_mesh.vertices)
halfedges = halfedge_mesh.half_edges

print("Verts: %d" % len(verts))
print("HalfEdges: %d" % len(halfedges))

f = open("../data/bunny_halfedge.txt", "w")
f.write("%d %d\n" % (len(verts), len(halfedges)))

for vert in verts:
    f.write("%f %f %f\n" % (vert[0], vert[1], vert[2]))

for halfedge in halfedges:
    if halfedge.is_boundary:
        print(halfedge)

    f.write("%d %d %d %d\n" % (halfedge.vertex_indices[0], halfedge.triangle_index, halfedge.next, halfedge.twin))

f.close()