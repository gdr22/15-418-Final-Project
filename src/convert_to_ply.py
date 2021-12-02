import numpy as np
import open3d as o3d

data = "../data/bunny_halfedge_reduced.txt"

with open(data) as file:
    lines = file.readlines()

    # Load in the mesh size
    line = lines[0]
    nums = line.split(' ')    

    vert_cnt = int(nums[0])
    halfedge_cnt = int(nums[1])

    verts = []
    halfedges = []
    tris = []

    tri_cnt = 0

    # Load mesh vertices
    for i in range(1, vert_cnt + 1):
        line = lines[i]
        nums = line.split(' ')

        vert = (float(nums[0]), float(nums[1]), float(nums[2]))
        verts.append(vert)

    # Load mesh halfedges
    for i in range(vert_cnt + 1, vert_cnt + halfedge_cnt + 1):
        line = lines[i]
        nums = line.split(' ')

        halfedge = (int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3]))
        (vert, tri, next, twin) = halfedge
        halfedges.append(halfedge)

        tri_cnt = max(tri_cnt, tri + 1)

    # Build the triangles
    for i in range(tri_cnt):
        tris.append((0, 0, 0))

    for halfedge in halfedges:
        if halfedge == None:
            continue

        # Get the three verts for this triangle
        (vert0, tri, next0, _) = halfedge
        (vert1, tri, next1, _) = halfedges[next0]
        (vert2, tri, next2, _) = halfedges[next1]

        # Mark these halfedges as visited
        halfedges[next0] = None
        halfedges[next1] = None
        halfedges[next2] = None

        # Store the triangle
        tris[tri] = (vert0, vert1, vert2)


    # Build the mesh
    mesh = o3d.geometry.TriangleMesh()

    verts = np.asarray(verts)
    tris  = np.asarray(tris)
    
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)

    # Write the mesh
    o3d.io.write_triangle_mesh("../data/bunny_reduced.ply", mesh)

