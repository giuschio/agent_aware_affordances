"""
Import an .obj polygonal mesh file, convert mesh to numpy array, scale and translate the mesh, convex decomposition, combine multiple meshes...
"""

import numpy as np
# https://github.com/kmammou/v-hacd
import os
import pybullet as p
# https://pymeshlab.readthedocs.io/en/latest/intro.html
import pymeshlab
# https://urdfpy.readthedocs.io/en/latest/index.html
from urdfpy import URDF, Collision, Geometry, Mesh, Visual

from tools.utils.misc import dir_exists


def get_mesh_matrices(path: str):
    """
    Import an .obj polygonal mesh file and convert mesh to numpy array
    :param path: path to .obj file
    :return: vertex and faces matrices
    :rtype: np.array
    """
    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    ms.load_new_mesh(path)
    # get a reference to the current mesh
    m = ms.current_mesh()
    # get numpy arrays of vertices and faces of the current mesh
    v_matrix = m.vertex_matrix()
    f_matrix = m.face_matrix()

    return v_matrix, f_matrix


def center_mesh(vertices: np.array):
    """
    Set the mesh origin to be (0,0,0)
    :param vertices: vertices np matrix
    :return: centered mesh vertices
    """
    mins = np.amin(vertices, axis=0)
    # do no subtract x coordinate, want the mesh to be centered around x
    mins[0] = 0
    return vertices - mins


def scale_mesh(vertices, scale=1.0):
    return vertices * scale


def save_poly_mesh(vertices: np.array, faces: np.array, fname: str):
    """
    Save an .obj polygonal mesh
    :param fname: filename
    """
    # create a new Mesh with the two arrays
    m = pymeshlab.Mesh(vertices, faces)

    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    # add the mesh to the MeshSet (note: ms contains a *copy* of m)
    ms.add_mesh(m, "mesh")

    # save the current mesh
    # obj format in meshlab supports saving polygonal meshes
    ms.save_current_mesh(fname)


def convex_decomp(in_path, out_path):
    """
    Convex decomposition of a mesh using VHACD algo
    :param in_path: input mesh path
    :param out_path: savepath for output mesh collection
    """
    name_log = "log.txt"
    p.vhacd(in_path, out_path, name_log, alpha=0.04, resolution=50000)


def combine_meshes_by_link(fname):
    """
    Given an URDF model, for each link combine all collision meshes into one
    :param object_path:
    :output: the output meshes are saved in object_path/consolidated_collision_meshes/
    """
    object_path = os.path.dirname(fname)
    model = URDF.load(fname)
    links = model.links
    for link in links:
        mesh = link.collision_mesh
        if mesh is not None:
            m = pymeshlab.Mesh(np.array(mesh.vertices), np.array(mesh.faces))
            # create a new MeshSet
            ms = pymeshlab.MeshSet()

            # add the mesh to the MeshSet (note: ms contains a *copy* of m)
            ms.add_mesh(m, str(link.name))

            # save the current mesh
            # obj format in meshlab supports saving polygonal meshes
            fname = str(link.name) + ".obj"
            directory = os.path.join(os.path.join(object_path, "consolidated_collision_meshes"))
            dir_exists(directory)
            m_path = os.path.join(os.path.join(object_path, "consolidated_collision_meshes"), fname)
            ms.save_current_mesh(m_path)

            new_mesh = Mesh(filename=os.path.join("consolidated_collision_meshes", fname),
                            meshes=[link.collision_mesh])
            coll_geom = Geometry(mesh=new_mesh)
            new_collision = Collision(geometry=coll_geom, name='coll_mesh', origin=np.identity(4))
            new_visuals = Visual(geometry=coll_geom, name='coll_mesh', origin=np.identity(4))
            link.collisions = [new_collision]
            link.visuals = [new_visuals]

    model.save(os.path.join(object_path, 'mobility_original.urdf'))


def bounding_box(mesh_path):
    """
    Print bounding box dimension and center of a mesh
    """
    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    ms.load_new_mesh(mesh_path)
    # get a reference to the current mesh
    m = ms.current_mesh()
    # get numpy arrays of vertices and faces of the current mesh
    v_matrix = m.vertex_matrix()
    m = np.min(v_matrix, axis=0)
    M = np.max(v_matrix, axis=0)
    print("bounding box dimensions", M-m)
    print("bounding box center", (M+m)/2)
