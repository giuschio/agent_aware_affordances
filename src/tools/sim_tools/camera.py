"""
The code in this file is based on the code published alongside the paper
Where2Act: From Pixels to Actions for Articulated 3D Objects(citation in the project README).

The MIT License from the original code is below.

Copyright 2022 Kaichun Mo

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.


    Conventions:
        - images (rgb, xyza, depth...) are of float type np.ndarray[np.float32], values are in [0.0, 1.0].
          We only convert to uint8 in [0, 255] when saving to .png format
"""
import cv2
import numpy as np
import open3d as o3d
import sapien.core as sapien
import typing

from PIL import Image


class Camera(object):
    depth2camera = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    gl2camera = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    def __init__(self,
                 scene: sapien.Scene,
                 mount_actor: sapien.Actor,
                 offset: sapien.Pose = sapien.Pose(),
                 noise: float = 0.0):
        """
        :param scene: scene instance from simulation
        :param mount_actor: link (i.e. rigid object) to which the camera should be mounted
        :param offset: camera offset w.r.t. the mount_actor reference frame
        """
        # For a SAPIEN camera, the x-axis points forward, the y-axis left, and the z-axis upwards.
        self.scene = scene
        self.mount_actor = mount_actor
        self.offset = offset
        self.near_plane = 0.3
        self.noise = noise
        self.far_plane = 5
        self.sizex = 720
        self.sizey = 1280
        self.fovx = np.deg2rad(58)
        self.fovy = np.deg2rad(87)
        self.camera = self.scene.add_mounted_camera('camera', mount_actor, offset, self.sizex, self.sizey, self.fovx,
                                                    self.fovy, self.near_plane, self.far_plane)

    # camera properties
    @property
    def intrinsic_matrix(self):
        # gets the intrinsic camera matrix (i.e. point (in camera frame) to pixel coord)
        return self.camera.get_camera_matrix()

    @property
    def depth2cam(self):
        # permutation matrix from the space defined by the depth image (z-axis points forward, the x-axis right, the y axis downwards)
        # to SAPIEN camera (x-axis points forward, the y-axis left, and the z-axis upwards)
        return self.depth2camera

    @property
    def glcam2world(self):
        # this gets me the inverse of the extrinsic camera matrix (the extrinsic is world2cam)
        # the model matrix is the transformation from OpenGL camera space to SAPIEN world space
        # OpenGL camera -> SAPIEN camera -> world
        # Always call scene.update_render() before querying the transforms
        return self.camera.get_model_matrix()

    @property
    def glcam2cam(self):
        # transformation from OpenGL camera space to SAPIEN camera space
        # OpenGL/Blender: y up and -z forward
        # SAPIEN: z up and x forward
        # Always call scene.update_render() before querying the transforms
        return self.gl2camera

    @property
    def cam2world(self):
        # transformation from SAPIEN camera space to world frame
        # Always call scene.update_render() before querying the transforms
        return self.glcam2world @ np.linalg.inv(self.glcam2cam)

    # getting pictures
    def take_picture(self):
        """
        This function needs to be called before the get_<something> functions are called
        You only need to call take_picture once. After that, you can call any of the get_<something> functions multiple times
        """
        self.camera.take_picture()

    def get_rgba(self):
        """
        Get float rgba image (i.e. values are in [0.0, 1.0])
        :rtype: np.ndarray[np.float32]
        """
        # SAPIEN return
        rgba = self.camera.get_float_texture('Color')  # [H, W, 4]
        return rgba

    def get_rgb(self):
        return self.get_rgba()[:, :, :3]

    def get_object_mask(self):
        """
        Returns a 0/1 object mask (2d picture)
        """
        link_seg = self.camera.get_actor_segmentation()
        return np.array(link_seg > 1).astype('float32')

    def get_links_mask(self, link_ids: typing.List):
        """
        Get a mask over a list of object links
        :param link_ids: list of link ids
        :return: link_mask: 0/1 binary image, a pixel is 1 if it belongs to one of the links in the link_ids list, zero otherwise
        """
        # link_seg associates to each pixel its respective link_id
        link_seg = self.camera.get_actor_segmentation()
        # initialize image
        link_mask = np.zeros((link_seg.shape[0], link_seg.shape[1])).astype('float32')
        for link_id in link_ids:
            # count number of pixels that have ID = link_id
            cur_link_pixels = int(np.sum(link_seg == link_id))
            if cur_link_pixels > 0:
                # if there are pixels in the link_seg with the current link_id, assign to them index+1
                link_mask[link_seg == link_id] = 1.0
        return link_mask

    def get_part_mask(self, part_name='handle'):
        """
        Returns a 0/1 part_mask
        :param part_name: name of the part to segment
        :return: part_mask
        :rtype: np.ndarray[np.float32]
        """
        # read part seg part_ids_to_render_ids
        part_ids_to_render_ids = dict()
        for k in self.scene.render_id_to_visual_name:
            if self.scene.render_id_to_visual_name[k].split('-')[0] == part_name:
                part_id = int(self.scene.render_id_to_visual_name[k].split('-')[-1])
                if part_id not in part_ids_to_render_ids:
                    part_ids_to_render_ids[part_id] = []
                part_ids_to_render_ids[part_id].append(k)
        # generate 0/1 part mask
        part_seg = self.camera.get_visual_segmentation()
        part_mask = np.zeros((part_seg.shape[0], part_seg.shape[1])).astype('float32')
        for partid in part_ids_to_render_ids:
            cur_part_mask = np.isin(part_seg, part_ids_to_render_ids[partid])
            cur_part_mask_pixels = int(np.sum(cur_part_mask))
            if cur_part_mask_pixels > 0:
                part_mask[cur_part_mask] = 1.0
        return part_mask

    def get_handle_mask(self):
        return self.get_part_mask(part_name='handle')

    def get_normalized_depth(self):
        """
        :return depth: normalized depth map (e.g. depth.shape = (448, 448), depth.max is 1.0 = near_plane, depth.min is 0.0 = far_plane)
        """
        return self.camera.get_depth().astype('float32')

    def get_xyza(self) -> np.array:
        """
        :rtype: np.ndarray[np.float32]
        """
        # depth is normalized depth (i.e. depth[n] is in (0,1))
        depth = self.get_normalized_depth()
        # indexes of pixels where depth < infinite
        valid_pixels_y, valid_pixels_x = np.where(depth < 1)
        pts = self.depth_to_pointcloud(depth)
        size1, size2 = depth.shape[0], depth.shape[1]
        return self.pointcloud_to_xyza(pts, valid_pixels_y, valid_pixels_x, size1, size2)

    def get_normal_map(self):
        nor = self.camera.get_normal_rgba()
        # convert from PartNet-space (x-right, y-up, z-backward) to SAPIEN-camera-space (x-front, y-left, z-up)
        # the vectors in new_nor are w.r.t. the camera frame-of-reference
        new_nor = np.array(nor, dtype=np.float32)
        new_nor[:, :, 0] = -nor[:, :, 2]
        new_nor[:, :, 1] = -nor[:, :, 0]
        new_nor[:, :, 2] = nor[:, :, 1]
        return new_nor

    # saving data
    @staticmethod
    def save_png(rgba_img: np.array, path):
        # needs to save in .png format
        if float(np.max(rgba_img)) <= 1.0:
            rgba_img = (rgba_img * 255).clip(0, 255)
        rgba_img = rgba_img.astype("uint8")
        rgba_pil = Image.fromarray(rgba_img)
        rgba_pil.save(path)

    @staticmethod
    def save_png16(rgba_img: np.array, path):
        if rgba_img.shape[2] != 3:
            raise ValueError("only three channels")
        if float(np.max(rgba_img)) <= 1.0:
            rgba_img = (rgba_img * 65536).clip(0, 65536)
        rgba_img = rgba_img.astype('uint16')
        # Save as PNG with imageio
        # imageio.imwrite('io.tif', rgba_img)
        # opencv uses the BGRA channel order instead of RGBA
        rgb_cv2 = rgba_img[:, :, [2, 1, 0]]
        cv2.imwrite(path, rgb_cv2)

    @staticmethod
    def load_png(fpath):
        """
        Load png as float numpy array (i.e. values are in [0.0, 1.0])
        :rtype: np.ndarray[np.float32]
        """
        pic = Image.open(fpath)
        pic = np.array(pic).astype('float32')
        pic /= np.max(pic)
        return pic

    @staticmethod
    def save_ply(pcd: o3d.geometry.PointCloud, path):
        """
        Save an open3d pointcloud to a .ply file
        """
        o3d.io.write_point_cloud(path, pcd)

    # conversions between formats
    def depth_to_pointcloud(self, depth: np.array):
        intrinsic_matrix = self.intrinsic_matrix[:3, :3]
        y, x = np.where(depth < 1)
        z = self.near_plane * self.far_plane / (self.far_plane + depth * (self.near_plane - self.far_plane))
        if self.noise > 0.0:
            z += np.random.normal(loc=0.0, scale=self.noise, size=z.shape)
        points_d = np.dot(np.linalg.inv(intrinsic_matrix), np.stack([x, y, np.ones_like(x)] * z[y, x], 0))

        # For a SAPIEN camera, the x-axis points forward, the y-axis left, and the z-axis upwards.
        # In points_d, the z-axis points forward, the x-axis right, the y axis downwards
        # So to have the pointcloud in camera frame we need z -> x, -x -> y, -y -> z (defined as a property in Camera.depth2cam)
        points = (self.depth2cam @ points_d).T
        return points

    def pointcloud_to_pixels(self, points: np.array) -> np.array:
        """
        This is (almost) the inverse of Camera.depth_to_pointcloud
        :param points: pointcloud points
                       points.shape = (N, 3) with N the number of points in the pointcloud.
                       the points in the pointcloud must be in SAPIEN camera coordinates
        :return pixels:  pixels.shape = (N, 2), in particular pixels[:,0] = column coords (x), pixels[:,1] = row coords (y)
        """
        # to "depth" frame
        points_d = np.linalg.inv(self.depth2cam) @ np.transpose(points)
        # normalized coordinates
        points_d = np.divide(points_d, points_d[2, :])
        points = np.transpose(self.intrinsic_matrix[:3, :3] @ points_d)
        pixels = np.rint(points).astype(int)
        return pixels[:, :2]

    @staticmethod
    def pointcloud_to_xyza(pts, valid_pixels_y, valid_pixels_x, size1, size2):
        """
        For each pixel (i,j), out(i,j) = x,y,z,a coordinates (in camera frame) of the point in the pixel (a = albedo)
        :param valid_pixels_y:
        :param valid_pixels_x:
        :param pts:
        :param size1:
        :param size2:
        :return: out.shape = (size1, size2, 4)
        :rtype: np.ndarray[np.float32]
        """
        out = np.zeros((size1, size2, 4), dtype=np.float32)
        out[valid_pixels_y, valid_pixels_x, :3] = pts
        out[valid_pixels_y, valid_pixels_x, 3] = 1
        return out

    @staticmethod
    def xyza_to_pointcloud(xyza: np.array, rgb: typing.Optional[np.array] = None) -> o3d.geometry.PointCloud:
        """
        From xyza and optional rgb image, to a PointCloud (colored if rgb was provided)
        :param xyza: xyza image (i.e. for each pixel (a,b) xyza[b,a,:3] are the coordinates of that pixel in camera space
        :param rgb: rgb image
        :return: pointcloud (open3d format)
        """
        pcd = o3d.geometry.PointCloud()
        size = xyza.shape[0] * xyza.shape[1]
        points = xyza.reshape((size, -1))
        if rgb is not None:
            colors = rgb.reshape((size, -1))
            pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        return pcd

    @staticmethod
    def xyza_to_downsampled_pointcloud(xyza, n_points):
        # open3d pointcloud
        pcd = Camera.xyza_to_pointcloud(xyza=xyza)
        # points to array
        points = np.array(pcd.points)
        # get all points that are invalid (i.e. have coordinates [0,0,0])
        invalid_points = np.multiply(np.multiply(points[:, 0] == 0, points[:, 1] == 0), points[:, 2] == 0)

        # get valid points
        points_valid = points[np.invert(invalid_points)]
        size = len(points_valid)
        # make sure the pcd is not empty
        if size < 10:
            return None
        replace = bool(size < n_points)
        # downsample the valid pointcloud
        # no distribution is given, numpy uses uniform by default
        samples = np.random.choice(a=size, replace=replace, size=n_points)
        # shape is (10000,3)
        points_downsampled = points_valid[samples, :]
        return points_downsampled

    @staticmethod
    def sample_pointcloud_by_distribution(xyza_map: np.array,
                                          pdf: np.array,
                                          n_points: int,
                                          rgb=None) -> o3d.geometry.PointCloud:
        """
        :param xyza_map:
        :param pdf: probability density function
        :param n_points:
        :param rgb:
        :return:
        """
        pcd = Camera.xyza_to_pointcloud(xyza_map, rgb)
        size = pdf.size
        flat = pdf.reshape(size).astype('float64')
        flat /= sum(flat)
        samples = np.random.choice(a=size, p=flat, size=n_points, replace=False)
        pcd_d = pcd.select_by_index(np.array(samples))
        return pcd_d

    @staticmethod
    def sample_pointcloud_to_size(pcd: o3d.geometry.PointCloud,
                                  n_points: int) -> o3d.geometry.PointCloud:
        """
        :param pcd: pointcloud
        :param n_points: number of points in the output pointcloud
        :return:
        """
        size = len(pcd.points)
        # no distribution is given, numpy uses uniform by default
        samples = np.random.choice(a=size, replace=False, size=n_points)
        pcd_d = pcd.select_by_index(np.array(samples))
        return pcd_d

    @staticmethod
    def draw_pointcloud(pcd: o3d.geometry.PointCloud):
        o3d.visualization.draw_geometries([pcd])

    @staticmethod
    def sample_pixel(xy_map):
        """
        Sample a pixel index from a probability map map
        :param xy_map: binary map (0/1) where only true (1) pixels can be sampled
        """
        lines = xy_map.shape[0]
        cols = xy_map.shape[1]
        n_points = cols * lines
        # Create a flat copy of the array
        flat = xy_map.reshape(n_points)
        normalizer = float(np.linalg.norm(flat, ord=1))
        flat = None if normalizer == 0. else flat / normalizer
        # Then, sample an index from the 1D array with the probability distribution from the original array
        sample_index = np.random.choice(a=n_points, p=flat)
        # [[a,b,c,d],[e,f,g,h]] -> f is at line 1 = 5//4 ant column 1 = 5%4
        # where 5 is the flattened index and 4 is the line length = number of columns
        y_line, x_col = sample_index // cols, sample_index % cols
        return int(y_line), int(x_col)
