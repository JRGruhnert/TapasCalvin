import math

import numpy as np
import pybullet as p


class Camera:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def render(self):

        image = self._render()
        rgb, depth, mask = self.process_rgbd(image, self._nearval, self._farval)

        # mask = self.rgb_image_to_mask(image)

        ptc = self.distance_map_to_point_cloud(depth, self.fov, self._width, self._height)
        # ptc = None  # TODO Make it parameter. I dont need it
        return rgb, depth, ptc, mask  # Now also returning the segmentation mask

    def _render(self):
        raise NotImplementedError

    def distance_map_to_point_cloud(self, distances, fov, width, height):
        """Converts from a depth map to a point cloud.
        Args:
          distances: An numpy array which has the shape of (height, width) that
            denotes a distance map. The unit is meter.
          fov: The field of view of the camera in the vertical direction. The unit
            is radian.
          width: The width of the image resolution of the camera.
          height: The height of the image resolution of the camera.
        Returns:
          point_cloud: The converted point cloud from the distance map. It is a numpy
            array of shape (height, width, 3).
        """
        f = height / (2 * math.tan(fov / 2.0))
        px = np.tile(np.arange(width), [height, 1])
        x = (2 * (px + 0.5) - width) / f * distances / 2
        py = np.tile(np.arange(height), [width, 1]).T
        y = (2 * (py + 0.5) - height) / f * distances / 2
        point_cloud = np.stack((x, y, distances), axis=-1)
        return point_cloud

    def z_buffer_to_real_distance(self, z_buffer, far, near):
        """Function to transform depth buffer values to distances in camera space"""
        return 1.0 * far * near / (far - (far - near) * z_buffer)

    def process_rgbd(self, obs, nearval, farval):
        (width, height, rgbPixels, depthPixels, segmentationMaskBuffer) = obs
        rgb = np.reshape(rgbPixels, (height, width, 4))
        rgb_img = rgb[:, :, :3]
        depth_buffer = np.reshape(depthPixels, [height, width])
        depth = self.z_buffer_to_real_distance(z_buffer=depth_buffer, far=farval, near=nearval)
        return rgb_img, depth, segmentationMaskBuffer

    # Reference: world2pixel
    # https://github.com/bulletphysics/bullet3/issues/1952
    def project(self, point):
        """
        Projects a world point in homogeneous coordinates to pixel coordinates
        Args
            point: np.array of len 4; indicates the desired point to project
        Output
            (x, y): tuple (u, v); pixel coordinates of the projected point
        """

        # reshape to get homogeneus transform
        persp_m = np.array(self._projectionMatrix).reshape((4, 4)).T
        view_m = np.array(self._viewMatrix).reshape((4, 4)).T

        # Perspective proj matrix
        world_pix_tran = persp_m @ view_m @ point
        world_pix_tran = world_pix_tran / world_pix_tran[-1]  # divide by w
        world_pix_tran[:3] = (world_pix_tran[:3] + 1) / 2
        x, y = world_pix_tran[0] * self._width, (1 - world_pix_tran[1]) * self._height
        x, y = np.floor(x).astype(int), np.floor(y).astype(int)
        return (x, y)

    def deproject(self, point, depth_img, homogeneous=False):
        """
        Deprojects a pixel point to 3D coordinates
        Args
            point: tuple (u, v); pixel coordinates of point to deproject
            depth_img: np.array; depth image used as reference to generate 3D coordinates
            homogeneous: bool; if true it returns the 3D point in homogeneous coordinates,
                         else returns the world coordinates (x, y, z) position
        Output
            (x, y): np.array; world coordinates of the deprojected point
        """
        T_world_cam = np.linalg.inv(np.array(self._viewMatrix).reshape((4, 4)).T)

        u, v = point
        z = depth_img[v, u]
        foc = self._height / (2 * np.tan(np.deg2rad(self.fov) / 2))
        x = (u - self._width // 2) * z / foc
        y = -(v - self._height // 2) * z / foc
        z = -z
        world_pos = T_world_cam @ np.array([x, y, z, 1])
        if not homogeneous:
            world_pos = world_pos[:3]
        return world_pos

    @property
    def viewMatrix(self):
        return self._viewMatrix

    @property
    def projectionMatrix(self):
        return calculate_intrinsic_matrix(self.fov, self._width, self._height)

    @property
    def nearval(self):
        return self._nearval

    @property
    def farval(self):
        return self._farval

    @property
    def name(self):
        return self._name

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


def calculate_intrinsic_matrix(fov_y, width, height):
    # Convert field of view from degrees to radians
    fov_y_rad = np.deg2rad(fov_y)
    # Calculate focal length in pixels
    fy = height / (2 * np.tan(fov_y_rad / 2))
    fx = fy  # Assuming square pixels
    # Principal point (image center)
    cx = width / 2
    cy = height / 2
    # Construct the intrinsic matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K
