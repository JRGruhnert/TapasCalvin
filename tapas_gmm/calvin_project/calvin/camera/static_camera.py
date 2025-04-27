import numpy as np
import pybullet as p

from tapas_gmm.calvin_project.calvin.camera.camera import Camera
import cv2


class StaticCamera(Camera):
    def __init__(
        self,
        fov,
        aspect,
        nearval,
        farval,
        width,
        height,
        look_at,
        look_from,
        up_vector,
        cid,
        name,
        robot_id=None,
        objects=None,
    ):
        """
        Initialize the camera
        Args:
            argument_group: initialize the camera and add needed arguments to argparse

        Returns:
            None
        """
        self._nearval = nearval
        self._farval = farval
        self.fov = fov
        self.aspect = aspect
        self.look_from = look_from
        self.look_at = look_at
        self.up_vector = up_vector
        self._width = width
        self._height = height
        self._viewMatrix = p.computeViewMatrix(
            cameraEyePosition=look_from, cameraTargetPosition=look_at, cameraUpVector=self.up_vector
        )
        self._projectionMatrix = p.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=self._nearval, farVal=self._farval
        )
        self.cid = cid
        self._name = name

        self.sphere_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.005,  # Adjust the radius as needed
            rgbaColor=[216, 250, 8, 1],  # Red color (RGBA)
        )
        self.marker = []

    def set_position_from_gui(self):
        info = p.getDebugVisualizerCamera(physicsClientId=self.cid)
        look_at = np.array(info[-1])
        dist = info[-2]
        forward = np.array(info[5])
        look_from = look_at - dist * forward
        self._viewMatrix = p.computeViewMatrix(
            cameraEyePosition=look_from, cameraTargetPosition=look_at, cameraUpVector=self.up_vector
        )
        look_from = [float(x) for x in look_from]
        look_at = [float(x) for x in look_at]
        return look_from, look_at

    def world_to_pixel(self, point, view_matrix, proj_matrix, img_width, img_height):
        """
        Projects a 3D world point to 2D pixel coordinates.

        :param point: A list or array [x, y, z] in world coordinates.
        :param view_matrix: The view matrix from PyBullet (flat list of 16 values).
        :param proj_matrix: The projection matrix from PyBullet (flat list of 16 values).
        :param img_width: Image width in pixels.
        :param img_height: Image height in pixels.
        :return: (u, v) pixel coordinates.
        """
        # Convert the view and projection matrices into 4x4 numpy arrays.
        # PyBullet returns them in column-major order, so use order='F'.
        view = np.array(view_matrix, dtype=np.float32).reshape((4, 4), order="F")
        proj = np.array(proj_matrix, dtype=np.float32).reshape((4, 4), order="F")

        # Convert the point to homogeneous coordinates.
        point_hom = np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)

        # Apply the view and projection matrices.
        clip_coords = proj.dot(view.dot(point_hom))

        # Perform perspective divide to get Normalized Device Coordinates (NDC)
        if clip_coords[3] != 0:
            ndc = clip_coords[:3] / clip_coords[3]
        else:
            ndc = clip_coords[:3]

        # Map NDC [-1, 1] to pixel coordinates.
        u = int((ndc[0] * 0.5 + 0.5) * img_width)
        # In many image coordinate systems, v=0 is the top of the image so flip the y-coordinate.
        v = int((1 - (ndc[1] * 0.5 + 0.5)) * img_height)

        return u, v

    def update_marker_points(self, points):
        # Update positions of existing markers.
        for marker in self.marker:
            p.removeBody(marker)
        self.marker = []

        for point in points:
            # Optionally, you can update the orientation as well; here we set it to identity.
            temp = p.createMultiBody(
                baseMass=0,  # Makes the object static
                baseCollisionShapeIndex=-1,  # No collision shape
                baseVisualShapeIndex=self.sphere_visual,
                basePosition=point[:3],  # The desired 3D position of the sphere
            )
            self.marker.append(temp)

    def _render(self):
        return p.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=self._viewMatrix,
            projectionMatrix=self._projectionMatrix,
            physicsClientId=self.cid,
        )
