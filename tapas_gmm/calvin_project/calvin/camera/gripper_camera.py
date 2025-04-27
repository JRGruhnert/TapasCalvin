import numpy as np
import pybullet as p

from tapas_gmm.calvin_project.calvin.camera.camera import Camera


class GripperCamera(Camera):
    def __init__(self, fov, aspect, nearval, farval, width, height, robot_id, cid, name, objects=None):
        self.cid = cid
        self.robot_uid = robot_id
        links = {
            p.getJointInfo(self.robot_uid, i, physicsClientId=self.cid)[12].decode("utf-8"): i
            for i in range(p.getNumJoints(self.robot_uid, physicsClientId=self.cid))
        }
        self.gripper_cam_link = links["gripper_cam"]
        self.fov = fov
        self.aspect = aspect
        self._nearval = nearval
        self._farval = farval
        self._width = width
        self._height = height

        self._name = name

    def _render(self):
        "Render the scene from the tcp's perspective."
        camera_ls = p.getLinkState(
            bodyUniqueId=self.robot_uid, linkIndex=self.gripper_cam_link, physicsClientId=self.cid
        )
        camera_pos, camera_orn = camera_ls[:2]
        cam_rot = p.getMatrixFromQuaternion(camera_orn)
        cam_rot = np.array(cam_rot).reshape(3, 3)
        cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
        # camera: eye position, target position, up vector
        self._viewMatrix = p.computeViewMatrix(camera_pos, camera_pos + cam_rot_y, -cam_rot_z)
        self._projectionMatrix = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=self.aspect, nearVal=self._nearval, farVal=self._farval
        )
        return p.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=self._viewMatrix,
            projectionMatrix=self._projectionMatrix,
            physicsClientId=self.cid,
        )
