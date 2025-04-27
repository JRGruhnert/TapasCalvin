from enum import Enum

import numpy as np

MAX_FORCE = 4


class ButtonState(Enum):
    ON = 1
    OFF = 0


class Switch:
    def __init__(self, name, cfg, uid, p, cid):
        self.name = name
        self.p = p
        self.cid = cid
        # get joint_index by name (to prevent index errors when additional joints are added)
        joint_index = next(
            i
            for i in range(self.p.getNumJoints(uid, physicsClientId=self.cid))
            if self.p.getJointInfo(uid, i, physicsClientId=self.cid)[1].decode("utf-8") == name
        )
        self.joint_index = joint_index
        self.uid = uid
        self.initial_state = cfg["initial_state"]
        self.effect = cfg["effect"]
        self.ll, self.ul = self.p.getJointInfo(uid, joint_index, physicsClientId=self.cid)[8:10]
        self.trigger_threshold = (self.ll + self.ul) / 2
        self.p.setJointMotorControl2(
            self.uid,
            self.joint_index,
            controlMode=p.VELOCITY_CONTROL,
            force=MAX_FORCE,
            physicsClientId=self.cid,
        )
        self.state = ButtonState.OFF
        self.light = None

    def reset(self, state=None):
        _state = self.initial_state if state is None else state
        self.p.resetJointState(
            self.uid,
            self.joint_index,
            _state,
            physicsClientId=self.cid,
        )
        self.state = ButtonState.OFF

    def step(self):
        if self.is_pressed:
            if self.light is not None and self.state == ButtonState.OFF:
                self.light.turn_on()
            self.state = ButtonState.ON
        else:
            if self.light is not None and self.state == ButtonState.ON:
                self.light.turn_off()
            self.state = ButtonState.OFF

    @property
    def is_pressed(self, state=None):
        joint_state = state
        if joint_state is None:
            joint_state = self.p.getJointState(self.uid, self.joint_index, physicsClientId=self.cid)[0]

        if self.initial_state <= self.trigger_threshold:
            return joint_state > self.trigger_threshold
        elif self.initial_state > self.trigger_threshold:
            return joint_state < self.trigger_threshold

    def get_state(self):
        """return button joint state"""
        return float(self.state.value)

    def get_pose(self, euler_obs=False):
        pos, orn = self.p.getBasePositionAndOrientation(self.uid, physicsClientId=self.cid)
        if euler_obs:
            orn = self.p.getEulerFromQuaternion(orn)
        return np.concatenate([pos, orn])

    def get_info(self):
        return {"joint_state": self.get_state(), "logical_state": self.state.value}

    def add_effect(self, light):
        self.light = light
