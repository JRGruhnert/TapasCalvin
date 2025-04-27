#!/usr/bin/python3
from copy import deepcopy
import logging
import os
import sys

import hydra
import pybullet as p
import quaternion  # noqa

from tapas_gmm.calvin_project.calvin.io_utils.data_recorder import DataRecorder
from tapas_gmm.calvin_project.calvin.io_utils.motion_planner import PandaArmMotionPlanningSolver

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config_data_collection")
def main(cfg):
    # Load Scene
    env = hydra.utils.instantiate(cfg.env)
    planner = hydra.utils.instantiate(cfg.motion_planner)
    route = planner.create_route(start, goal, **kwargs)

    data_recorder = None
    if cfg.recorder.record:
        data_recorder = DataRecorder(env, cfg.recorder.record_fps, cfg.recorder.enable_tts)

    log.info("Level Initiated and Route Calculated!")

    done = False

    while not done:
        # get input events
        prev_obs = env.get_state_obs()
        action = route.step(prev_obs)
        obs, _, _, info = env.step(action)
        done = info.get("goal_reached", False)

        if cfg.recorder.record:
            data_recorder.step(None, obs, done, info)  # 'None' or planner-specific events

        env.reset()


if __name__ == "__main__":
    main()
