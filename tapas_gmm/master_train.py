from dataclasses import dataclass
from datetime import datetime
from omegaconf import OmegaConf, SCMode

from tapas_gmm.master_project.definitions import (
    RewardMode,
    StateSpace,
    TaskSpace,
)
from tapas_gmm.master_project.environment import MasterEnv, MasterEnvConfig
from tapas_gmm.master_project.agent import Agent, AgentConfig
from tapas_gmm.utils.argparse import parse_and_build_config


@dataclass
class MasterConfig:
    tag: str
    agent: AgentConfig
    env: MasterEnvConfig
    reward_mode: RewardMode = RewardMode.SPARSE
    task_space: TaskSpace = TaskSpace.SMALL
    state_space: StateSpace = StateSpace.SMALL
    verbose: bool = True


def train_agent(config: MasterConfig):
    # Initialize the environment and agent
    env = MasterEnv(config.env)
    tasks, states, tps = env.publish()
    agent = Agent(config.agent, tasks, states, tps)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    batch_start_time = datetime.now().replace(microsecond=0)
    stop_training = False
    while not stop_training:  # Training loop
        terminal = False
        batch_rdy = False
        obs, goal = env.reset()
        while not terminal and not batch_rdy:
            task_id = agent.act(obs, goal)
            reward, terminal, obs = env.step(task_id)
            batch_rdy = agent.feedback(reward, terminal)
        if batch_rdy:
            train_start_time = datetime.now().replace(microsecond=0)
            stop_training = agent.learn(verbose=config.verbose)
            train_end_time = datetime.now().replace(microsecond=0)
            print(
                "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            )
            print(f"Time: {train_end_time}")
            print(f"Batch Duration: {train_start_time - batch_start_time}")
            print(f"Train Duration: {train_end_time - train_start_time}")
            print(f"Elapsed Time: {train_end_time - start_time}")
            print(
                "--------------------------------------------------------------------------------------------"
            )
            batch_start_time = datetime.now().replace(microsecond=0)

    env.close()

    # print total training time
    print(
        "============================================================================================"
    )
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at: ", start_time)
    print("Finished training at: ", end_time)
    print("Total training time: ", end_time - start_time)
    print(
        "============================================================================================"
    )


def entry_point():

    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    dict_config.agent.name = dict_config.tag
    dict_config.env.evaluator.reward_mode = dict_config.reward_mode
    dict_config.env.task_space = dict_config.task_space
    dict_config.env.state_space = dict_config.state_space

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    train_agent(config)


if __name__ == "__main__":
    entry_point()
