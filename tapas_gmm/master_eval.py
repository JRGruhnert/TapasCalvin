from dataclasses import dataclass
from datetime import datetime
from omegaconf import OmegaConf, SCMode

from tapas_gmm.master_project.environment import MasterEnv, MasterEnvConfig
from tapas_gmm.master_project.agent import Agent, AgentConfig
from tapas_gmm.master_project.networks import NetworkType
from tapas_gmm.utils.argparse import parse_and_build_config


@dataclass
class MasterConfig:
    tag: str
    nt: NetworkType
    agent: AgentConfig
    env: MasterEnvConfig
    verbose: bool = True


def train_agent(config: MasterConfig):
    # Initialize the environment and agent
    env = MasterEnv(config.env)
    tasks, states, tps = env.publish()
    agent = Agent(config.agent, config.nt, config.tag, tasks, states, tps)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    batch_start_time = datetime.now().replace(microsecond=0)
    stop_training = False
    while not stop_training:  # Training loop
        terminal = False
        batch_rdy = False
        obs, goal = env.reset()
        while not terminal and not batch_rdy:
            # TODO ask in console about next task. preferable via index with name of task
            print(f"{0}: Reset")
            for i, task in enumerate(tasks, start=1):
                print(f"{i}: {task.name}")
            choice = input("Enter the Task id: ")
            task_id = int(choice)
            if task_id == 0:
                print("Resetting environment...")
                obs, goal = env.reset()
                terminal = False
            else:
                task_id -= 1  # Adjust for zero-based index
                reward, terminal, obs = env.step(task_id, verbose=True)
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

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    train_agent(config)


if __name__ == "__main__":
    entry_point()
