from tapas_gmm.master_project.definitions import StateSpace, TaskSpace
from tapas_gmm.master_project.environment import MasterEnvConfig
from conf.master.shared.evaluator import evaluator
from conf.master.shared.storage import storage
from conf.master.shared.sampler import sampler

env1 = MasterEnvConfig(
    task_space=TaskSpace.SMALL,
    state_space=StateSpace.SMALL,
    eval_mode=False,
    pybullet_vis=False,
    debug_vis=False,
    evaluator=evaluator,
    storage=storage,
    sampler=sampler,
)

env2 = MasterEnvConfig(
    task_space=TaskSpace.SMALL,
    state_space=StateSpace.ALL,
    eval_mode=False,
    pybullet_vis=False,
    debug_vis=False,
    evaluator=evaluator,
    storage=storage,
    sampler=sampler,
)

env3 = MasterEnvConfig(
    task_space=TaskSpace.ALL,
    state_space=StateSpace.ALL,
    eval_mode=False,
    pybullet_vis=False,
    debug_vis=False,
    evaluator=evaluator,
    storage=storage,
    sampler=sampler,
)
