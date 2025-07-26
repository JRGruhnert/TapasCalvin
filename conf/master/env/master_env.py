from tapas_gmm.master_project.definitions import StateSpace, TaskSpace
from tapas_gmm.master_project.environment import MasterEnvConfig
from conf.master.env.evaluator import evaluator
from conf.master.env.storage import storage
from conf.master.env.sampler import sampler

env = MasterEnvConfig(
    task_space=TaskSpace.SMALL,
    state_space=StateSpace.SMALL,
    eval_mode=False,
    pybullet_vis=False,
    debug_vis=False,
    evaluator=evaluator,
    storage=storage,
    sampler=sampler,
)
