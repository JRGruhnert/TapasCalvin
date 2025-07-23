from dataclasses import dataclass
import pathlib
from typing import Dict, Set

from loguru import logger
import numpy as np
from tapas_gmm.master_project.master_definitions import (
    TaskSpace,
    Task,
    State,
    _origin_ee_tp_pose,
)

from tapas_gmm.utils.select_gpu import device
from tapas_gmm.master_project.master_observation import MasterObservation
from tapas_gmm.policy import import_policy
from tapas_gmm.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm.policy.models.tpgmm import (
    AutoTPGMM,
    AutoTPGMMConfig,
    Gaussian,
    ModelType,
    TPGMMConfig,
)


@dataclass(frozen=True)
class HRLHelper:
    """Utility class for returning meaningful model sets."""

    @classmethod
    def convert_observation(
        cls, task: Task, obs: MasterObservation
    ) -> MasterObservation:
        if task.value.reversed:
            obs.update_ee_pose(
                _origin_ee_tp_pose,
            )
        return obs

    @classmethod
    @classmethod
    def _get_gaussians_from_model(cls, tpgmm: AutoTPGMM) -> Dict[int, Gaussian]:
        result: Dict[int, Gaussian] = {}
        # This gives me one HMM per frame; each HMM has K Gaussians and a K×K transition matrix.
        frame_hmms = tpgmm.get_frame_marginals(time_based=False)
        for i, segment in enumerate(tpgmm.segment_frames):
            for j, frame_idx in enumerate(segment):
                for gaussian in frame_hmms[i][j].gaussians:
                    mu3, sigma1 = gaussian.get_mu_sigma(
                        mu_on_tangent=False, as_np=False
                    )
                if result.get(frame_idx) is None:
                    # for gaussian in frame_hmms[i][j].gaussians:
                    #    mu1, sigma1 = gaussian.get_mu_sigma(mu_on_tangent=False, as_np=True)
                    #    mu2, sigma2 = gaussian.get_mu_sigma(mu_on_tangent=True, as_np=True)
                    #    mu3, sigma1 = gaussian.get_mu_sigma(
                    #        mu_on_tangent=False, as_np=False
                    #    )
                    #    mu4, sigma2 = gaussian.get_mu_sigma(mu_on_tangent=True, as_np=False)

                    result[frame_idx] = frame_hmms[i][j].gaussians[0]
                    mu, sigman = result[frame_idx].get_mu_sigma(
                        mu_on_tangent=True, as_np=False
                    )
        return result
