from dataclasses import dataclass

from tapas_gmm.policy.models.master_project.composite_model import (
    CompositeModel,
    TapasWrapperModel,
)
from tapas_gmm.dataset.demos import Demos, DemosSegment, PartialFrameViewDemos
from tapas_gmm.dataset.trajectory import Trajectory
from tapas_gmm.policy.models.tpgmm import (
    AutoTPGMM,
    AutoTPGMMConfig,
    DemoSegmentationConfig,
    FrameSelectionConfig,
)

import dataclasses
import enum
import itertools
from collections import OrderedDict

# import pickle
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Callable, List, Sequence, Union

import dill as pickle
import numpy as np
import pandas as pd
import riepybdlib as rbd
import torch
from loguru import logger
from omegaconf import OmegaConf
from scipy.linalg import block_diag
from tqdm.auto import tqdm


@dataclass
class DifferentialEvoluationConfig:
    value_a: bool = True


@dataclass
class SequenceOptimumConfig:
    value_a: bool = True


@dataclass
class HierarchicalLearnerConfig:
    sequencer: DifferentialEvoluationConfig = DifferentialEvoluationConfig()
    sequence_optimum: SequenceOptimumConfig = SequenceOptimumConfig()

    learn_new_model: bool = True

    tpgmm: AutoTPGMMConfig = AutoTPGMMConfig(
        frame_selection=FrameSelectionConfig(),
        demos_segmentation=DemoSegmentationConfig(
            max_idx_distance=15,
        ),
    )


GMM = rbd.statistics.GMM
Gaussian = rbd.statistics.Gaussian
MarkovModel = rbd.statistics.HMM | rbd.statistics.HSMM
Manifold = rbd.manifold.Manifold


class HierarchicalLearner:
    def __init__(self, config: HierarchicalLearnerConfig):
        self.config = config

        self.model_pool: List[CompositeModel] = []
        self._demos: Demos | None = None
        self._demos_segments: tuple[DemosSegment, ...] | None = None
        self.segment_frames: list[tuple[int]] | None = None
        self.segment_frame_views: list[PartialFrameViewDemos] | None = None

        self._online_active_segment: int | None = None
        self._online_joint_models: tuple[GMM, ...] | None = None
        self._online_trans_margs_joint: tuple[GMM, ...] | None = None  # for dbg
        self._online_first_step: bool = True

    def _delete_model(self, idx: int):
        """
        Delete the model at the given index out of the pool of models which are used by the policy.
        """
        del self.model_pool[idx]

    def _add_model(self, model: CompositeModel):
        """
        Add a model to the pool of models which are used by the policy.
        """
        self.model_pool.append(model)

    def reset_episode(self):
        """
        Reset the episode for online prediction. Ie. next prediction is treated as the
        first step of a new episode.
        """
        self._online_first_step = True
        self._online_active_segment = 0

    def load_models(self, model_paths: List[str]):
        """
        Load the models from the given paths.
        """
        for path in model_paths:
            with open(path, "rb") as f:
                tapas_model = pickle.load(f)

                model = TapasWrapperModel(tapas_model)
                self._add_model(tapas_model)

    @cached_property
    def _segment_lengths(self) -> np.ndarray:
        assert self._demos_segments is not None

        lens = np.array([s.mean_traj_len for s in self._demos_segments])
        return lens / lens.sum()

    def copy(self):
        logger.warning("Copy ATPGMM not properly tested yet.")
        other = AutoTPGMM(self.config)

        other.segment_gmms = (
            [m.copy() for m in self.segment_gmms]
            if self.segment_gmms is not None
            else None
        )

        other._demos_segments = self._demos_segments
        other.segment_frame_views = self.segment_frame_views
        other.segment_frames = self.segment_frames

        other._fix_frames = self._fix_frames

    def _create_tpgmm(self, overwrites: dict[str, Any] | None = None) -> TPGMM:
        config = self.config.tpgmm
        if overwrites is not None:
            config = dataclasses.replace(config, **overwrites)
        return TPGMM(config)

    def _setup_data(self, demos: Demos) -> None:
        if self._demos is None:
            self._demos = demos
        elif self._demos != demos:
            logger.warning("Overwriting demos. Need to reset the model?")
            self._demos = demos
        else:
            return

    def fit_trajectories(
        self,
        demos: Demos,
        fix_frames: bool = True,
        init_strategy: InitStrategy | None = None,
        fitting_actions: Sequence[FittingStage] | None = None,
        plot_optim: bool = False,
        global_frames: bool = False,
    ) -> tuple[tuple[float], tuple[float]]:
        """
        Fit the model to the given trajectories.

        Upon initialization, segments the trajectories and selects the frames for each
        segment. Then fits a TPGMM to each segment.

        Returned liks and average log liks are per segment.
        """
        logger.info("Fitting AutoTPGMM", filter=False)

        if self._fix_frames is None:
            self._fix_frames = fix_frames
        else:
            assert (
                self._fix_frames == fix_frames
            ), "Should use consistent fix_frames for all fitting calls."

        self._setup_data(demos)

        fitting_actions = self._select_fitting_actions(fitting_actions)

        if FittingStage.INIT in fitting_actions:
            self._segment_and_frame_select(
                demos=demos,
                fix_frames=fix_frames,
                init_strategy=init_strategy,
                fitting_actions=fitting_actions,
                plot_optim=plot_optim,
                global_frames=global_frames,
            )

        assert self.segment_gmms is not None and self.segment_frame_views is not None

        liks = []
        avg_logliks = []

        with tqdm(total=len(self.segment_gmms), desc="Fitting segments") as pbar:
            for gmm, data in zip(self.segment_gmms, self.segment_frame_views):
                lik, avg_loglik = gmm.fit_trajectories(
                    data,
                    fix_frames=fix_frames,
                    init_strategy=init_strategy,
                    fitting_actions=fitting_actions,
                    plot_optim=plot_optim,
                )
                liks.append(lik)
                avg_logliks.append(avg_loglik)
                pbar.update(1)

        return tuple(liks), tuple(avg_logliks)

    def _get_segment_tpgmm_overwrites(
        self,
        n_segments: int,
        relative_duration: float | None = None,
        min_n_components: int = 1,
        drop_action_component: bool = False,
    ) -> dict[str, Any]:
        """
        Helper function for parameterizing the segment TPGMMs.
        Most config params are taken from the main TPGMM config, but some are overwritten.
        """

        n_components = (
            int(self.config.tpgmm.n_components * relative_duration)
            if self.config.demos_segmentation.components_prop_to_len
            else int(self.config.tpgmm.n_components / n_segments)
        )
        n_components = max(
            n_components,
            min_n_components,
            self.config.demos_segmentation.min_n_components,
        )

        return {
            "n_components": n_components,
            "fixed_first_component_n_steps": self.config.demos_segmentation.repeat_first_step
            or self.config.tpgmm.fixed_first_component_n_steps,
            "fixed_last_component_n_steps": self.config.demos_segmentation.repeat_final_step
            or self.config.tpgmm.fixed_last_component_n_steps,
            "add_action_component": self.config.tpgmm.add_action_component
            and not drop_action_component,
        }

    def calculate_segment_transition_probabilities(
        self,
        keep_time_dim: bool = True,
        keep_action_dim: bool = True,
        keep_rotation_dim: bool = True,
        models_are_sequential: bool = True,
        sigma_scale: float | None = None,
    ) -> tuple[np.ndarray, ...]:
        """
        For the sequence of segment models, calculate the transition probabilities
        between the segment. Assumes that the segment models are sequential, ie. there
        is no branching. (Can easily adapt to that - would need to compute transition
        prob between all pairs of segments).

        For state-action models only! Does not work for time-based models.

        NOTE: to use these probabilities, need to add them to the transition matrix and
        renormalize.
        """
        assert self._model_check()

        # assert self.model_type in MarkovTypes
        assert self.segment_frames is not None
        assert self.segment_gmms is not None

        # Get the common frames between the segments and the corresponding manifold idcs
        seg_frames: list[tuple[int]] = self.segment_frames

        pairwise_common_frames = tuple(
            sorted(set.intersection(set(f1), set(f2)))
            for f1, f2 in zip(seg_frames, seg_frames[1:])
        )

        idcs_of_common_frames_in_first_gmm = tuple(
            tuple(seg_frames[i].index(f) for f in common_frames)
            for i, common_frames in enumerate(pairwise_common_frames)
        )

        idcs_of_common_frames_in_second_gmm = tuple(
            tuple(seg_frames[i + 1].index(f) for f in common_frames)
            for i, common_frames in enumerate(pairwise_common_frames)
        )

        manifolds_per_frame = _get_rbd_manifolds_per_frame(
            self.config.tpgmm.position_only, self.config.tpgmm.add_action_component
        )

        # TODO: add global dims
        first_manifold_idcs = tuple(
            _get_rbd_manifold_indices(
                frames,
                time_based=self.config.tpgmm.add_time_component,
                manifolds_per_frame=manifolds_per_frame,
                keep_time_dim=keep_time_dim,
            )
            for frames in idcs_of_common_frames_in_first_gmm
        )

        second_manifold_idcs = tuple(
            _get_rbd_manifold_indices(
                f,
                time_based=self.config.tpgmm.add_time_component,
                manifolds_per_frame=manifolds_per_frame,
                keep_time_dim=keep_time_dim,
            )
            for f in idcs_of_common_frames_in_second_gmm
        )

        probs = tuple(
            hmm_transition_probabilities(
                g1.model,
                g2.model,
                i1,
                i2,
                drop_action_dim=not keep_action_dim,
                drop_rotation_dim=not keep_rotation_dim,
                includes_time=self.config.tpgmm.add_time_component and keep_time_dim,
                sigma_scale=sigma_scale,
                models_are_sequential=models_are_sequential,
            )
            for g1, g2, i1, i2 in zip(
                self.segment_gmms,
                self.segment_gmms[1:],
                first_manifold_idcs,
                second_manifold_idcs,
            )
        )

        if self.config.cascade.min_prob is not None:
            probs = tuple(np.maximum(p, self.config.cascade.min_prob) for p in probs)

        return probs

    def reconstruct(
        self,
        demos: Demos | None = None,
        use_ss: bool = False,
        dbg: bool = False,
        strategy: ReconstructionStrategy | None = None,
        time_based: bool | None = None,
        per_segment: bool = False,
    ) -> tuple[
        tuple[list[list[GMM]]], tuple[list[list[GMM]]], tuple[list[GMM]], tuple[Any]
    ]:
        if demos is not None:
            raise NotImplementedError(
                "Can currently only reconstruct the fitted demos as novel demos would "
                "need to be segmented and the frames selected first."
            )

        # if self.fitting_stage < FittingStage.EM_GMM:
        #     logger.error(
        #         f"Model not fitted yet. Fitting stage: {self.fitting_stage.name}."
        #     )
        #     raise RuntimeError("Model not fitted yet.")

        if strategy is None:
            # strategy = ReconstructionStrategy.GMR if self.model_type \
            #     is ModelType.GMM else ReconstructionStrategy.LQR_VITERBI
            strategy = ReconstructionStrategy.GMR
            logger.info(f"Selected reconstruction strategy {strategy}.")

        if time_based:
            assert (
                self.config.tpgmm.add_time_component
            ), "Need time-based model for time-based reconstruction."

        if time_based is None:
            time_based = self.config.tpgmm.add_time_component
            logger.info(
                f"Time-based reconstruction not specified. Auto selected {time_based}."
            )

        per_segment = per_segment or len(self.segment_gmms) == 1

        if per_segment:
            return self._reconstruct_per_segment(
                demos=demos,
                use_ss=use_ss,
                dbg=dbg,
                strategy=strategy,
                time_based=time_based,
            )
        else:
            # assert self.model_type in MarkovTypes, "No cascading for GMMs."
            if self.fitting_stage == FittingStage.EM_HMM:
                return self._cascade_segment_hmms(
                    demos=demos,
                    use_ss=use_ss,
                    dbg=dbg,
                    strategy=strategy,
                    time_based=time_based,
                )
            else:
                return self._cascade_segment_gmms(
                    demos=demos,
                    use_ss=use_ss,
                    dbg=dbg,
                    strategy=strategy,
                    time_based=time_based,
                )

    def _cascade_segment_hmms(
        self,
        demos: Demos | None = None,
        use_ss: bool = False,
        dbg: bool = False,
        strategy: ReconstructionStrategy | None = None,
        time_based: bool | None = None,
    ):
        if self.fitting_stage < FittingStage.EM_HMM:
            logger.error(
                f"Transition model not fitted yet. Fitting stage: {self.fitting_stage.name}."
            )
            raise RuntimeError("Model not fitted yet.")

        if demos is None:
            demos = self._demos

        fix_frames = self._fix_frames
        add_action_dim = self.config.tpgmm.add_action_component
        heal_time_variance = time_based and self.config.tpgmm.heal_time_variance

        segment_transition_probs = self.calculate_segment_transition_probabilities(
            keep_time_dim=self.config.cascade.kl_keep_time_dim,
            keep_action_dim=self.config.cascade.kl_keep_action_dim,
            keep_rotation_dim=self.config.cascade.kl_keep_rotation_dim,
            models_are_sequential=self.config.cascade.models_are_sequential,
            sigma_scale=self.config.cascade.kl_sigma_scale,
        )

        logger.info(
            f"Caculated segment transition probabilities: {segment_transition_probs}",
            filter=False,
        )

        if (np.concatenate(segment_transition_probs) < 0.05).any():
            logger.warning(
                "At least one segment transition prob below 5%. Can lead to problems."
                "Consider increasing the diag reg."
            )

        local_marginals = []
        trans_marginals = []
        joint_models = []

        for frame_idcs, segment_gmm in zip(self.segment_frames, self.segment_gmms):
            segment_frame_view = PartialFrameViewDemos(demos, list(frame_idcs))

            world_data, frame_trans, frame_quats = segment_gmm.get_gmr_data(
                use_ss=use_ss,
                time_based=time_based,
                dbg=dbg,
                demos=segment_frame_view,
            )

            loc_marg, trans_marg, joint = segment_gmm.get_marginals_and_joint(
                fix_frames=fix_frames,
                time_based=time_based,
                heal_time_variance=heal_time_variance,
                frame_trans=frame_trans,
                frame_quats=frame_quats,
            )

            local_marginals.append(loc_marg)
            trans_marginals.append(trans_marg)
            joint_models.append(joint)  # per segment and trajectory

        assert (
            fix_frames
        ), "Need to adapt the next two statements (joint_models, cascaded_hmms)"
        joint_models = tuple(zip(*joint_models))  # per trajectory and segment

        trans_marginals = tuple(zip(*trans_marginals))  # per trajectory, segment, frame
        trans_marginals_dict = tuple(dict() for _ in range(len(trans_marginals)))
        for dic, traj in zip(trans_marginals_dict, trans_marginals):
            for f_list, m_list in zip(self.segment_frames, traj):
                for f, m in zip(f_list, m_list):
                    if f not in dic:
                        dic[f] = []
                    dic[f].append(m)

        trans_marginals_joint = tuple(
            tuple(
                rbd.statistics.ModelList(traj[f]) if f in traj else None
                for f in range(self.n_frames)
            )
            for traj in trans_marginals_dict
        )

        cascaded_hmms = tuple(
            rbd.statistics.HMMCascade(
                segment_models=segment_models,
                transition_probs=segment_transition_probs,
            )
            for segment_models in joint_models
        )

        if strategy is ReconstructionStrategy.GMR:
            ret = self._gmr(
                trajs=world_data,
                joint_models=cascaded_hmms,
                fix_frames=fix_frames,
                time_based=time_based,
                with_action_dim=add_action_dim,
            )
        elif strategy is ReconstructionStrategy.LQR_VITERBI:
            raise NotImplementedError("LQR-Viterbi not implemented for ATPGMM yet.")
        else:
            raise NotImplementedError(
                "Unexpected reconstruction strategy: {strategy}".format(
                    strategy=strategy
                )
            )

        return (
            local_marginals,
            trans_marginals,
            trans_marginals_joint,
            joint_models,
            cascaded_hmms,
            ret,
        )

    def _cascade_segment_gmms(
        self,
        demos: Demos | None = None,
        use_ss: bool = False,
        dbg: bool = False,
        strategy: ReconstructionStrategy | None = None,
        time_based: bool | None = None,
    ):
        if self.fitting_stage < FittingStage.EM_GMM:
            raise RuntimeError("Model not fitted yet.")

        if demos is None:
            demos = self._demos

        fix_frames = self._fix_frames
        add_action_dim = self.config.tpgmm.add_action_component
        heal_time_variance = time_based and self.config.tpgmm.heal_time_variance

        local_marginals = []
        trans_marginals = []
        joint_models = []

        for frame_idcs, segment_gmm in zip(self.segment_frames, self.segment_gmms):
            segment_frame_view = PartialFrameViewDemos(demos, list(frame_idcs))

            world_data, frame_trans, frame_quats = segment_gmm.get_gmr_data(
                use_ss=use_ss,
                time_based=time_based,
                dbg=dbg,
                demos=segment_frame_view,
            )

            loc_marg, trans_marg, joint = segment_gmm.get_marginals_and_joint(
                fix_frames=fix_frames,
                time_based=time_based,
                heal_time_variance=heal_time_variance,
                frame_trans=frame_trans,
                frame_quats=frame_quats,
            )

            local_marginals.append(loc_marg)
            trans_marginals.append(trans_marg)
            joint_models.append(joint)  # per segment and trajectory

        assert (
            fix_frames
        ), "Need to adapt the next two statements (joint_models, cascaded_gmms)"
        joint_models = tuple(zip(*joint_models))  # per trajectory and segment

        trans_marginals = tuple(zip(*trans_marginals))  # per trajectory, segment, frame
        trans_marginals_dict = tuple(dict() for _ in range(len(trans_marginals)))
        for dic, traj in zip(trans_marginals_dict, trans_marginals):
            for f_list, m_list in zip(self.segment_frames, traj):
                for f, m in zip(f_list, m_list):
                    if f not in dic:
                        dic[f] = []
                    dic[f].append(m)

        trans_marginals_joint = tuple(
            tuple(
                rbd.statistics.ModelList(traj[f]) if f in traj else None
                for f in range(self.n_frames)
            )
            for traj in trans_marginals_dict
        )

        cascaded_gmms = tuple(
            rbd.statistics.GMMCascade(
                segment_models=segment_models,
                prior_weights=self._segment_lengths,
            )
            for segment_models in joint_models
        )

        if strategy is ReconstructionStrategy.GMR:
            ret = self._gmr(
                trajs=world_data,
                joint_models=cascaded_gmms,
                fix_frames=fix_frames,
                time_based=time_based,
                with_action_dim=add_action_dim,
            )
        elif strategy is ReconstructionStrategy.LQR_VITERBI:
            raise NotImplementedError("LQR-Viterbi not implemented for ATPGMM yet.")
        else:
            raise NotImplementedError(
                "Unexpected reconstruction strategy: {strategy}".format(
                    strategy=strategy
                )
            )

        return (
            local_marginals,
            trans_marginals,
            trans_marginals_joint,
            joint_models,
            cascaded_gmms,
            ret,
        )

    def reset_episode(self):
        """
        Reset the episode for online prediction. Ie. next prediction is treated as the
        first step of a new episode.
        """
        self._online_first_step = True
        self._online_active_segment = 0

    def get_frame_marginals(
        self, time_based: bool, models: tuple[GMM, ...] | None = None
    ) -> tuple[tuple[GMM, ...], ...]:
        """
        Split up the joint segment GMMs into per-frame marginals each.
        To be used for online prediction. In reconstruction, the segment GMMs should be
        used directly.
        """
        if models is None:
            assert self._model_check()
            models = self.segment_gmms

        return tuple(
            model.get_frame_marginals(time_based=time_based) for model in models
        )

    @cached_property
    def _segment_t_delta(self) -> tuple[float, ...]:
        return tuple(
            s._demos.relative_duration / s._demos.ss_len for s in self.segment_gmms
        )

    @cached_property
    def _segment_start_relative(self) -> tuple[float, ...]:
        return tuple(s.relative_start_time for s in self._demos_segments)

    @cached_property
    def _segment_stop_relative(self) -> tuple[float, ...]:
        return tuple(s.relative_stop_time for s in self._demos_segments)

    @property
    def _t_delta(self) -> float:
        if self._online_hmm_cascade is None:  # naive time sequencing
            idx = self._online_active_segment
            assert self.segment_gmms[idx]._demos is not None

            return 1 / self.segment_gmms[idx]._demos.ss_len
        else:
            if self.fitting_stage == FittingStage.EM_HMM:
                # Using the HMM cascade: weight sum of segment time delta by activation
                activation = self._online_hmm_cascade._alpha_tmp
                start_idcs = self._online_hmm_cascade.segment_start_idcs

                segment_activations = tuple(
                    np.sum(a) for a in np.split(activation, start_idcs)
                )
            else:
                # GMM cascade: weight by the relative segment length for now
                # TODO: consider using the segment activations?
                segment_activations = self._segment_lengths

            weighted_sum = sum(
                (a * d for a, d in zip(segment_activations, self._segment_t_delta))
            )

            return weighted_sum

    def _online_step_time(
        self, t_curr: float | np.ndarray, time_scale: float = 1.0
    ) -> float | np.ndarray:
        t_next = t_curr + time_scale * self._t_delta

        idx = self._online_active_segment

        # Naive time sequencing: just switch to the next segment when the time is up.
        if (
            self._online_hmm_cascade is None
            and t_next >= self._segment_stop_relative[idx]
            and idx < len(self.segment_gmms) - 1
        ):
            self._online_active_segment += 1
            self._online_first_step = True

        return t_next

    def online_predict(
        self,
        input_data: np.ndarray,
        frame_trans: list[np.ndarray] | None,
        frame_quats: list[np.ndarray] | None,
        time_based: bool = False,
        local_marginals: tuple[tuple[GMM]] | None = None,
        strategy: ReconstructionStrategy = ReconstructionStrategy.GMR,
        heal_time_variance: bool = False,
        per_segment: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._fix_frames:
            raise NotImplementedError(
                "Online prediction of ATPGMM only implemented for fixed frames. "
                "Currently batch-computing the joint models in the beginning to "
                "prevent lags due to computing the joint model of the current "
                "segment on the fly. "
                "Could store the marginals and transform them on-demand (maybe "
                "in a separate thread to prevent lags, or pause the execution)."
            )
        assert self._fix_frames is not None

        per_segment = per_segment or len(self.segment_gmms) == 1

        if frame_trans is not None:  # intial time step or change in frame position
            assert frame_quats is not None and local_marginals is not None

            (
                self._online_joint_models,
                self._online_trans_margs_joint,
            ) = self._make_segment_joint_models(
                frame_trans=frame_trans,
                frame_quats=frame_quats,
                local_marginals=local_marginals,
                heal_time_variance=heal_time_variance,
                time_based=time_based,
            )

            if not per_segment:
                if self.fitting_stage == FittingStage.EM_HMM:
                    transition_probs = self.calculate_segment_transition_probabilities(
                        keep_time_dim=self.config.cascade.kl_keep_time_dim,
                        keep_action_dim=self.config.cascade.kl_keep_action_dim,
                        keep_rotation_dim=self.config.cascade.kl_keep_rotation_dim,
                        models_are_sequential=self.config.cascade.models_are_sequential,
                    )
                    logger.info(
                        f"Calculated segment transition probabilities: {transition_probs}",
                        filter=False,
                    )
                    self._online_hmm_cascade = rbd.statistics.HMMCascade(
                        segment_models=self._online_joint_models,
                        transition_probs=transition_probs,
                    )
                else:
                    self._online_hmm_cascade = rbd.statistics.GMMCascade(
                        segment_models=self._online_joint_models,
                        prior_weights=self._segment_lengths,
                    )
        if per_segment:
            assert time_based, "For State-Action need to use sequencing of models."
            ret = self._online_predict_per_segment(
                input_data=input_data,
                strategy=strategy,
                time_based=time_based,
            )
        else:
            ret = self._online_cascade_segment_hmms(
                input_data=input_data,
                strategy=strategy,
                time_based=time_based,
            )

        self._online_first_step = False

        return ret

    def batch_predict(
        self,
        input_data: np.ndarray,
        frame_trans: list[np.ndarray] | None,
        frame_quats: list[np.ndarray] | None,
        time_based: bool = False,
        local_marginals: tuple[tuple[GMM]] | None = None,
        strategy: ReconstructionStrategy = ReconstructionStrategy.GMR,
        heal_time_variance: bool = True,
        per_segment: bool = False,
    ) -> np.ndarray:
        """
        Batch predict the given input data using the segment models. Akin to
        online_predict, but over a full trajectory.
        Assumes that frames are fixed -> given in the same format as in online_predict.
        """

        predictions = tuple(
            self.online_predict(
                input_data=input_data[i],
                frame_trans=frame_trans,
                frame_quats=frame_quats,
                time_based=time_based,
                local_marginals=local_marginals,
                strategy=strategy,
                heal_time_variance=heal_time_variance,
                per_segment=per_segment,
            )
            for i in range(input_data.shape[0])
        )

        return np.stack(predictions)

    def _make_segment_joint_models(
        self,
        frame_trans: list[np.ndarray] | None,
        frame_quats: list[np.ndarray] | None,
        local_marginals: tuple[tuple[GMM]] | None = None,
        time_based: bool = True,
        heal_time_variance: bool = True,
    ) -> tuple[tuple[GMM, ...], tuple[rbd.statistics.ModelList, ...]]:
        joint_models = []
        trans_marginals = []

        for i, margs in enumerate(local_marginals):
            selected_trans = [frame_trans[j] for j in self.segment_frames[i]]
            selected_quats = [frame_quats[j] for j in self.segment_frames[i]]

            joint_model, trans_margs = self.segment_gmms[i].make_joint_model(
                frame_trans=selected_trans,
                frame_quats=selected_quats,
                time_based=time_based,
                local_marginals=margs,
                heal_time_variance=heal_time_variance,
                use_riemann=self.use_riemann,
            )
            joint_models.append(joint_model)
            trans_marginals.append(trans_margs)

        trans_marg_dict = dict()
        for f_list, m_list in zip(self.segment_frames, trans_marginals):
            for f, m in zip(f_list, m_list):
                if f not in trans_marg_dict:
                    trans_marg_dict[f] = []
                trans_marg_dict[f].append(m)

        trans_marg_joint = tuple(
            (
                rbd.statistics.ModelList(trans_marg_dict[f])
                if f in trans_marg_dict
                else None
            )
            for f in range(self.n_frames)
        )

        return tuple(joint_models), trans_marg_joint

    def _online_predict_per_segment(
        self,
        input_data: np.ndarray,
        strategy: ReconstructionStrategy = ReconstructionStrategy.GMR,
        time_based: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not time_based:
            raise NotImplementedError("No state-based segment-wise prediction.")

        idx = self._online_active_segment

        if strategy is ReconstructionStrategy.GMR:
            prediction, extras = self.segment_gmms[idx]._online_gmr(
                input_data=input_data,
                joint_model=self._online_joint_models[idx],
                fix_frames=self._fix_frames,
                time_based=True,
                first_step=self._online_first_step,
            )
        else:
            raise NotImplementedError

        return prediction, extras

    def _online_cascade_segment_hmms(
        self,
        input_data: np.ndarray,
        strategy: ReconstructionStrategy = ReconstructionStrategy.GMR,
        time_based: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if strategy is ReconstructionStrategy.GMR:
            prediction, extras = self._online_gmr(
                input_data=input_data,
                joint_model=self._online_hmm_cascade,
                fix_frames=self._fix_frames,
                time_based=time_based,
                first_step=self._online_first_step,
            )
        else:
            raise NotImplementedError

        return prediction, extras

    def plot_model(
        self,
        plot_traj=True,
        plot_gaussians=True,
        scatter=False,
        rotations_raw=False,
        gaussian_mean_only=False,
        gaussian_cmap="Oranges",
        time_based=None,
        xdx_based=False,
        mean_as_base=True,
        annotate_gaussians=True,
        annotate_trajs=False,
        title=None,
        plot_derivatives=False,
        plot_traj_means=False,
        per_segment: bool = False,
        size=None,
    ):
        if per_segment:
            for seg_model in self.segment_gmms:
                seg_model.plot_model(
                    plot_traj=plot_traj,
                    plot_gaussians=plot_gaussians,
                    scatter=scatter,
                    rotations_raw=rotations_raw,
                    gaussian_mean_only=gaussian_mean_only,
                    gaussian_cmap=gaussian_cmap,
                    time_based=time_based,
                    xdx_based=xdx_based,
                    mean_as_base=mean_as_base,
                    annotate_gaussians=annotate_gaussians,
                    annotate_trajs=annotate_trajs,
                    title=title,
                    plot_derivatives=plot_derivatives,
                    plot_traj_means=plot_traj_means,
                    model=None,
                )
        else:
            if not self._model_check():
                return

            assert self._demos is not None

            if time_based is None:
                logger.info(
                    "Did not specify time_based, deciding automatically.", filter=False
                )
                time_based = self.config.tpgmm.add_time_component

            segment_data = [
                m._split_and_tangent_project_frame_data(
                    model=m.model,
                    data=None,
                    time_based=time_based,
                    xdx_based=xdx_based,
                    mean_as_base=mean_as_base,
                    rotations_raw=rotations_raw,
                )
                for m in self.segment_gmms
            ]

            all_frames = sorted(set.union(*[set(s) for s in self.segment_frames]))
            n_frames = len(all_frames)

            joint_data = []

            n_plot_rows = len(segment_data[0].dims)

            for i in range(n_plot_rows):
                dim_segments = [seg.dims[i] for seg in segment_data]

                n_time_steps = [d.data.shape[1] for d in dim_segments]
                n_time_total = sum(n_time_steps)
                time_starts = np.cumsum([0] + n_time_steps[:-1])
                time_stops = np.cumsum(n_time_steps)

                n_gaus_per_component = [d.mu.shape[-2] for d in dim_segments]
                n_gaus_total = sum(n_gaus_per_component)

                # labels to which segment each gaussian belongs
                segment_per_gaussian = np.concatenate(
                    [np.repeat(g, n) for g, n in enumerate(n_gaus_per_component)]
                )

                if len(dim_segments[0].data.shape) == 2:  # global data
                    dim_data = np.concatenate([d.data for d in dim_segments], axis=1)
                    per_frame = False

                    mus = np.concatenate([d.mu for d in dim_segments], axis=0)
                    sigmas = np.concatenate([d.sigma for d in dim_segments], axis=0)

                else:  # per frame data
                    n_trajs = dim_segments[0].data.shape[0]
                    n_dims = dim_segments[0].data.shape[3]

                    # initialize stacked data with NaNs to account for missing frames
                    dim_data = np.empty((n_trajs, n_time_total, n_frames, n_dims))
                    dim_data.fill(np.nan)

                    for s, t, (j, d) in zip(
                        time_starts, time_stops, enumerate(dim_segments)
                    ):
                        for k, frame_no in enumerate(self.segment_frames[j]):
                            global_frame_no = all_frames.index(frame_no)
                            dim_data[:, s:t, global_frame_no, :] = d.data[:, :, k, :]

                    per_frame = True

                    man_dim = dim_segments[0].mu.shape[-1]  # same as n_dims
                    tan_dim = dim_segments[0].sigma.shape[-1]

                    gaus_seg_starts = np.cumsum([0] + n_gaus_per_component[:-1])
                    gaus_seg_stops = np.cumsum(n_gaus_per_component)

                    mus = np.empty((n_frames, n_gaus_total, man_dim))
                    mus.fill(np.nan)
                    sigmas = np.empty((n_frames, n_gaus_total, tan_dim, tan_dim))
                    sigmas.fill(np.nan)

                    for s, t, (j, d) in zip(
                        gaus_seg_starts, gaus_seg_stops, enumerate(dim_segments)
                    ):
                        for k, frame_no in enumerate(self.segment_frames[j]):
                            global_frame_no = all_frames.index(frame_no)
                            local_mu = d.mu[k, ...]
                            local_sigma = d.sigma[k, ...]
                            mus[global_frame_no, s:t, :] = local_mu
                            sigmas[global_frame_no, s:t, ...] = local_sigma

                # TODO: fix the data. Assignement looks wrong.

                joint_data.append(
                    SingleDimPlotData(
                        data=dim_data,
                        name=segment_data[0].dims[i].name,
                        per_frame=per_frame,
                        manifold=segment_data[0].dims[i].manifold,
                        mu=mus,
                        sigma=sigmas,
                        gauss_labels=segment_per_gaussian,
                        base=segment_data[0].dims[i].base,
                        on_tangent=segment_data[0].dims[i].on_tangent,
                    )
                )

            frame_names = tuple(self._demos.frame_names[i] for i in all_frames)

            joint_container = TPGMMPlotData(frame_names=frame_names, dims=joint_data)

            segment_borders = [n / n_time_total for n in time_starts]

            if xdx_based:
                plot_gmm_xdx_based(
                    pos_per_frame=pos,
                    rot_per_frame=rot,
                    pos_delta_per_frame=pos_delta,
                    rot_delta_per_frame=rot_delta,
                    model=model,
                    frame_names=self._demos.frame_names,
                    plot_traj=plot_traj,
                    plot_gaussians=plot_gaussians,
                    scatter=scatter,
                    rot_on_tangent=self.use_riemann,
                    gaussian_mean_only=gaussian_mean_only,
                    cmap=gaussian_cmap,
                    annotate_gaussians=annotate_gaussians,
                    title=title,
                    rot_base=rot_bases_per_frame,
                    model_includes_time=self.config.tpgmm.add_time_component,
                    size=size,
                )
            elif time_based:
                plot_gmm_time_based(
                    container=joint_container,
                    plot_traj=plot_traj,
                    plot_gaussians=plot_gaussians,
                    rot_on_tangent=self.use_riemann,
                    gaussian_mean_only=gaussian_mean_only,
                    annotate_gaussians=annotate_gaussians,
                    annotate_trajs=annotate_trajs,
                    title=title,
                    plot_derivatives=plot_derivatives,
                    plot_traj_means=plot_traj_means,
                    component_borders=segment_borders,
                    size=size,
                )
            else:
                plot_gmm_trajs_3d(
                    container=joint_container,
                    plot_traj=plot_traj,
                    plot_gaussians=plot_gaussians,
                    scatter=scatter,
                    gaussian_mean_only=gaussian_mean_only,
                    cmap_gauss=gaussian_cmap,
                    annotate_gaussians=annotate_gaussians,
                    title=title,
                    size=size,
                )

    def plot_reconstructions(
        self,
        marginals: tuple[tuple[tuple[GMM]]],
        joint_models: tuple[tuple[GMM]],
        reconstructions: tuple[list[np.ndarray]],
        original_trajectories: tuple[list[np.ndarray]],
        demos: Demos | None = None,
        time_based: bool | None = None,
        per_segment: bool = False,
        **kwargs,
    ) -> None:
        if per_segment:
            assert (
                type(joint_models[0]) is not rbd.statistics.HMMCascade
            ), "Use joint reconstruction for cascaded models."
            self._plot_reconstructions_per_segment(
                marginals=marginals,
                joint_models=joint_models,
                reconstructions=reconstructions,
                original_trajectories=original_trajectories,
                time_based=time_based,
                **kwargs,
            )
        else:
            self._plot_reconstructions_jointly(
                marginals=marginals,
                joint_models=joint_models,
                reconstructions=reconstructions,
                original_trajectories=original_trajectories,
                time_based=time_based,
                **kwargs,
            )

    def _plot_reconstructions_per_segment(
        self,
        marginals: tuple[tuple[tuple[GMM]]],
        joint_models: tuple[tuple[GMM]],
        reconstructions: tuple[list[np.ndarray]],
        original_trajectories: tuple[list[np.ndarray]],
        demos: Demos | None = None,
        time_based: bool | None = None,
        **kwargs,
    ) -> None:
        for model, marg, joint, rec, orig, data in zip(
            self.segment_gmms,
            marginals,
            joint_models,
            reconstructions,
            original_trajectories,
            self.segment_frame_views,
        ):
            model.plot_reconstructions(
                marginals=marg,
                joint_models=joint,
                reconstructions=rec,
                original_trajectories=orig,
                demos=data,
                time_based=time_based,
                **kwargs,
            )

    def _plot_reconstructions_jointly(
        self,
        marginals: tuple[tuple[tuple[GMM]]],
        joint_models: tuple[tuple[GMM]] | tuple[rbd.statistics.HMMCascade],
        reconstructions: tuple[list[np.ndarray]],
        original_trajectories: tuple[list[np.ndarray]],
        demos: Demos | None = None,
        time_based: bool | None = None,
        frame_orig_wquats: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        if time_based is None:
            logger.info(
                "Did not specify time_based, deciding automatically.", filter=False
            )
            time_based = self.config.tpgmm.add_time_component

        if demos is None:
            demos = self._demos
        else:
            assert demos == self._demos

        reconstruction_is_per_segment = not isinstance(reconstructions[0], np.ndarray)

        if reconstruction_is_per_segment:
            joint_models = joint_models[0]

        marginals = tuple(
            tuple(traj[f] for f in self._used_frames) for traj in marginals
        )

        if time_based:
            if frame_orig_wquats is None:
                logger.info("Taking frame origins from demos.", filter=False)
                frame_orig_wquats = demos._frame_origins_fixed_wquats.numpy()

            frame_orig_wquats = frame_orig_wquats[:, self._used_frames]

            frame_names = [demos.frame_names[i] for i in self._used_frames]

            frame_origs_log = np.stack(
                [
                    j.np_to_manifold_to_np(f, i_in=[1, 2])
                    for j, f in zip(joint_models, frame_orig_wquats)
                ]
            )

            if reconstruction_is_per_segment:
                seg_lens = tuple(
                    tuple(len(r) for r in recs) for recs in zip(*reconstructions)
                )
            else:
                seg_lens = tuple(s.traj_lens for s in self._demos_segments)
                seg_lens = tuple(tuple(lens) for lens in zip(*seg_lens))
            seg_starts = tuple(np.cumsum([0] + list(lens[:-1])) for lens in seg_lens)
            traj_lens = tuple(np.sum(lens) for lens in seg_lens)
            seg_borders = tuple(
                tuple(starts / tlen) for starts, tlen in zip(seg_starts, traj_lens)
            )

        else:
            frame_origins = demos._frame_origins_fixed

            frame_origins = frame_origins[:, self._used_frames]

            frame_names = [demos.frame_names[i] for i in self._used_frames]

        if (
            "plot_gaussians" in kwargs
            and kwargs["plot_gaussians"]
            and reconstruction_is_per_segment
        ):
            logger.info(
                "Cannot plot gaussians for per segment reconstruction yet.",
                filter=False,
            )
            kwargs["plot_gaussians"] = False

        if reconstruction_is_per_segment:
            # Flatten over segments
            reconstructions_cat = tuple(
                np.concatenate([r for r in rec], axis=0)
                for rec in zip(*reconstructions)
            )
            original_trajectories_cat = tuple(
                np.concatenate([t for t in traj], axis=0)
                for traj in zip(*original_trajectories)
            )
        else:
            reconstructions_cat = reconstructions
            original_trajectories_cat = original_trajectories

        if time_based:
            plot_reconstructions_time_based(
                marginals,
                joint_models,
                reconstructions_cat,
                original_trajectories_cat,
                frame_origs_log,
                frame_names,
                includes_rotations=not self.config.tpgmm.position_only,
                includes_time=self.config.tpgmm.add_time_component,
                includes_actions=self.config.tpgmm.add_action_component,
                includes_action_magnitudes=self.config.tpgmm.action_with_magnitude,
                includes_gripper_actions=self.config.tpgmm.add_gripper_action,
                component_borders=seg_borders,
                **kwargs,
            )
        else:
            plot_reconstructions_3d(
                marginals,
                joint_models,
                reconstructions_cat,
                original_trajectories_cat,
                frame_origins,
                frame_names,
                includes_time=self.config.tpgmm.add_time_component,
                includes_rotations=not self.config.tpgmm.position_only,
                **kwargs,
            )

    def plot_hmm_transition_matrix(self):
        for seg_model in self.segment_gmms:
            seg_model.plot_hmm_transition_matrix()

    def from_disk(self, file_name: str, force_config_overwrite: bool = False) -> None:
        logger.info("Loading model:")

        with open(file_name, "rb") as f:
            ckpt = pickle.load(f)

        if self.config != ckpt.config:
            diff_string = recursive_compare_dataclass(ckpt.config, self.config)
            logger.error("Config mismatch\n" + diff_string)
            if force_config_overwrite:
                logger.warning(
                    "Overwriting config. This can lead to unexpected errors."
                )
                self.config = ckpt.config
            else:
                raise ValueError("Config mismatch")

        self._fix_frames = ckpt._fix_frames

        self.segment_gmms = ckpt.segment_gmms
        self._demos = ckpt._demos
        self._demos_segments = ckpt._demos_segments
        self.segment_frames = ckpt.segment_frames
        self.segment_frame_views = ckpt.segment_frame_views
