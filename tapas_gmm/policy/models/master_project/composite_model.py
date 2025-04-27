from typing import List, Union
import numpy as np
from tapas_gmm.policy.models.tpgmm import AutoTPGMM


class CompositeModel:
    def __init__(
        self,
        task_models: List[Union[AutoTPGMM, "CompositeModel"]],
        transition_model: AutoTPGMM,
        original_model: bool = False,
    ):
        self.models: List[Union[AutoTPGMM, "CompositeModel"]] = task_models
        self.transition_model: AutoTPGMM = transition_model

        # Check that the model isnt empty. That indicates a mistake.
        if len(self.models) < 2:
            raise ValueError("Composite model must contain at least one task model.")

        self.current_model = None

        self.current_model_idx = 0
        self.current_model_transition = False

    def next_model(self):
        self.current_model_idx += 1
        if self.current_model_transition:
            self.current_model = self.models[self.current_model_idx]
            self.current_model_transition = False
        else:
            self.current_model = self.transition_model
            self.current_model_transition = True

    def predict(self, state: np.ndarray) -> np.ndarray:
        # If the model has not started, use the first model
        if self.current_model_idx == 0:
            self.current_model = self.models[0]

        # Predict the next action
        return self.current_model.predict(state)


class SelfLearnedModel(CompositeModel):
    def __init__(
        self,
        task_models: List[AutoTPGMM],
        transition_model: AutoTPGMM = None,
        original_model: bool = False,
    ):
        super().__init__(task_models, transition_model, original_model)
        self.current_model = self.models[0]

    def next_model(self):
        self.current_model_idx += 1
        if self.current_model_transition:
            self.current_model = self.models[self.current_model_idx]
            self.current_model_transition = False
        else:
            self.current_model = self.transition_model
            self.current_model_transition = True

    def predict(self, state: np.ndarray) -> np.ndarray:
        # Predict the next action
        return self.current_model.predict(state)


class TapasWrapperModel(CompositeModel):
    def __init__(self, model: AutoTPGMM):
        self.current_model = model

    def predict(self, state: np.ndarray) -> np.ndarray:
        # Predict the next action
        return self.current_model.predict(state)
