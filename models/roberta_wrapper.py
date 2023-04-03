from deepchem.models.torch_models.modular import ModularTorchModel
from transformers import RobertaForMaskedLM
from typing import Dict, Any


class RobertaForMaskedLMModel(ModularTorchModel):
    """Wrapper for huggingface RobertaForMaskedLM model."""

    def __init__(self, model, **kwargs) -> None:
        for attr in dir(model):
            try:
                setattr(self, attr, getattr(model, attr))
            except AttributeError:
                continue

        self.source_model = model
        self.config = self.source_model.config

        self.components = self.build_components()
        self.model = self.build_model()
        super().__init__(self.model, self.components, **kwargs)

    def build_components(self) -> Dict[str, Any]:
        return {'model': self.roberta, 'prediction_head': self.lm_head}

    def build_model(self) -> RobertaForMaskedLM:
        return RobertaForMaskedLM(config=self.config)
