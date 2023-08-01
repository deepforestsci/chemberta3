import deepchem as dc
from deepchem.models.torch_models import InfoGraphModel

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import List, Optional, Dict


def load_infograph(num_feat: int, edge_dim: int):
    """Load an InfoGraph model.
    Parameters
    ----------
    num_feat: int
        Number of atom features
    edge_dim: int
        Number of edge features

    Returns
    -------
    model: dc.models.torch_models.modular.ModularTorchModel
        Loaded model.
    """
    # NOTE: cannot pass in `self.loss` because of how InfoGraphModel is constructed
    # TODO: fix this (only regression currently supported)
    model = InfoGraphModel(
        num_feat,
        edge_dim,
        64,
        use_unsup_loss=False,
        separate_encoder=True,
    )
    if checkpoint_path is not None:
        model.load_pretrained_components(checkpoint=checkpoint_path)
    return model

def load_random_forest(task: str, hyperparams: Optional[Dict] = None, checkpoint_path: Optional[str] = None, **kwargs):
    """Loads a random forest model

    Parameters
    ----------
    task: str
        Type of dataset (classification or regression)
    hyperparams: Dict
        Parameters of the random forest classifier model
    checkpoint_path: str
        Path to model checkpoint

    Returns
    -------
    model: dc.models.SklearnModel
    """
    assert task in ['classification', 'regression'], 'Dataset task should be either classification or regression'
    if task == 'classification':
        base_model = RandomForestClassifier
    elif task == 'regression':
        base_model = RandomForestRegressor

    if checkpoint_path is not None:
        model = dc.models.SklearnModel(base_model(), model_dir=checkpoint_path)
        model.reload()
        return model

    if hyperparams is not None:
        model = dc.models.SklearnModel(base_model(**hyperparams))
    else:
        model = dc.models.SklearnModel(base_model())
    return model
