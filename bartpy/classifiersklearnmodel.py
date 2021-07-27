from copy import deepcopy
from typing import List, Callable, Mapping, Union, Optional

import numpy as np
from sklearnmodel import SklearnModel
from bartpy.initializers.initializer import Initializer
from bartpy.samplers.leafnode import LeafNodeSampler
from bartpy.samplers.modelsampler import ModelSampler
from bartpy.samplers.schedule import SampleSchedule
from bartpy.samplers.sigma import ConstantSigmaSampler
from bartpy.samplers.treemutation import TreeMutationSampler
from bartpy.samplers.unconstrainedtree.treemutation import get_tree_sampler

def run_chain(model: 'SklearnModel', X: np.ndarray, y: np.ndarray):
    """
    Run a single chain for a model
    Primarily used as a building block for constructing a parallel run of multiple chains
    """
    model.model = model._construct_model(X, y)
    return model.sampler.samples(model.model,
                                 model.n_samples,
                                 model.n_burn,
                                 model.thin,
                                 model.store_in_sample_predictions,
                                 model.store_acceptance_trace)


def delayed_run_chain():
    return run_chain

# See this for an implementation of a BART Classifier Wrapper: https://github.com/kapelner/bartMachine/blob/master/src/bartMachine/bartMachineClassificationMultThread.java
# See this for the definition of the GibbsSample for classification: https://github.com/kapelner/bartMachine/blob/586f55bf4de5290793474f55a928695fbd508ac8/src/bartMachine/bartMachineClassification.java#L14
# PS: Evaluate function should use a classification threshold (ex: classification_rule)
class ClassifierSklearnModel(SklearnModel):
    """
    The main access point to building BART models for classification in BartPy

    Parameters
    ----------
    n_trees: int
        the number of trees to use, more trees will make a smoother fit, but slow training and fitting
    n_chains: int
        the number of independent chains to run
        more chains will improve the quality of the samples, but will require more computation
    sigma_a: float
        shape parameter of the prior on sigma
    sigma_b: float
        scale parameter of the prior on sigma
    n_samples: int
        how many recorded samples to take
    n_burn: int
        how many samples to run without recording to reach convergence
    thin: float
        percentage of samples to store.
        use this to save memory when running large models
    p_grow: float
        probability of choosing a grow mutation in tree mutation sampling
    p_prune: float
        probability of choosing a prune mutation in tree mutation sampling
    alpha: float
        prior parameter on tree structure
    beta: float
        prior parameter on tree structure
    store_in_sample_predictions: bool
        whether to store full prediction samples
        set to False if you don't need in sample results - saves a lot of memory
    store_acceptance_trace: bool
        whether to store acceptance rates of the gibbs samples
        unless you're very memory constrained, you wouldn't want to set this to false
        useful for diagnostics
    tree_sampler: TreeMutationSampler
        Method of sampling used on trees
        defaults to `bartpy.samplers.unconstrainedtree`
    initializer: Initializer
        Class that handles the initialization of tree structure and leaf values
    n_jobs: int
        how many cores to use when computing MCMC samples
        set to `-1` to use all cores
    """

    def __init__(self,
                 n_trees: int = 200,
                 n_chains: int = 4,
                 sigma_a: float = 0.001,
                 sigma_b: float = 0.001,
                 n_samples: int = 200,
                 n_burn: int = 200,
                 thin: float = 0.1,
                 alpha: float = 0.95,
                 beta: float = 2.,
                 store_in_sample_predictions: bool=False,
                 store_acceptance_trace: bool=False,
                 tree_sampler: TreeMutationSampler=get_tree_sampler(0.5, 0.5),
                 initializer: Optional[Initializer]=None,
                 n_jobs=-1):

        SklearnModel.__init__(self,
            n_trees, n_chains, sigma_a, sigma_b, n_samples,
            n_burn, thin, alpha, beta, store_in_sample_predictions, 
            store_acceptance_trace, tree_sampler, initializer, n_jobs)

        self.schedule = SampleSchedule(self.tree_sampler, LeafNodeSampler(), ConstantSigmaSampler())
        self.sampler = ModelSampler(self.schedule)

    # @staticmethod
    # def _combine_chains(extract: List[Chain]) -> Chain:
    #     keys = list(extract[0].keys())
    #     combined = {}
    #     for key in keys:
    #         combined[key] = np.concatenate([chain[key] for chain in extract], axis=0)
    #     return combined

    # @staticmethod
    # def _convert_covariates_to_data(X: np.ndarray, y: np.ndarray) -> Data:
    #     from copy import deepcopy
    #     if type(X) == pd.DataFrame:
    #         X: pd.DataFrame = X
    #         X = X.values
    #     return Data(deepcopy(X), deepcopy(y), normalize=True)

    # @property
    # def model_samples(self) -> List[Model]:
    #     """
    #     Array of the model as it was after each sample.
    #     Useful for examining for:

    #      - examining the state of trees, nodes and sigma throughout the sampling
    #      - out of sample prediction

    #     Returns None if the model hasn't been fit

    #     Returns
    #     -------
    #     List[Model]
    #     """
    #     return self._model_samples

    # @property
    # def acceptance_trace(self) -> List[Mapping[str, float]]:
    #     """
    #     List of Mappings from variable name to acceptance rates

    #     Each entry is the acceptance rate of the variable in each iteration of the model

    #     Returns
    #     -------
    #     List[Mapping[str, float]]
    #     """
    #     return self._acceptance_trace

    # @property
    # def prediction_samples(self) -> np.ndarray:
    #     """
    #     Matrix of prediction samples at each point in sampling
    #     Useful for assessing convergence, calculating point estimates etc.

    #     Returns
    #     -------
    #     np.ndarray
    #         prediction samples with dimensionality n_samples * n_points
    #     """
    #     return self.prediction_samples
