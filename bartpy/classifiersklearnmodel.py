from typing import Optional, Union

import numpy as np
import pandas as pd

from bartpy.data import Data
from bartpy.model import ClassifierModel
from bartpy.sigma import Sigma
from bartpy.sklearnmodel import SklearnModel
from bartpy.initializers.initializer import Initializer
from bartpy.samplers.leafnode import LeafNodeSampler
from bartpy.samplers.modelsampler import ModelSampler
from bartpy.samplers.schedule import ClassifierSampleSchedule
from bartpy.samplers.sigma import ConstantSigmaSampler
from bartpy.samplers.treemutation import TreeMutationSampler
from bartpy.samplers.unconstrainedtree.treemutation import get_tree_sampler

from scipy.stats import norm


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

    def _construct_model(self, X: np.ndarray, y: np.ndarray) -> ClassifierModel:
        if len(X) == 0 or X.shape[1] == 0:
            raise ValueError("Empty covariate matrix passed")
        self.data = self._convert_covariates_to_data(X, y)
        self.sigma = Sigma(self.sigma_a, self.sigma_b, 1)   # scaling_factor can be anything. it will be ignored
        self.model = ClassifierModel(self.data,
                                     self.sigma,
                                     n_trees=self.n_trees,
                                     alpha=self.alpha,  
                                     beta=self.beta,
                                     initializer=self.initializer)
        return self.model

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
                 store_in_sample_predictions: bool = False,
                 store_acceptance_trace: bool = False,
                 tree_sampler: TreeMutationSampler = get_tree_sampler(0.5, 0.5),
                 initializer: Optional[Initializer] = None,
                 n_jobs=-1):

        SklearnModel.__init__(self,
                              n_trees, n_chains, sigma_a, sigma_b, n_samples,
                              n_burn, thin, alpha, beta, store_in_sample_predictions,
                              store_acceptance_trace, tree_sampler, initializer, n_jobs)

        self.schedule = ClassifierSampleSchedule(self.tree_sampler, LeafNodeSampler(), ConstantSigmaSampler())
        self.sampler = ModelSampler(self.schedule)
        self.DEFAULT_CLASSIFICATION_RULE = 0.5

    def predict(self, X: np.ndarray = None) -> np.ndarray:
        if X is None and self.store_in_sample_predictions:
            binary_pred = [1 if x > self.DEFAULT_CLASSIFICATION_RULE else 0 for x in
                           np.mean(self._prediction_samples, axis=0)]
            return np.array(binary_pred)
        elif X is None and not self.store_in_sample_predictions:
            raise ValueError(
                "In sample predictions only possible if model.store_in_sample_predictions is `True`.  Either set the parameter to True or pass a non-None X parameter")
        else:
            prob_pred = norm.cdf(self._out_of_sample_predict(X))
            return prob_pred

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> 'SklearnModel':
        """
        Learn the model based on training data

        Parameters
        ----------
        X: pd.DataFrame
            training covariates
        y: np.ndarray
            training targets

        Returns
        -------
        SklearnModel
            self with trained parameter values
        """
        assert y.max() - y.min() > 0, "in classification mode ymax should be greater than ymin. Perhaps increase" \
                                      " sample size?"
        assert len([yi for yi in y if yi != 0 and yi != 1]) == 0, "class labels in Y should only contain 0/1 for this" \
                                                                  " binary classifier"
        return super().fit(X, y)

    def _out_of_sample_predict(self, X):
        return np.mean([x.predict(X) for x in self._model_samples], axis=0)

    @staticmethod
    def _convert_covariates_to_data(X: np.ndarray, y: np.ndarray) -> Data:
        from copy import deepcopy
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        return Data(deepcopy(X), deepcopy(y), normalize=False)