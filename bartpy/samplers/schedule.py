from typing import Callable, Generator, Text, Tuple, Union

import numpy as np

from bartpy.model import Model
from bartpy.samplers.leafnode import LeafNodeSampler
from bartpy.samplers.sigma import SigmaSampler, ConstantSigmaSampler
from bartpy.samplers.treemutation import TreeMutationSampler
import random
from scipy.stats import norm


class SampleSchedule:
    """
    The SampleSchedule class is responsible for handling the ordering of sampling within a Gibbs step
    It is useful to encapsulate this logic if we wish to expand the model

    Parameters
    ----------
    tree_sampler: TreeMutationSampler
        How to sample tree mutation space
    leaf_sampler: LeafNodeSampler
        How to sample leaf node predictions
    sigma_sampler: SigmaSampler
        How to sample sigma values
    """

    def __init__(self,
                 tree_sampler: TreeMutationSampler,
                 leaf_sampler: LeafNodeSampler,
                 sigma_sampler: Union[SigmaSampler, ConstantSigmaSampler]):
        self.leaf_sampler = leaf_sampler
        self.sigma_sampler = sigma_sampler
        self.tree_sampler = tree_sampler

    def steps(self, model: Model, _: int) -> Generator[Tuple[Text, Callable[[], float]], None, None]:
        """
        Create a generator of the steps that need to be called to complete a full Gibbs sample

        Parameters
        ----------
        model: Model
            The model being sampled

        Returns
        -------
        Generator[Callable[[Model], Sampler], None, None]
            A generator a function to be called
        """

        for tree in model.refreshed_trees():
            yield "Tree", lambda: self.tree_sampler.step(model, tree)

            for leaf_node in tree.leaf_nodes:
                yield "Node", lambda: self.leaf_sampler.step(model, leaf_node)
        yield "Node", lambda: self.sigma_sampler.step(model, model.sigma)


class ClassifierSampleSchedule:

    def __init__(self,
                 tree_sampler: TreeMutationSampler,
                 leaf_sampler: LeafNodeSampler,
                 sigma_sampler: Union[SigmaSampler, ConstantSigmaSampler]):
        self.leaf_sampler = leaf_sampler
        self.sigma_sampler = sigma_sampler
        self.tree_sampler = tree_sampler

    def steps(self, model: Model, _: int) -> Generator[Tuple[Text, Callable[[], float]], None, None]:
        """
        Create a generator of the steps that need to be called to complete a full Gibbs sample

        Parameters
        ----------
        model: Model
            The model being sampled

        Returns
        -------
        Generator[Callable[[Model], Sampler], None, None]
            A generator a function to be called
        """

        g = np.zeros(model.data.X.n_obsv)
        for tree in model.trees:
            g = [sum(x) for x in zip(g, tree.predict())]
        z = self.sample_z(g, model.data.y._original_y)

        for tree in model.trees:
            for node in tree.nodes:
                node.data.update_y(z)
        model.data.update_y(z)

        for tree in model.refreshed_trees():
            yield "Tree", lambda: self.tree_sampler.step(model, tree)

            for leaf_node in tree.leaf_nodes:
                yield "Node", lambda: self.leaf_sampler.step(model, leaf_node)
        yield "Node", lambda: self.sigma_sampler.step(model, model.sigma)

    def sample_z(self, g, y) -> np.ndarray:
        return np.array([self.conditional_z(x[0], x[1]) for x in zip(g, y)])

    @staticmethod
    def conditional_z(gi, yi) -> float:
        u = random.uniform(0, 1)
        if yi == 0:
            zi = gi - norm.ppf((1 - u) * norm.cdf(gi) + u)
            assert zi <= 0, "this should always be greater or equal zero"
        else:
            zi = gi + norm.ppf((1 - u) * norm.cdf(-gi) + u)
            assert zi >= 0, "this should always be less or equal zero"
        return zi
