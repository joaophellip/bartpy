from typing import Callable, Generator, Text, Tuple

import numpy as np

from bartpy.model import Model
from bartpy.samplers.leafnode import LeafNodeSampler
from bartpy.samplers.sigma import SigmaSampler
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
                 sigma_sampler: SigmaSampler):
        self.leaf_sampler = leaf_sampler
        self.sigma_sampler = sigma_sampler
        self.tree_sampler = tree_sampler
        self.g = -1
        self.z = -1

    def steps(self, model: Model) -> Generator[Tuple[Text, Callable[[], float]], None, None]:
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

        # See: https://github.com/kapelner/bartMachine/blob/master/src/bartMachine/bartMachineClassification.java L61
        self.g = np.zeros(model.data.X.n_obsv)
        for tree in model.trees:
            self.g = [sum(x) for x in zip(self.g, tree.predict())]
        self.z = self.sample_z(self.g, model.data.y.unnormalized_y)

        for tree in model.trees:
            for node in tree.nodes:
                node.data.update_y(self.z)

        for tree in model.refreshed_trees():
            yield "Tree", lambda: self.tree_sampler.step(model, tree)

            for leaf_node in tree.leaf_nodes:
                yield "Node", lambda: self.leaf_sampler.step(model, leaf_node)
        yield "Node", lambda: self.sigma_sampler.step(model, model.sigma)

    # todo: test SampleZ and ConditionalZ against Java code (https://github.com/kapelner/bartMachine/blob/master/src/bartMachine/bartMachineClassification.java#L61)
    def sample_z(self, g, y) -> np.ndarray:
        return np.array([self.conditional_z(x[0], x[1]) for x in zip(g, y)])

    def conditional_z(self, gi, yi) -> float:
        u = random.uniform(0, 1)
        if yi == 0:
            return gi - norm.ppf((1-u)*norm.cdf(gi) + u)
        elif yi == 1:
            return gi + norm.ppf((1-u)*norm.cdf(-gi) + u)

