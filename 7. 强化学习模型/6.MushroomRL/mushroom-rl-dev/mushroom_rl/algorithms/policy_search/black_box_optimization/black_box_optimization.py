import numpy as np

from mushroom_rl.core import Agent


class BlackBoxOptimization(Agent):
    """
    Base class for black box optimization algorithms.
    These algorithms work on a distribution of policy parameters and often they
    do not rely on stochastic and differentiable policies.

    """
    def __init__(self, mdp_info, distribution, policy):
        """
        Constructor.

        Args:
            distribution (Distribution): the distribution of policy parameters;
            policy (ParametricPolicy): the policy to use.

        """
        self.distribution = distribution

        self._add_save_attr(distribution='mushroom')

        super().__init__(mdp_info, policy, is_episodic=True)

    def episode_start(self, episode_info):
        theta = self.distribution.sample()
        self.policy.set_weights(theta)

        policy_state, _ = super().episode_start(episode_info)

        return policy_state, theta

    def fit(self, dataset):
        Jep = np.array(dataset.discounted_return)
        theta = np.array(dataset.theta_list)

        self._update(Jep, theta)

    def _update(self, Jep, theta):
        """
        Function that implements the update routine of distribution parameters.
        Every black box algorithms should implement this function with the
        proper update.

        Args:
            Jep (np.ndarray): a vector containing the J of the considered
                trajectories;
            theta (np.ndarray): a matrix of policy parameters of the considered
                trajectories.

        """
        raise NotImplementedError('BlackBoxOptimization is an abstract class')
