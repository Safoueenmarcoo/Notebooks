from typing import Any, Optional, Union, List
import jax.numpy as jnp  # type: ignore

from sklearn.preprocessing import normalize  # type: ignore


class HMM:
    """
    A Hidden Markov Model (HMM) class for modeling systems with hidden states and observable outputs.

    Attributes:
        observations (jnp.ndarray): A 1-D array of observations.
        labels (list or jnp.ndarray): Labels of the hidden states.
        transition_matrix (jnp.ndarray): A square matrix defining state transition probabilities.
        emission_matrix (jnp.ndarray): A matrix defining observation probabilities given states.
        initial_state (jnp.ndarray): Initial starting probabilities of the states.
        states (jnp.ndarray): Array of state indices.
        T (int): Number of observations.
        N (int): Number of states.
        alpha (jnp.ndarray): Forward probabilities matrix.
        beta (jnp.ndarray): Backward probabilities matrix.
        gamma (jnp.ndarray): Posterior probabilities matrix.
        theta (jnp.ndarray): Intermediate matrix for Baum-Welch algorithm.
    """

    def __init__(
        self,
        observations: jnp.ndarray,
        labels: Union[List[str], jnp.ndarray],
        transition_matrix: jnp.ndarray,
        emission_matrix: jnp.ndarray,
        initial_state: Optional[jnp.ndarray] = None,
    ):
        """
        Initializes the HMM with observations, state characteristics, transition matrix, emission matrix,
        and optional initial state.

        Args:
            observations (jnp.ndarray): A 1-D array of observations.
            labels (list or jnp.ndarray): Labels of the hidden states.
            transition_matrix (jnp.ndarray): A square matrix of state transition probabilities.
            emission_matrix (jnp.ndarray): A matrix of observation probabilities given states.
            initial_state (jnp.ndarray, optional): Initial probabilities of the states. If None, the
                stationary distribution of the transition matrix is used.

        Raises:
            Exception: If the input matrices or vectors do not meet the required conditions.
        """
        self.observations = observations
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.initial_state = (
            initial_state if initial_state is not None else self.stationary_states()
        )
        self.labels = labels
        self.states = jnp.array(list(range(len(labels))))
        self.T = len(observations)
        self.N = len(self.states)
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.theta = None
        assert self.verify_matrices()

    def stationary_states(self) -> jnp.ndarray:
        """
        Computes the stationary states of the transition matrix.

        Returns:
            jnp.ndarray: The stationary state probabilities.
        """
        eigenvalues, eigenvectors = jnp.linalg.eig(self.transition_matrix.T)
        stationary = eigenvectors[:, jnp.isclose(eigenvalues, 1)].flatten()
        stationary = abs(stationary) / jnp.sum(stationary)
        return jnp.real(stationary)

    def verify_matrices(self) -> bool:
        """
        Validates the shapes and properties of the matrices and vectors used in the HMM.

        Returns:
            bool: True if all matrices and vectors are valid.

        Raises:
            Exception: If any of the validation checks fail.
        """
        shape_trans_mat = self.transition_matrix.shape
        if self.observations.ndim != 1 and (
            self.observations.ndim != 2 or 1 not in self.observations.shape
        ):
            raise Exception("The observations must be 1-D vector !")
        elif self.states.ndim != 1 and (
            self.states.ndim != 2 or 1 not in self.states.shape
        ):
            raise Exception("The states must be 1-D vector !")
        elif shape_trans_mat[0] != shape_trans_mat[1]:
            raise Exception(
                f"Invalid transition matrix, transition matrix must be a squared matrix as the number of states ({shape_trans_mat[0]})!"
            )
        elif shape_trans_mat[0] != self.emission_matrix.shape[0]:
            raise Exception(
                f"Invalid emission matrix, the emission matrix must have the same number of rows as the transition matrix as they share the same number of states ({shape_trans_mat[0]})!"
            )
        elif jnp.any(jnp.sum(self.transition_matrix, axis=1) != 1):
            raise Exception(
                "Invalid transition matrix, sum of probabilities of each state must be 1!"
            )
        elif jnp.any(jnp.sum(self.emission_matrix, axis=1) != 1):
            raise Exception(
                "Invalid emission matrix, sum of probabilities of each state must be 1!"
            )
        elif self.initial_state.shape != (
            1,
            shape_trans_mat[0],
        ) and self.initial_state.shape != (shape_trans_mat[0],):
            raise Exception(
                f"Invalid initial state vector, it must be (1,{shape_trans_mat[0]}) or {(shape_trans_mat[0],)}"
            )
        elif jnp.any(self.observations < 0) or jnp.any(
            self.observations > self.emission_matrix.shape[1] - 1
        ):
            raise Exception(
                f"Invalid observations vector, this system has only {self.emission_matrix.shape[1]} possible observations ({list(range(self.emission_matrix.shape[1]))})"
            )
        return True

    def verify_obs(self, obs: int) -> bool:
        """
        Verifies if an observation is valid.

        Args:
            obs (int): The observation to verify.

        Raises:
            Exception: If the observation is not an integer or is out of range.
        """
        if not isinstance(obs, int):
            raise Exception("The observation must be an integer")
        elif obs < 0 or obs > self.emission_matrix.shape[1] - 1:
            raise Exception(
                f"This observation doesn't exist, this system has only {self.emission_matrix.shape[1]} possible observations ({list(range(self.emission_matrix.shape[1]))})"
            )
        return True

    def verify_position(self, position: Optional[int]) -> bool:
        """
        Verifies if a position is valid.

        Args:
            position (int): The position to verify.

        Raises:
            Exception: If the position is not an integer or is out of range.
        """
        if position is not None:
            if not isinstance(position, int):
                raise Exception("The position must be an integer")
            elif position < 1 or position > self.gamma.shape[1]:
                raise Exception(
                    f"This position doesn't exist, this system has only {self.gamma.shape[1]} possible positions ({list(range(1, self.gamma.shape[1] + 1))})"
                )
            return True
        return False

    def initial_alpha(self) -> None:
        """
        Initializes the forward probabilities (alpha) at time t=0.

        Updates:
            self.alpha (jnp.ndarray): Forward probabilities matrix initialized for t=0.
        """
        self.alpha = jnp.zeros((self.N, self.T))
        for i in range(self.N):
            self.alpha = self.alpha.at[i, 0].set(
                self.initial_state[i] * self.emission_matrix[i, self.observations[0]]
            )

    def last_alpha(self, t: int) -> jnp.ndarray:
        """
        Computes the forward probabilities (alpha) at time t.

        Args:
            t (int): The time step for which to compute the forward probabilities.

        Returns:
            jnp.ndarray: Updated forward probabilities matrix.

        Updates:
            self.alpha (jnp.ndarray): Forward probabilities matrix updated for time t.
        """
        for j in range(self.N):
            sum = 0
            for i in range(self.N):
                sum += self.alpha[i, t - 1] * self.transition_matrix[i, j]
            self.alpha = self.alpha.at[j, t].set(
                sum * self.emission_matrix[j, self.observations[t]]
            )
        return self.alpha

    def forward_pass(self) -> jnp.ndarray:
        """
        Performs the forward pass to compute the forward probabilities for all time steps.

        Returns:
            jnp.ndarray: The forward probabilities matrix.
        """
        self.initial_alpha()
        for t in range(1, self.T):
            self.last_alpha(t)
        return self.alpha

    def initial_beta(self) -> jnp.ndarray:
        """
        Initializes the backward probabilities (beta) at time T-1.

        Updates:
            self.beta (jnp.ndarray): Backward probabilities matrix initialized for t=T-1.
        """
        self.beta = jnp.zeros((self.N, self.T))
        self.beta = self.beta.at[:, self.T - 1].set(1)
        return self.beta

    def last_beta(self, t: int) -> jnp.ndarray:
        """
        Computes the backward probabilities (beta) at time t.

        Args:
            t (int): The time step for which to compute the backward probabilities.

        Returns:
            jnp.ndarray: Updated backward probabilities matrix.

        Updates:
            self.beta (jnp.ndarray): Backward probabilities matrix updated for time t.
        """
        for j in range(self.N):
            sum = 0
            for i in range(self.N):
                sum += (
                    self.beta[i, t + 1]
                    * self.transition_matrix[j, i]
                    * self.emission_matrix[i, self.observations[t + 1]]
                )
            self.beta = self.beta.at[j, t].set(sum)
        return self.beta

    def backward_pass(self) -> jnp.ndarray:
        """
        Performs the backward pass to compute the backward probabilities for all time steps.

        Returns:
            jnp.ndarray: The backward probabilities matrix.
        """
        self.initial_beta()
        for t in range(self.T - 2, -1, -1):
            self.last_beta(t)
        return self.beta

    def posterior_probabilities(self, position: Optional[int] = None) -> jnp.ndarray:
        """
        Computes the posterior probabilities (gamma) for all states and time steps.

        Args:
            position (int, optional): The position to print probabilities for.

        Returns:
            jnp.ndarray: The posterior probabilities matrix.

        Prints:
            The probabilities of being in the most and least likely states at the specified position.
        """

        self.forward_pass()
        self.backward_pass()
        self.gamma = normalize(self.beta * self.alpha, "l1", axis=0)
        if self.verify_position(position):
            print(
                f"For the position {position}, you have {jnp.max(self.gamma[:, position - 1]):.3f} chance to be in state '{self.labels[jnp.argmax(self.gamma[:, position - 1])]}' and {jnp.min(self.gamma[:, position - 1]):.3f} chance to be in state '{self.labels[jnp.argmin(self.gamma[:, position - 1])]}'"
            )
        return self.gamma

    def baum_welch(self, threshold: float = 0.001, verbose: bool = True) -> None:
        """
        Performs the Baum-Welch algorithm to estimate the HMM parameters (transition and emission matrices).

        Args:
            threshold (float): Convergence threshold for parameter updates.
            verbose (bool): Whether to print intermediate results.

        Returns:
            None
        """

        new_transition = self.transition_matrix
        new_emission = self.emission_matrix
        x = 0
        while True:
            self.theta = jnp.zeros((self.T, self.N, self.N))
            self.posterior_probabilities()
            for t in range(self.T):
                for i in range(self.N):
                    for j in range(self.N):
                        self.theta = self.theta.at[t, i, j].set(
                            self.alpha[i, t]
                            * self.transition_matrix[i, j]
                            * self.emission_matrix[j, self.observations[t + 1]]
                            * self.beta[j, t + 1]
                        )
                self.theta = self.theta.at[t, :, :].set(
                    self.theta[t, :, :] / jnp.sum(self.theta[t, :, :])
                )
            for i in range(self.N):
                for j in range(self.N):
                    new_transition = new_transition.at[i, j].set(
                        jnp.sum(self.theta[:, i, j]) / jnp.sum(self.gamma[i, :-1])
                    )
            new_transition = jnp.array(normalize(new_transition, "l1", axis=1))
            for i in range(self.N):
                sum = jnp.sum(self.gamma[i, :])
                for t in range(self.T):
                    new_emission = new_emission.at[i, t].set(
                        jnp.sum(self.gamma[i, self.observations == t]) / sum
                    )
            new_emission = jnp.array(normalize(new_emission, "l1", axis=1))
            x += 1
            if jnp.max(jnp.abs(new_transition - self.transition_matrix)) < threshold:
                self.transition_matrix = new_transition
                self.emission_matrix = new_emission
                self.initial_state = self.stationary_states()
                if verbose:
                    print(f"Converged after {x} iterations.")
                break
            self.transition_matrix = new_transition
            self.emission_matrix = new_emission

    def add_observation(self, obs: int) -> jnp.ndarray:
        """
        Adds a new observation to the sequence and updates the forward probabilities.

        Args:
            obs (int): The new observation to add.

        Returns:
            jnp.ndarray: The updated forward probabilities matrix.

        Raises:
            AssertionError: If the observation is invalid.
        """
        assert self.verify_obs(obs)
        self.observations = jnp.hstack([self.observations, obs])
        self.T += 1
        self.alpha = jnp.hstack([self.alpha, jnp.zeros((self.N, 1))])
        self.last_alpha(self.T - 1)
        self.baum_welch()


"################################################################################# CHMM(not completed) ###########################################################################"


class ContinuousHMM(HMM):
    """
    A class representing a continuous Hidden Markov Model (HMM) with Gaussian emissions.
    Inherits from the base HMM class.
    """

    def __init__(
        self,
        observations: jnp.ndarray,
        labels: Union[List[Any], jnp.ndarray],
        transition_matrix: jnp.ndarray,
        means: jnp.ndarray,
        cov: jnp.ndarray,
        initial_state: Optional[jnp.ndarray] = None,
    ):
        """
        Initializes the ContinuousHMM object with the given parameters.

        Args:
            observations (jnp.ndarray): The sequence of observed data points.
            labels (list or jnp.ndarray): The corresponding labels for the observations (if any).
            transition_matrix (jnp.ndarray): The matrix defining state transition probabilities.
            means (jnp.ndarray): The mean vectors for each hidden state's Gaussian distribution.
            cov (jnp.ndarray): The covariance matrices for each hidden state's Gaussian distribution.
            initial_state (jnp.ndarray, optional): The initial state probability distribution. Defaults to None.

        Attributes:
            T (int): The total number of time steps in the observation sequence.
            means (jnp.ndarray): Mean vectors for Gaussian emissions.
            cov (jnp.ndarray): Covariance matrices for Gaussian emissions.
            emission_matrix (jnp.ndarray): Matrix containing emission probabilities for each state and observation.
        """
        super().__init__(observations, labels, transition_matrix, None, initial_state)
        self.T = self.observations.shape[0]
        self.means = means
        self.cov = cov
        self.emission_matrix = self.initiate_emissions()

    def verify_matrices(self) -> bool:
        """
        Validates the shapes and properties of the matrices and vectors used in the HMM.

        Returns:
            bool: True if all matrices and vectors are valid.

        Raises:
            Exception: If any validation check fails.
        """
        shape_trans_mat = self.transition_matrix.shape
        if self.states.ndim != 1 and (
            self.states.ndim != 2 or 1 not in self.states.shape
        ):
            raise Exception("The states must be a 1-D vector!")
        elif shape_trans_mat[0] != shape_trans_mat[1]:
            raise Exception(
                f"Invalid transition matrix, transition matrix must be a squared matrix as the number of states ({shape_trans_mat[0]})!"
            )
        elif jnp.sum(self.transition_matrix) != len(self.transition_matrix):
            raise Exception(
                "Invalid transition matrix, sum of probabilities of each state must be 1!"
            )
        elif self.initial_state.shape != (
            1,
            shape_trans_mat[0],
        ) and self.initial_state.shape != (shape_trans_mat[0],):
            raise Exception(
                f"Invalid initial state vector, it must be (1,{shape_trans_mat[0]}) or {(shape_trans_mat[0],)}"
            )
        return True

    def gaussian_pdf(
        self, x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray
    ) -> float:
        """
        Computes the probability density function (PDF) of a multivariate Gaussian distribution.

        Args:
            x (jnp.ndarray): The input data point.
            mu (jnp.ndarray): The mean vector of the Gaussian distribution.
            sigma (jnp.ndarray): The covariance matrix of the Gaussian distribution.

        Returns:
            float: The computed PDF value.
        """
        return jnp.exp(
            -0.5 * jnp.dot((x - mu).T, jnp.linalg.solve(sigma, (x - mu)))
        ) / jnp.sqrt((2 * jnp.pi) ** len(mu) * jnp.linalg.det(sigma))

    def initial_alpha(self) -> None:
        """
        Initializes the forward probabilities (alpha) at time t=0.

        Updates:
            self.alpha (jnp.ndarray): Forward probabilities matrix initialized for t=0.
        """
        self.alpha = jnp.zeros((self.N, self.T))
        for i in range(self.N):
            self.alpha = self.alpha.at[i, 0].set(
                self.initial_state[i] * self.emission_matrix[i, 0]
            )

    def last_alpha(self, t: int) -> jnp.ndarray:
        """
        Computes the forward probabilities (alpha) at time t using recursion.

        Args:
            t (int): The time step for which to compute the forward probabilities.

        Returns:
            jnp.ndarray: Updated forward probabilities matrix.

        Updates:
            self.alpha (jnp.ndarray): Forward probabilities matrix updated for time t.
        """
        for j in range(self.N):
            sum_ = 0
            for i in range(self.N):
                sum_ += self.alpha[i, t - 1] * self.transition_matrix[i, j]
            self.alpha = self.alpha.at[j, t].set(sum_ * self.emission_matrix[j, t])
        return self.alpha

    def last_beta(self, t: int) -> jnp.ndarray:
        """
        Computes the backward probabilities (beta) at time t using recursion.

        Args:
            t (int): The time step for which to compute the backward probabilities.

        Returns:
            jnp.ndarray: Updated backward probabilities matrix.

        Updates:
            self.beta (jnp.ndarray): Backward probabilities matrix updated for time t.
        """
        for j in range(self.N):
            sum_ = 0
            for i in range(self.N):
                sum_ += (
                    self.beta[i, t + 1]
                    * self.transition_matrix[j, i]
                    * self.emission_matrix[i, t + 1]
                )
            self.beta = self.beta.at[j, t].set(sum_)
        return self.beta

    def initiate_emissions(self) -> jnp.ndarray:
        """
        Initializes the emission probabilities matrix based on the Gaussian PDF.

        Returns:
            jnp.ndarray: The emission probabilities matrix.
        """

        emission_matrix = jnp.zeros((self.N, self.observations.shape[1]))
        for state in range(self.N):
            for obs in range(self.observations.shape[1]):
                emission_matrix = emission_matrix.at[state, obs].set(
                    self.gaussian_pdf(
                        self.observations[obs, :], self.means[state], self.cov[state]
                    )
                )
        emission_matrix = jnp.nan_to_num(emission_matrix, nan=0)
        emission_matrix = jnp.array(normalize(emission_matrix, "l1", axis=1))
        return emission_matrix

    def baum_welch(self, threshold: float = 0.001, verbose: bool = True) -> None:
        """
        Performs the Baum-Welch algorithm to estimate the HMM parameters (transition and emission matrices).

        Args:
            threshold (float): Convergence threshold for parameter updates.
            verbose (bool): Whether to print intermediate results.

        Returns:
            None
        """

        new_transition = self.transition_matrix
        x = 0
        while True:
            self.theta = jnp.zeros((self.T, self.N, self.N))
            self.posterior_probabilities()
            for t in range(self.T):
                for i in range(self.N):
                    for j in range(self.N):
                        self.theta = self.theta.at[t, i, j].set(
                            self.alpha[i, t]
                            * self.transition_matrix[i, j]
                            * self.emission_matrix[j, t + 1]
                            * self.beta[j, t + 1]
                        )
                self.theta = self.theta.at[t, :, :].set(
                    self.theta[t, :, :] / jnp.sum(self.theta[t, :, :])
                )
            for i in range(self.N):
                for j in range(self.N):
                    new_transition = new_transition.at[i, j].set(
                        jnp.sum(self.theta[:, i, j]) / jnp.sum(self.gamma[i, :-1])
                    )
            new_transition = jnp.nan_to_num(new_transition, nan=0)
            new_transition = jnp.array(normalize(new_transition, "l1", axis=1))
            for state in range(self.N):
                self.means = self.means.at[state].set(
                    jnp.sum(
                        jnp.expand_dims(self.gamma[state, :], axis=-1)
                        * self.observations,
                        axis=0,
                    )
                    / self.gamma[state, :].sum()
                )
                centered = self.observations - self.means[state]
                self.cov = self.cov.at[state].set(
                    (
                        (jnp.expand_dims(self.gamma[state, :], axis=-1) * centered).T
                        @ centered
                    )
                    / self.gamma[state, :].sum()
                )
            x += 1
            if jnp.max(jnp.abs(new_transition - self.transition_matrix)) < threshold:
                self.transition_matrix = new_transition
                self.emission_matrix = self.initiate_emissions()
                if verbose:
                    print(f"Converged after {x} iterations.")
                break
            self.transition_matrix = new_transition
            self.emission_matrix = self.initiate_emissions()
