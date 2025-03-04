import jax.numpy as jnp  # type: ignore
import gymnasium as gym  # type: ignore
from sklearn.preprocessing import normalize  # type: ignore
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, Input, Dropout  # type: ignore
from keras.optimizers import Adam  # type: ignore
from collections import deque
from keras.callbacks import TensorBoard  # type: ignore
import tensorflow as tf  # type: ignore
from time import time
from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore
from gymnasium.spaces import Discrete  # type: ignore
import random
from stable_baselines3.common.callbacks import BaseCallback  # type: ignore

"########################################################################################### Control ###########################################################################################"


# Function to check if a matrix A is stable
# A matrix is considered stable if all its eigenvalues have negative real parts
def is_stable(A):
    # Compute the eigenvalues of matrix A
    eigenvalues = jnp.linalg.eigvals(A)

    # Check if all eigenvalues have negative real parts
    stable = bool(jnp.all(jnp.real(eigenvalues) < 0))

    # If the matrix is not stable, print the real parts of the eigenvalues
    if not stable:
        print(jnp.real(eigenvalues))

    # Return whether the matrix is stable or not
    return stable


# Function to compute the controllability matrix of a system defined by matrices A and B
# The controllability matrix is used to determine if the system is controllable


def controllability_matrix(A, B):
    # Get the dimension of the state matrix A
    n = A.shape[0]

    # Initialize the controllability matrix with B
    C = B

    # Iteratively build the controllability matrix
    for i in range(1, n):
        # Append the product of A^i and B to the controllability matrix
        C = jnp.hstack((C, jnp.linalg.matrix_power(A, i) @ B))

    # Return the controllability matrix
    return C


# Function to check if a system defined by matrices A and B is controllable
# A system is controllable if the controllability matrix has full rank
def is_controllable(A, B):
    # Compute the controllability matrix
    C = controllability_matrix(A, B)

    # Compute the rank of the controllability matrix
    rank = jnp.linalg.matrix_rank(C)

    # Return whether the system is controllable (i.e., rank equals the dimension of A)
    return bool(rank == A.shape[0])


# Function to compute the observability matrix of a system defined by matrices A and C
# The observability matrix is used to determine if the system is observable
def observability_matrix(A, C):
    # Get the dimension of the state matrix A
    n = A.shape[0]

    # Initialize the observability matrix with C
    Obs = C

    # Iteratively build the observability matrix
    for i in range(1, n):
        # Append the product of C and A^i to the observability matrix
        Obs = jnp.vstack((Obs, C @ jnp.linalg.matrix_power(A, i)))

    # Return the observability matrix
    return Obs


# Function to check if a system defined by matrices A and C is observable
# A system is observable if the observability matrix has full rank
def is_observable(A, C):
    # Compute the observability matrix
    Obs = observability_matrix(A, C)

    # Compute the rank of the observability matrix
    rank = jnp.linalg.matrix_rank(Obs)

    # Return whether the system is observable (i.e., rank equals the dimension of A)
    return bool(rank == A.shape[0])


"########################################################################################### Kalman Filter ###########################################################################################"


def KalmanFilter_1D(est, mea_err, measurements):
    ests = []
    err = 1
    for i in range(len(measurements)):
        KG = err / (err + mea_err)
        est = est + KG * (measurements[i] - est)
        err = (1 - KG) * err
        ests.append(est)
    return ests


def step_estimation(A, B, x_k, u_k, w_k):
    """
    Predicts the next state of the system based on the current state, input, and noise.

    Parameters:
        A (ndarray): State transition matrix of shape (n, n).
        B (ndarray): Control input matrix of shape (n, m).
        x_k (ndarray): Current state vector of shape (n, 1).
        u_k (ndarray): Control input vector of shape (m, 1).
        w_k (ndarray): Process noise vector of shape (n, 1).

    Returns:
        ndarray: Predicted next state vector of shape (n, 1).
    """
    try:
        x_k = x_k.reshape((-1, 1))
        return A @ x_k + B @ u_k + w_k
    except Exception as e:
        print("Error in the step estimation function, the error: ", e)


def process_covariance(A, P, Q):
    """
    Updates the covariance matrix of the state based on the state transition matrix and process noise.

    Parameters:
        A (ndarray): State transition matrix of shape (n, n).
        P (ndarray): Current error covariance matrix of shape (n, n).
        Q (ndarray): Process noise covariance matrix of shape (n, n).

    Returns:
        ndarray: Updated error covariance matrix of shape (n, n).
    """
    try:
        return A @ P @ A.T + Q
    except Exception as e:
        print("Error in the proccess covariance function, the error: ", e)


def kalman_function(P, H, R):
    """
    Computes the Kalman gain for updating the state estimate.

    Parameters:
        P (ndarray): Error covariance matrix of shape (n, n).
        H (ndarray): Observation matrix of shape (m, n).
        R (ndarray): Measurement noise covariance matrix of shape (m, m).

    Returns:
        ndarray: Kalman gain matrix of shape (n, m).
    """
    try:
        x = P @ H.T
        K = x @ jnp.linalg.inv(H @ x + R)
        return jnp.nan_to_num(K, nan=0)
    except Exception as e:
        print("Error in the kalman gain function, the error: ", e)


def current_state_and_process(K, H, C, x_km, x_kp, p_kp, Z):
    """
    Updates the state estimate and covariance matrix based on the Kalman gain and measurement.

    Parameters:
        K (ndarray): Kalman gain matrix of shape (n, m).
        H (ndarray): Observation matrix of shape (m, n).
        C (ndarray): System output matrix of shape (m, n).
        x_km (ndarray): Measured state vector of shape (n, 1).
        x_kp (ndarray): Predicted state vector of shape (n, 1).
        p_kp (ndarray): Predicted error covariance matrix of shape (n, n).
        Z (ndarray): Measurement noise matrix of shape (m, 1).

    Returns:
        tuple: Updated state estimate vector (n, 1) and updated error covariance matrix (n, n).
    """
    try:
        Y = C @ x_km.reshape((-1, 1)) + Z
        x_k = x_kp + K @ (Y - H @ x_kp)
        p_k = (jnp.eye(K.shape[0]) - K @ H) @ p_kp
        return x_k, p_k
    except Exception as e:
        print(
            "Error in the new state and proccess calculation function, the error: ", e
        )


class KalmanFilter:
    """
    Implements a Kalman Filter for state estimation in dynamic systems.

    Attributes:
        x_k (ndarray): Current state estimate vector (n, 1).
        A (ndarray): State transition matrix of shape (n, n).
        B (ndarray): Control input matrix of shape (n, m).
        H (ndarray): Observation matrix of shape (m, n).
        R (ndarray): Measurement noise covariance matrix of shape (m, m).
        C (ndarray): System output matrix of shape (m, n).
        Q (ndarray): Process noise covariance matrix of shape (n, n).
        Z (ndarray): Measurement noise matrix of shape (m, 1).
        w_k (ndarray): Process noise vector of shape (n, 1).
        P (ndarray): Error covariance matrix of shape (n, n).
    """

    def __init__(self, x_0, A, B, H, R, C, Q, Z, w_k, P_0):
        """
        Initialize the Kalman Filter with the given parameters.

        Parameters:
            x_0 (ndarray): Initial state estimate vector (n, 1).
            A (ndarray): State transition matrix of shape (n, n).
            B (ndarray): Control input matrix of shape (n, m).
            H (ndarray): Observation matrix of shape (m, n).
            R (ndarray): Measurement noise covariance matrix of shape (m, m).
            C (ndarray): System output matrix of shape (m, n).
            Q (ndarray): Process noise covariance matrix of shape (n, n).
            Z (ndarray): Measurement noise matrix of shape (m, 1).
            w_k (ndarray): Process noise vector of shape (n, 1).
            P_0 (ndarray): Initial error covariance matrix of shape (n, n).
        """
        self.x_0 = x_0.reshape((-1, 1))
        self.x_k = self.x_0
        self.A = A
        self.B = B
        self.H = H
        self.R = R
        self.C = C
        self.Q = Q
        self.Z = Z
        self.w_k = w_k
        self.P = P_0
        self.P_0 = P_0

    def reset(self):
        """Resets the state and covariance matrix to their initial values."""
        self.x_k = self.x_0
        self.P = self.P_0

    def predict(self, u_k):
        """
        Predicts the next state based on the control input and process model.

        Parameters:
            u_k (ndarray): Control input vector of shape (m, 1).

        Returns:
            ndarray: Updated state estimate vector of shape (n,).
        """
        try:
            self.x_k = step_estimation(
                self.A, self.B, self.x_k, u_k, self.w_k
            ).squeeze()
            self.P = process_covariance(self.A, self.P, self.Q)
            return self.x_k
        except Exception as e:
            print("Error in the predict method, the error: ", e)

    def update(self, x_km):
        """
        Updates the state estimate based on the new measurement.

        Parameters:
            x_km (ndarray): Measured state vector of shape (n, 1).

        Returns:
            ndarray: Updated state estimate vector of shape (n,).
        """
        try:
            self.x_k = self.x_k.reshape((-1, 1))
            K = kalman_function(self.P, self.H, self.R)
            self.x_k, self.P = current_state_and_process(
                K, self.H, self.C, x_km, self.x_k, self.P, self.Z
            )
            return self.x_k.squeeze()
        except Exception as e:
            print("Error in the update method, the error: ", e)


"########################################################################################### Kalman Filter + RL ###########################################################################################"


class KalmanRLWrapper(gym.Env):
    """
    A wrapper for combining a reinforcement learning environment with a Kalman Filter.

    Parameters:
        env: The underlying RL environment to be wrapped.
        kalman_filter: An instance of the KalmanFilter class for state estimation.

    """

    def __init__(self, env, kalman_filter):
        super(KalmanRLWrapper, self).__init__()
        self.env = env  # The underlying RL environment

        self.kf = kalman_filter  # The Kalman Filter instance

        self.action_space = (
            env.action_space
        )  # Action space inherited from the environment

        self.observation_space = (
            env.observation_space
        )  # Observation space inherited from the environment

    def reset(self, seed=None, **kwargs):
        """
        Reset the environment and initialize the Kalman Filter state.

        Parameters:
            seed: Seed for random number generation in the environment.
            **kwargs: Additional arguments for the environment's reset method.

        Returns:
            Corrected initial observation.
        """
        noisy_obs = self.env.reset(seed=seed, **kwargs)[
            0
        ]  # Reset the environment and get the noisy initial observation

        self.kf.x_k = (
            noisy_obs  # Initialize the Kalman Filter state with the noisy observation
        )

        return self.kf.x_k, {}

    def step(self, action):
        """
        Take a step in the environment and update the Kalman Filter state.

        Parameters:
            action: The action to take in the environment.

        Returns:
            Tuple containing the corrected observation, reward, done flag, truncation flag, and additional info.
        """
        noisy_obs, reward, done, truncuated, info = self.env.step(
            action
        )  # Step through the environment with the given action

        action = (
            -1 if action == 0 else 1
        )  # Convert the action for the Kalman Filter's control input

        u_k = jnp.array([[action]])  # Create a control input Matrix

        self.kf.predict(u_k)  # Predict the next state using the Kalman Filter

        corrected_obs = self.kf.update(
            noisy_obs
        )  # Update the Kalman Filter with the noisy observation

        return corrected_obs, reward, done, truncuated, info


"########################################################################################### HMM ###########################################################################################"


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
        observations,
        labels,
        transition_matrix,
        emission_matrix,
        initial_state=None,
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
        self.alpha = None  # Forward probabilities matrix
        self.beta = None  # Backward probabilities matrix
        self.gamma = None  # Posterior probabilities matrix
        self.theta = None  # Intermediate matrix for Baum-Welch algorithm

        assert self.verify_matrices()  # Validate input matrices

    def stationary_states(self):
        """
        Computes the stationary states of the transition matrix.

        Returns:
            jnp.ndarray: The stationary state probabilities.
        """
        eigenvalues, eigenvectors = jnp.linalg.eig(
            self.transition_matrix.T
        )  # Eigen decomposition
        stationary = eigenvectors[
            :, jnp.isclose(eigenvalues, 1)
        ].flatten()  # Eigenvector for eigenvalue 1
        stationary = abs(stationary) / jnp.sum(stationary)  # Normalize to sum to 1
        return jnp.real(stationary)  # Return real part of the stationary state

    def verify_matrices(self):
        """
        Validates the shapes and properties of the matrices and vectors used in the HMM.

        Returns:
            bool: True if all matrices and vectors are valid.

        Raises:
            Exception: If any of the validation checks fail.
        """
        shape_trans_mat = self.transition_matrix.shape

        # Check if observations is a 1-D vector
        if self.observations.ndim != 1 and (
            self.observations.ndim != 2 or 1 not in self.observations.shape
        ):
            raise Exception("The observations must be 1-D vector !")

        # Check if states is a 1-D vector or a 2-D vector with one dimension being 1
        elif self.states.ndim != 1 and (
            self.states.ndim != 2 or 1 not in self.states.shape
        ):
            raise Exception("The states must be 1-D vector !")

        # Check if the transition matrix is square
        elif shape_trans_mat[0] != shape_trans_mat[1]:
            raise Exception(
                f"Invalid transition matrix, transition matrix must be a squared matrix as the number of states ({shape_trans_mat[0]})!"
            )

        # Check if the emission matrix has the same number of rows as the transition matrix
        elif shape_trans_mat[0] != self.emission_matrix.shape[0]:
            raise Exception(
                f"Invalid emission matrix, the emission matrix must have the same number of rows as the transition matrix as they share the same number of states ({shape_trans_mat[0]})!"
            )

        # Check if the transition matrix rows sum to 1 (valid probability distribution)
        elif jnp.any(jnp.sum(self.transition_matrix, axis=1) != 1):
            raise Exception(
                "Invalid transition matrix, sum of probabilities of each state must be 1!"
            )

        # Check if the emission matrix rows sum to 1 (valid probability distribution)
        elif jnp.any(jnp.sum(self.emission_matrix, axis=1) != 1):
            raise Exception(
                "Invalid emission matrix, sum of probabilities of each state must be 1!"
            )

        # Check if the initial state vector has the correct shape
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

        return True  # All checks passed

    def verify_obs(self, obs):
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

    def verify_position(self, position):
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

    def initial_alpha(self):
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

    def last_alpha(self, t):
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

    def forward_pass(self):
        """
        Performs the forward pass to compute the forward probabilities for all time steps.

        Returns:
            jnp.ndarray: The forward probabilities matrix.
        """
        self.initial_alpha()
        for t in range(1, self.T):
            self.last_alpha(t)
        return self.alpha

    def initial_beta(self):
        """
        Initializes the backward probabilities (beta) at time T-1.

        Updates:
            self.beta (jnp.ndarray): Backward probabilities matrix initialized for t=T-1.
        """
        self.beta = jnp.zeros((self.N, self.T))
        self.beta = self.beta.at[:, self.T - 1].set(1)
        return self.beta

    def last_beta(self, t):
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

    def backward_pass(self):
        """
        Performs the backward pass to compute the backward probabilities for all time steps.

        Returns:
            jnp.ndarray: The backward probabilities matrix.
        """
        self.initial_beta()
        for t in range(self.T - 2, -1, -1):
            self.last_beta(t)
        return self.beta

    def posterior_probabilities(self, position=None):
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
        self.gamma = normalize(
            self.beta * self.alpha, "l1", axis=0
        )  # Normalize to get probabilities

        if self.verify_position(position):
            print(
                f"For the position {position}, you have {jnp.max(self.gamma[:, position - 1]):.3f} chance to be in state '{self.labels[jnp.argmax(self.gamma[:, position - 1])]}' and {jnp.min(self.gamma[:, position - 1]):.3f} chance to be in state '{self.labels[jnp.argmin(self.gamma[:, position - 1])]}'"
            )
        return self.gamma

    def baum_welch(self, threshold=0.001, verbose=True):
        """
        Performs the Baum-Welch algorithm to estimate the HMM parameters (transition and emission matrices).

        Args:
            threshold (float): Convergence threshold for parameter updates.
            verbose (bool): Whether to print intermediate results.

        Returns:
            None
        """
        # Compute posterior probabilities if not already done

        new_transition = self.transition_matrix
        new_emission = self.emission_matrix
        x = 0  # Iteration counter

        while True:
            self.theta = jnp.zeros(
                (self.T, self.N, self.N)
            )  # Initialize intermediate matrix

            self.posterior_probabilities()

            # Compute theta for each time step and state pair
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
                )  # Normalize

            # Update transition matrix
            for i in range(self.N):
                for j in range(self.N):
                    new_transition = new_transition.at[i, j].set(
                        jnp.sum(self.theta[:, i, j]) / jnp.sum(self.gamma[i, :-1])
                    )
            new_transition = jnp.array(
                normalize(new_transition, "l1", axis=1)
            )  # Normalize rows

            # Update emission matrix
            for i in range(self.N):
                sum = jnp.sum(self.gamma[i, :])
                for t in range(self.T):
                    new_emission = new_emission.at[i, t].set(
                        jnp.sum(self.gamma[i, self.observations == t]) / sum
                    )
            new_emission = jnp.array(
                normalize(new_emission, "l1", axis=1)
            )  # Normalize rows

            x += 1  # Increment iteration counter

            # Check for convergence
            if jnp.max(jnp.abs(new_transition - self.transition_matrix)) < threshold:
                self.transition_matrix = new_transition
                self.emission_matrix = new_emission
                self.initial_state = self.stationary_states()
                if verbose:
                    print(f"Converged after {x} iterations.")
                break

            self.transition_matrix = new_transition
            self.emission_matrix = new_emission

    def add_observation(self, obs):
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


"########################################################################################### CHMM(not completed) ###########################################################################"


class ContinuousHMM(HMM):
    """
    A class representing a continuous Hidden Markov Model (HMM) with Gaussian emissions.
    Inherits from the base HMM class.
    """

    def __init__(
        self, observations, labels, transition_matrix, means, cov, initial_state=None
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
        self.T = self.observations.shape[0]  # Number of time steps
        self.means = means  # Mean vectors for Gaussian emissions
        self.cov = cov  # Covariance matrices for Gaussian emissions
        self.emission_matrix = (
            self.initiate_emissions()
        )  # Initialize emission probabilities

    def verify_matrices(self):
        """
        Validates the shapes and properties of the matrices and vectors used in the HMM.

        Returns:
            bool: True if all matrices and vectors are valid.

        Raises:
            Exception: If any validation check fails.
        """
        shape_trans_mat = self.transition_matrix.shape

        # Validate the states vector
        if self.states.ndim != 1 and (
            self.states.ndim != 2 or 1 not in self.states.shape
        ):
            raise Exception("The states must be a 1-D vector!")

        # Validate the transition matrix (must be square)
        elif shape_trans_mat[0] != shape_trans_mat[1]:
            raise Exception(
                f"Invalid transition matrix, transition matrix must be a squared matrix as the number of states ({shape_trans_mat[0]})!"
            )

        # Validate that rows of the transition matrix sum to 1
        elif jnp.sum(self.transition_matrix) != len(self.transition_matrix):
            raise Exception(
                "Invalid transition matrix, sum of probabilities of each state must be 1!"
            )

        # Validate the initial state vector shape
        elif self.initial_state.shape != (
            1,
            shape_trans_mat[0],
        ) and self.initial_state.shape != (shape_trans_mat[0],):
            raise Exception(
                f"Invalid initial state vector, it must be (1,{shape_trans_mat[0]}) or {(shape_trans_mat[0],)}"
            )

        return True  # All checks passed

    def gaussian_pdf(self, x, mu, sigma):
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

    def initial_alpha(self):
        """
        Initializes the forward probabilities (alpha) at time t=0.

        Updates:
            self.alpha (jnp.ndarray): Forward probabilities matrix initialized for t=0.
        """
        self.alpha = jnp.zeros((self.N, self.T))  # Initialize alpha matrix
        for i in range(self.N):  # For each state
            self.alpha = self.alpha.at[i, 0].set(
                self.initial_state[i] * self.emission_matrix[i, 0]
            )  # Set initial alpha values

    def last_alpha(self, t):
        """
        Computes the forward probabilities (alpha) at time t using recursion.

        Args:
            t (int): The time step for which to compute the forward probabilities.

        Returns:
            jnp.ndarray: Updated forward probabilities matrix.

        Updates:
            self.alpha (jnp.ndarray): Forward probabilities matrix updated for time t.
        """
        for j in range(self.N):  # For each state at time t
            sum_ = 0
            for i in range(self.N):  # Sum over all possible previous states
                sum_ += self.alpha[i, t - 1] * self.transition_matrix[i, j]
            self.alpha = self.alpha.at[j, t].set(
                sum_ * self.emission_matrix[j, t]
            )  # Update alpha for state j at time t
        return self.alpha

    def last_beta(self, t):
        """
        Computes the backward probabilities (beta) at time t using recursion.

        Args:
            t (int): The time step for which to compute the backward probabilities.

        Returns:
            jnp.ndarray: Updated backward probabilities matrix.

        Updates:
            self.beta (jnp.ndarray): Backward probabilities matrix updated for time t.
        """
        for j in range(self.N):  # For each state at time t
            sum_ = 0
            for i in range(self.N):  # Sum over all possible next states
                sum_ += (
                    self.beta[i, t + 1]
                    * self.transition_matrix[j, i]
                    * self.emission_matrix[i, t + 1]
                )
            self.beta = self.beta.at[j, t].set(
                sum_
            )  # Update beta for state j at time t
        return self.beta

    def initiate_emissions(self):
        """
        Initializes the emission probabilities matrix based on the Gaussian PDF.

        Returns:
            jnp.ndarray: The emission probabilities matrix.
        """
        emission_matrix = jnp.zeros(
            (self.N, self.observations.shape[1])
        )  # Initialize emission matrix
        for state in range(self.N):  # For each hidden state
            for obs in range(self.observations.shape[1]):  # For each observation
                emission_matrix = emission_matrix.at[state, obs].set(
                    self.gaussian_pdf(
                        self.observations[obs, :], self.means[state], self.cov[state]
                    )
                )  # Compute emission probability using Gaussian PDF
        emission_matrix = jnp.nan_to_num(emission_matrix, nan=0)  # Replace NaNs with 0
        emission_matrix = jnp.array(
            normalize(emission_matrix, "l1", axis=1)
        )  # Normalize rows
        return emission_matrix

    def baum_welch(self, threshold=0.001, verbose=True):
        """
        Performs the Baum-Welch algorithm to estimate the HMM parameters (transition and emission matrices).

        Args:
            threshold (float): Convergence threshold for parameter updates.
            verbose (bool): Whether to print intermediate results.

        Returns:
            None
        """
        new_transition = self.transition_matrix  # Initialize new transition matrix
        x = 0  # Iteration counter

        while True:
            self.theta = jnp.zeros(
                (self.T, self.N, self.N)
            )  # Initialize intermediate matrix
            self.posterior_probabilities()  # Compute posterior probabilities

            # Compute theta for each time step and state pair
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
                )  # Normalize

            # Update transition matrix
            for i in range(self.N):
                for j in range(self.N):
                    new_transition = new_transition.at[i, j].set(
                        jnp.sum(self.theta[:, i, j]) / jnp.sum(self.gamma[i, :-1])
                    )
            new_transition = jnp.nan_to_num(
                new_transition, nan=0
            )  # Replace NaNs with 0
            new_transition = jnp.array(
                normalize(new_transition, "l1", axis=1)
            )  # Normalize rows

            # Update emission parameters (means and covariances)
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

            x += 1  # Increment iteration counter

            # Check for convergence
            if jnp.max(jnp.abs(new_transition - self.transition_matrix)) < threshold:
                self.transition_matrix = new_transition
                self.emission_matrix = (
                    self.initiate_emissions()
                )  # Recompute emission probabilities
                if verbose:
                    print(f"Converged after {x} iterations.")
                break

            self.transition_matrix = new_transition
            self.emission_matrix = (
                self.initiate_emissions()
            )  # Recompute emission probabilities


"########################################################################################### Logging CallBack ###########################################################################################"


class Monitor(BaseCallback):
    """
    Custom callback for logging training metrics in reinforcement learning experiments.

    Features:
    - Tracks episode rewards and lengths
    - Logs metrics at specified intervals
    - Saves logs to CSV file
    - Maintains running averages for monitoring

    Args:
        save_to_csv (bool): Enable/disable CSV logging
        log_file (str): Path to output CSV file
        log_interval (int): Number of steps between logging events
        verbose (int): Verbosity level (0: silent, 1: show metrics)
    """

    def __init__(
        self,
        save_to_csv=True,
        log_file="ppo_custom_logs.csv",
        log_interval=1000,
        verbose=1,
    ):
        super().__init__(verbose)
        self.log_file = log_file
        self.save_to_csv = save_to_csv
        self.log_interval = log_interval
        self.episode_rewards = []  # Stores total rewards for logged episodes
        self.episode_lengths = []  # Stores durations for logged episodes
        self.current_reward = 0  # Accumulates rewards between log intervals
        self.current_length = 0  # Counts steps between log intervals

    def _on_step(self) -> bool:
        """
        Called at each environment step during training.

        Performs:
        - Reward accumulation
        - Interval-based logging
        - CSV data saving
        - Metric resetting

        Returns:
            bool: Always returns True to continue training
        """
        # Collect rewards from all parallel environments
        rewards = jnp.array(self.training_env.get_attr("reward"))

        # Aggregate rewards across parallel environments
        if rewards:
            # Use JAX's mean for accelerator compatibility
            self.current_reward += jnp.mean(rewards)
            self.current_length += 1

        # Logging condition (every N steps)
        if self.num_timesteps % self.log_interval == 0:
            # Store episode statistics
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)

            # Persist to CSV if enabled
            if self.save_to_csv:
                import pandas as pd  # type: ignore

                df = pd.DataFrame(
                    {
                        "episode_rewards": self.episode_rewards,
                        "episode_lengths": self.episode_lengths,
                    }
                )
                df.to_csv(self.log_file, index=False)

            # Reset accumulation buffers
            self.current_reward = 0
            self.current_length = 0

            # Print progress if verbose enabled
            if self.verbose > 0:
                avg_reward = np.mean(
                    self.episode_rewards[-10:]
                )  # 10-episode moving average
                print(
                    f"Step {self.num_timesteps} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Current Length: {self.current_length}"
                )

        return True  # Continue training


"########################################################################################### DQN ###########################################################################################"


class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self._train_dir = self.log_dir
        self._train_step = self.step
        self._should_write_train_graph = False
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            for name, value in stats.items():
                tf.summary.scalar(name, value, step=self.step)
            self.writer.flush()
        self.step += 1


class Agent:
    def __init__(
        self,
        env,
        input_shape,
        output_shape,
        episodes,
        epsilon_decay=0.9995,
        min_epsilon=0.001,
        replay_memory_size=500_000,
        min_train_size=130,
        minibatch_size=64,
        discount=0.995,
        update_every=10,
        aggregate_stats_every=50,
    ):
        """
        Initializes the Agent with the given parameters.

        Args:
            env: The environment in which the agent operates.
            input_shape: The shape of the input state.
            output_shape: The shape of the output action space.
            episodes: The total number of episodes for training.
            epsilon_decay: The decay rate for the exploration factor (epsilon).
            min_epsilon: The minimum value for epsilon.
            replay_memory_size: The maximum size of the replay memory.
            min_train_size: The minimum number of experiences required to start training.
            minibatch_size: The size of the minibatch used for training.
            discount: The discount factor for future rewards.
            update_every: The frequency (in episodes) at which the target model is updated.
            aggregate_stats_every: The frequency (in episodes) at which statistics are logged.
        """
        self.env = env  # The environment
        self.episodes = episodes  # Total number of episodes
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.min_epsilon = min_epsilon  # Minimum value for epsilon
        self.min_train_size = min_train_size  # Minimum experiences to start training
        self.minibatch_size = minibatch_size  # Size of the minibatch for training
        self.discount = discount  # Discount factor for future rewards
        self.update_every = update_every  # Frequency to update the target model
        self.aggregate_stats_every = (
            aggregate_stats_every  # Frequency to log statistics
        )

        # Create the main and target neural networks
        self.model = self.create_model(input_shape, output_shape)
        self.target_model = self.create_model(input_shape, output_shape)
        self.target_model.set_weights(
            self.model.get_weights()
        )  # Initialize target model weights

        # Initialize replay memory and counters
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.update_counter = 0  # Counter for target model updates
        self.tensorboard = ModifiedTensorBoard(
            log_dir="logs/{}-{}".format("agent", int(time()))
        )  # TensorBoard for logging

    def create_model(self, input_shape, output_shape):
        """
        Creates a neural network model for the agent.

        Args:
            input_shape: The shape of the input state.
            output_shape: The shape of the output action space.

        Returns:
            A compiled Keras model.
        """
        model = Sequential()
        model.add(Input(shape=input_shape))  # Input layer
        model.add(Dense(24, activation="relu"))  # First hidden layer
        model.add(Dropout(0.2))  # Dropout for regularization
        model.add(Dense(24, activation="relu"))  # Second hidden layer
        model.add(Dense(output_shape, activation="linear"))  # Output layer
        model.compile(
            loss="mse", optimizer=Adam(learning_rate=0.005), metrics=["accuracy"]
        )  # Compile the model
        return model

    def add_observation(self, observation):
        """
        Adds a new observation (experience) to the replay memory.

        Args:
            observation: A tuple containing (state, action, reward, done, new_state).
        """
        self.replay_memory.append(observation)

    def train(self, final_state):
        """
        Trains the agent using experiences from the replay memory.

        Args:
            final_state: Whether the current episode has ended.
        """
        if len(self.replay_memory) < self.min_train_size:
            return  # Do not train if there are not enough experiences

        # Sample a minibatch from the replay memory
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        current_states = []
        new_states = []
        for obs in minibatch:
            current_states.append(obs[0])  # Current state
            new_states.append(obs[4])  # New state after taking the action

        current_states, new_states = jnp.array(current_states), jnp.array(new_states)

        # Predict Q-values for current and new states
        current_qs = self.model.predict(current_states, verbose=0)
        new_qs = self.target_model.predict(new_states, verbose=0)

        x = []
        y = []
        for id, (state, action, reward, done, new_state) in enumerate(minibatch):
            new_q = reward  # Initialize Q-value with the immediate reward
            if not done:
                max_new_q = jnp.max(new_qs[id])  # Maximum Q-value for the new state
                new_q += self.discount * max_new_q  # Add discounted future reward
            current_q = current_qs[id]
            current_q[action] = new_q  # Update the Q-value for the taken action

            x.append(state)
            y.append(current_q)

        x, y = jnp.array(x), jnp.array(y)
        # Train the model on the minibatch
        self.model.fit(
            x,
            y,
            batch_size=self.minibatch_size,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard] if final_state else None,
        )

        if final_state:
            self.update_counter += 1  # Increment the update counter

        # Update the target model if the update counter exceeds the threshold
        if self.update_counter > self.update_every:
            self.update_counter = 0
            self.target_model.set_weights(self.model.get_weights())

    def get_best_action(self, state):
        """
        Returns the best action for the given state based on the current model.

        Args:
            state: The current state.

        Returns:
            The best action (index with the highest Q-value).
        """
        return jnp.argmax(
            self.model.predict(
                jnp.array(state).reshape((-1, state.shape[0])), verbose=0
            )[0]
        )

    def learn(self):
        """
        Trains the agent over multiple episodes using the Q-learning algorithm.
        """
        epsilon = 1  # Initial exploration factor
        ep_rewards = []  # List to store episode rewards

        for episode in tqdm(range(1, self.episodes + 1), ascii=True, unit="episodes"):
            self.tensorboard.step = episode  # Set the current episode for TensorBoard
            episode_reward = 0  # Initialize episode reward

            # Reset the environment and get the initial state
            if isinstance(self.env.observation_space, Discrete):
                current_state = jnp.array([self.env.reset()[0]])
            else:
                current_state = self.env.reset()[0]

            while True:
                # Choose an action using epsilon-greedy strategy
                if np.random.random() < epsilon:
                    action = self.get_best_action(
                        current_state
                    )  # Exploit: choose the best action
                else:
                    action = (
                        self.env.action_space.sample()
                    )  # Explore: choose a random action

                # Take the action and observe the result
                new_state, reward, done, truncuated, _ = self.env.step(int(action))
                if isinstance(self.env.observation_space, Discrete):
                    new_state = jnp.array([new_state])
                episode_reward += reward  # Accumulate the reward

                # Add the experience to the replay memory
                self.add_observation(
                    (current_state, action, reward, done or truncuated, new_state)
                )
                self.train(done)  # Train the model

                current_state = new_state  # Update the current state
                ep_rewards.append(episode_reward)  # Store the episode reward

                if done or truncuated:
                    break  # End the episode if done or truncated

            ep_rewards.append(episode_reward)  # Store the final episode reward

            # Log statistics periodically
            if not episode % self.aggregate_stats_every or episode == 1:
                average_reward = sum(ep_rewards[-self.aggregate_stats_every :]) / len(
                    ep_rewards[-self.aggregate_stats_every :]
                )
                min_reward = min(ep_rewards[-self.aggregate_stats_every :])
                max_reward = max(ep_rewards[-self.aggregate_stats_every :])
                self.tensorboard.update_stats(
                    reward_avg=average_reward,
                    reward_min=min_reward,
                    reward_max=max_reward,
                    epsilon=epsilon,
                )

                # Save the model if it achieves a high reward
                self.model.save(
                    f"models/agent__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time())}.keras"
                )

            # Decay epsilon to reduce exploration over time
            if epsilon > self.min_epsilon:
                epsilon *= self.epsilon_decay
                epsilon = max(self.min_epsilon, epsilon)
