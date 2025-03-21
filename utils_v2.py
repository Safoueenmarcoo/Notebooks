import jax.numpy as jnp  # type: ignore
from jax import jacfwd  # type: ignore
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
from typing import Any, Tuple, Optional, Union, List, Dict
from jax import jit as jjit  # type: ignore
from numba import jit as njit  # type: ignore

"########################################################################################### Control ###########################################################################################"


@jjit
def is_stable(A: jnp.ndarray) -> bool:
    """
    Check if a matrix A is stable.
    A matrix is considered stable if all its eigenvalues have negative real parts.

    Args:
        A (jnp.ndarray): The matrix to check.

    Returns:
        bool: True if the matrix is stable, False otherwise.
    """
    eigenvalues = jnp.linalg.eigvals(A)
    stable = bool(jnp.all(jnp.real(eigenvalues) < 0))
    if not stable:
        print(jnp.real(eigenvalues))
    return stable


@jjit
def controllability_matrix(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the controllability matrix of a system defined by matrices A and B.

    Args:
        A (jnp.ndarray): The state matrix.
        B (jnp.ndarray): The input matrix.

    Returns:
        jnp.ndarray: The controllability matrix.
    """
    n = A.shape[0]
    C = B
    for i in range(1, n):
        C = jnp.hstack((C, jnp.linalg.matrix_power(A, i) @ B))
    return C


@jjit
def is_controllable(A: jnp.ndarray, B: jnp.ndarray) -> bool:
    """
    Check if a system defined by matrices A and B is controllable.

    Args:
        A (jnp.ndarray): The state matrix.
        B (jnp.ndarray): The input matrix.

    Returns:
        bool: True if the system is controllable, False otherwise.
    """
    C = controllability_matrix(A, B)
    rank = jnp.linalg.matrix_rank(C)
    return bool(rank == A.shape[0])


@jjit
def observability_matrix(A: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the observability matrix of a system defined by matrices A and C.

    Args:
        A (jnp.ndarray): The state matrix.
        C (jnp.ndarray): The output matrix.

    Returns:
        jnp.ndarray: The observability matrix.
    """
    n = A.shape[0]
    Obs = C
    for i in range(1, n):
        Obs = jnp.vstack((Obs, C @ jnp.linalg.matrix_power(A, i)))
    return Obs


@jjit
def is_observable(A: jnp.ndarray, C: jnp.ndarray) -> bool:
    """
    Check if a system defined by matrices A and C is observable.

    Args:
        A (jnp.ndarray): The state matrix.
        C (jnp.ndarray): The output matrix.

    Returns:
        bool: True if the system is observable, False otherwise.
    """
    Obs = observability_matrix(A, C)
    rank = jnp.linalg.matrix_rank(Obs)
    return bool(rank == A.shape[0])


"########################################################################################### Kalman Filter ###########################################################################################"


@njit(nopython=True)
def KalmanFilter_1D(
    est: float, mea_err: float, measurements: List[float]
) -> List[float]:
    """
    Perform 1D Kalman filtering on a sequence of measurements.

    Args:
        est (float): Initial estimate.
        mea_err (float): Measurement error.
        measurements (List[float]): List of measurements.

    Returns:
        List[float]: List of filtered estimates.
    """
    ests = []
    err = 1
    for i in range(len(measurements)):
        KG = err / (err + mea_err)
        est = est + KG * (measurements[i] - est)
        err = (1 - KG) * err
        ests.append(est)
    return ests


class KalmanFilter:
    """
    Implements a discrete-time Kalman Filter for state estimation in linear dynamic systems.

    The filter operates in two stages:
    1. Prediction - estimates the next state using system dynamics.
    2. Update - corrects estimates using noisy measurements.

    Attributes:
        x_k (jnp.ndarray): Current state estimate vector (shape n x 1).
        A (jnp.ndarray): State transition matrix (shape n x n).
        B (jnp.ndarray): Control input matrix (shape n x m).
        H (jnp.ndarray): Observation matrix (shape p x n).
        C (jnp.ndarray): Output matrix for measurements (shape p x n).
        R (jnp.ndarray): Measurement noise covariance matrix (shape p x p).
        Q (jnp.ndarray): Process noise covariance matrix (shape n x n).
        Z (jnp.ndarray): Measurement noise vector (shape p x 1).
        w_k (jnp.ndarray): Process noise vector (shape n x 1).
        P (jnp.ndarray): Error covariance matrix (shape n x n).
        P_0 (jnp.ndarray): Initial error covariance matrix (shape n x n).
        K (jnp.ndarray): Kalman gain matrix from last update (shape n x p).
    """

    def __init__(
        self,
        x_0: jnp.ndarray,
        A: jnp.ndarray,
        B: jnp.ndarray,
        H: jnp.ndarray,
        C: jnp.ndarray,
        R: jnp.ndarray,
        Q: jnp.ndarray,
        Z: jnp.ndarray,
        w_k: jnp.ndarray,
        P_0: jnp.ndarray,
    ):
        """
        Initialize Kalman Filter with system parameters and initial state.

        Args:
            x_0 (jnp.ndarray): Initial state estimate (n x 1).
            A (jnp.ndarray): State transition matrix (n x n).
            B (jnp.ndarray): Control input matrix (n x m).
            H (jnp.ndarray): Observation matrix (p x n).
            C (jnp.ndarray): Output matrix for measurements (p x n).
            R (jnp.ndarray): Measurement noise covariance matrix (p x p).
            Q (jnp.ndarray): Process noise covariance matrix (n x n).
            Z (jnp.ndarray): Measurement noise vector (p x 1).
            w_k (jnp.ndarray): Process noise vector (n x 1).
            P_0 (jnp.ndarray): Initial error covariance matrix (n x n).

        Raises:
            RuntimeError: If any matrix or vector does not match the expected shape.
        """
        self.x_0 = x_0.reshape((-1, 1))
        self.x_k = self.x_0
        self.A = A
        self.B = B
        self.H = H
        self.C = C
        self.R = R
        self.Q = Q
        self.Z = Z
        self.w_k = w_k
        self.P = P_0
        self.P_0 = P_0
        assert self._verify_matrices()

    def _verify_matrices(self):
        """
        Verify that all matrices and vectors have the correct shapes.

        Returns:
            bool: True if all shapes are valid.

        Raises:
            RuntimeError: If any matrix or vector does not match the expected shape.
        """
        x_0_shape = self.x_0.shape[0]
        if self.A.shape[0] != self.A.shape[1] or self.A.shape[0] != x_0_shape:
            raise RuntimeError(
                f"The State transition matrix A must be a square matrix with shape ({x_0_shape}, {x_0_shape}), got {self.A.shape}!"
            )
        if self.B.shape[0] != x_0_shape:
            raise RuntimeError(
                f"The Control input matrix B must have the same number of rows as the initial state estimate ({x_0_shape}, m), got {self.B.shape}!"
            )
        if self.H.shape[1] != x_0_shape:
            raise RuntimeError(
                f"The Observation matrix H must have the same number of columns as the initial state estimate ({self.H.shape[0]}, {x_0_shape}), got {self.H.shape}!"
            )
        if self.C.shape != self.H.shape:
            raise RuntimeError(
                f"The Output matrix for measurements C must have the same shape as the Observation matrix ({self.H.shape}), got {self.C.shape}!"
            )
        if self.R.shape[0] != self.R.shape[1] or self.R.shape[0] != self.C.shape[0]:
            raise RuntimeError(
                f"The Measurement noise covariance R must be a square matrix with shape ({self.C.shape[0]}, {self.C.shape[0]}), got {self.R.shape}!"
            )
        if self.Q.shape[0] != self.Q.shape[1] or self.Q.shape[0] != x_0_shape:
            raise RuntimeError(
                f"The Process noise covariance Q must be a square matrix with shape ({x_0_shape}, {x_0_shape}), got {self.Q.shape}!"
            )
        if self.Z.shape != (self.H.shape[0], 1):
            raise RuntimeError(
                f"The Measurement noise Z must be a column vector with shape ({self.H.shape[0]}, 1), got {self.Z.shape}!"
            )
        if self.w_k.shape != (self.A.shape[0], 1):
            raise RuntimeError(
                f"The Process noise vector w_k must be a column vector with shape ({x_0_shape}, 1), got {self.w_k.shape}!"
            )
        if self.P_0.shape != self.A.shape:
            raise RuntimeError(
                f"The Initial error covariance P_0 must be a square matrix with shape ({x_0_shape}, {x_0_shape}), got {self.P_0.shape}!"
            )
        return True

    def reset(self) -> None:
        """
        Reset the filter to the initial state and covariance.

        This is useful when reusing the filter for a new sequence of data.
        """
        self.x_k = self.x_0
        self.P = self.P_0

    def _step_estimation(self, u_k: jnp.ndarray) -> jnp.ndarray:
        """
        Predict the next state using system dynamics and control input.

        Args:
            u_k (jnp.ndarray): Control input vector (m x 1).

        Returns:
            jnp.ndarray: Predicted state vector (n x 1).

        Raises:
            RuntimeError: If an error occurs during state prediction.
        """
        try:
            x_k = self.x_k.reshape((-1, 1))
            new_x_k = self.A @ x_k + self.B @ u_k + self.w_k
            return new_x_k
        except Exception as e:
            raise RuntimeError(f"Error in the step estimation function: {e}") from e

    def _process_covariance(self) -> jnp.ndarray:
        """
        Update the error covariance matrix using system dynamics and process noise.

        Returns:
            jnp.ndarray: Updated covariance matrix (n x n).

        Raises:
            RuntimeError: If an error occurs during covariance update.
        """
        try:
            new_P = self.A @ self.P @ self.A.T + self.Q
            return new_P
        except Exception as e:
            raise RuntimeError(f"Error in the process covariance function: {e}") from e

    def _kalman_function(self) -> jnp.ndarray:
        """
        Compute the optimal Kalman gain using current covariance estimates.

        Returns:
            jnp.ndarray: Kalman gain matrix (n x p).

        Raises:
            RuntimeError: If an error occurs during Kalman gain computation.
        """
        try:
            x = self.P @ self.H.T
            K = x @ jnp.linalg.inv(self.H @ x + self.R)
            K = jnp.nan_to_num(K, nan=0)
            return K
        except Exception as e:
            raise RuntimeError(f"Error in the Kalman gain function: {e}") from e

    def _current_state_and_process(
        self, x_km: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Update the state estimate using a measurement and the Kalman gain.

        Args:
            x_km (jnp.ndarray): Noisy measurement vector (shape p x 1).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                - Corrected state estimate (shape n x 1).
                - Updated error covariance matrix (shape n x n).

        Raises:
            RuntimeError: If an error occurs during state update.
        """
        try:
            measurements = self.C @ x_km.reshape((-1, 1)) + self.Z
            x_k = self.x_k + self.K @ (measurements - self.H @ self.x_k)
            p_k = (jnp.eye(self.K.shape[0]) - self.K @ self.H) @ self.P
            return x_k, p_k
        except Exception as e:
            raise RuntimeError(
                f"Error in the new state and process calculation function: {e}"
            ) from e

    def predict(self, u_k: jnp.ndarray) -> jnp.ndarray:
        """
        Predict the next state based on the control input and process model.

        Args:
            u_k (jnp.ndarray): Control input vector (m x 1).

        Returns:
            jnp.ndarray: Updated state estimate vector (n x 1).

        Raises:
            RuntimeError: If the control input vector has an invalid shape or an error occurs during prediction.
        """
        if u_k.shape != (self.B.shape[1], 1):
            raise RuntimeError(
                f"The Control input vector u_k must be a column vector with shape ({self.B.shape[1]}, 1), got {u_k.shape}!"
            )
        try:
            u_k = u_k.reshape((-1, 1))
            self.x_k = self._step_estimation(u_k).squeeze()
            self.P = self._process_covariance()
            return self.x_k
        except Exception as e:
            raise RuntimeError(f"Error in the predict method: {e}") from e

    def update(self, x_km: jnp.ndarray) -> jnp.ndarray:
        """
        Update the state estimate based on the new measurement.

        Args:
            x_km (jnp.ndarray): Measured state vector (shape n x 1).

        Returns:
            jnp.ndarray: Updated state estimate vector (shape n x 1).

        Raises:
            RuntimeError: If the measurement vector has an invalid shape or an error occurs during update.
        """
        if x_km.shape != (self.x_0.shape[0], 1):
            raise RuntimeError(
                f"The Measured state vector x_km must be a column vector with shape ({self.x_0.shape[0]}, 1), got {x_km.shape}!"
            )
        try:
            self.x_k = self.x_k.reshape((-1, 1))
            self.K = self._kalman_function()
            self.x_k, self.P = self._current_state_and_process(x_km)
            return self.x_k.squeeze()
        except Exception as e:
            raise RuntimeError(f"Error in the update method: {e}") from e


"########################################################################################### Extended Kalman Filter ###########################################################################################"


class ExtendedKalmanFilter(KalmanFilter):
    """
    Implements an Extended Kalman Filter (EKF) for state estimation in nonlinear systems.

    Attributes:
        x_k (jnp.ndarray): Current state estimate (shape n x 1).
        f (callable): Nonlinear state transition function, f(x, u), where:
            - x is the state vector (n x 1).
            - u is the control input vector (m x 1).
            - Returns the predicted state vector (n x 1).
        h (callable): Nonlinear measurement function, h(x), where:
            - x is the state vector (n x 1).
            - Returns the predicted measurement vector (p x 1).
        R (jnp.ndarray): Measurement noise covariance matrix (p x p).
        Q (jnp.ndarray): Process noise covariance matrix (n x n).
        Z (jnp.ndarray): Measurement noise vector (p x 1).
        w_k (jnp.ndarray): Process noise vector (n x 1).
        P (jnp.ndarray): Error covariance matrix (n x n).
        _function_f (callable): Jacobian of the state transition function, f(x, u).
        _function_h (callable): Jacobian of the measurement function, h(x).
    """

    def __init__(
        self,
        x_0: jnp.ndarray | float | int,
        f: callable,
        h: callable,
        R: jnp.ndarray,
        Q: jnp.ndarray,
        Z: jnp.ndarray,
        w_k: jnp.ndarray,
        P_0: jnp.ndarray,
        jaccobian_f: callable = None,
        jaccobian_h: callable = None,
    ) -> None:
        """
        Initializes the Extended Kalman Filter with the given parameters.

        Args:
            x_0 (jnp.ndarray | float | int): Initial state estimate. If scalar, it is expanded to a 1D array (shape n x 1).
            f (callable): Nonlinear state transition function, f(x, u).
            h (callable): Nonlinear measurement function, h(x).
            R (jnp.ndarray): Measurement noise covariance matrix (shape p x p).
            Q (jnp.ndarray): Process noise covariance matrix (shape n x n).
            Z (jnp.ndarray): Measurement noise vector (shape p x 1).
            w_k (jnp.ndarray): Process noise vector (shape n x 1).
            P_0 (jnp.ndarray): Initial error covariance matrix (shape n x n).
            jaccobian_f (callable, optional): Jacobian of the state transition function, f(x, u). If not provided, it is computed numerically.
            jaccobian_h (callable, optional): Jacobian of the measurement function, h(x). If not provided, it is computed numerically.

        Raises:
            Exception: If the initial state `x_0` is not of type `jnp.ndarray`, `float`, or `int`.
        """
        if not isinstance(x_0, (jnp.ndarray, float, int)):
            raise Exception(
                "The State input must be of type jnp.ndarray, float, or int"
            )
        # Expand scalar state into array if necessary
        x_0 = jnp.expand_dims(x_0, axis=-1) if type(x_0) in (int, float) else x_0
        super().__init__(x_0, None, None, None, None, R, Q, Z, w_k, P_0)
        if jaccobian_f is not None:
            self._function_f = jaccobian_f
            self._set_matrix_f = self._set_none
            self.A = self._function_f(self.x_k)

        else:
            self._set_matrix_f = self._matrix_f

        self.f = f  # Nonlinear state transition function: f(x, u)

        if jaccobian_h is not None:
            self._function_h = jaccobian_h
            self._set_matrix_h = self._set_none
            self.H = self._function_h(self.x_k)
        else:
            self._set_matrix_h = self._matrix_h

        self.h = h  # Nonlinear measurement function: h(x)
        assert self._verify_matrices()
    
    def _verify_matrices(self):
        """
        Verify that all matrices and vectors have the correct shapes.

        Returns:
            bool: True if all shapes are valid.

        Raises:
            RuntimeError: If any matrix or vector does not match the expected shape.
        """
        x_0_shape = self.x_0.shape[0]
        if self.R.shape[0] != self.R.shape[1] or self.R.shape[0] != self.H.shape[0]:
            raise RuntimeError(
                f"The Measurement noise covariance R must be a square matrix with shape ({self.H.shape[0]}, {self.H.shape[0]}), got {self.R.shape}!"
            )
        if self.Q.shape[0] != self.Q.shape[1] or self.Q.shape[0] != x_0_shape:
            raise RuntimeError(
                f"The Process noise covariance Q must be a square matrix with shape ({x_0_shape}, {x_0_shape}), got {self.Q.shape}!"
            )
        if self.Z.shape != (self.H.shape[0], 1):
            raise RuntimeError(
                f"The Measurement noise Z must be a column vector with shape ({self.H.shape[0]}, 1), got {self.Z.shape}!"
            )
        if self.w_k.shape != (x_0_shape, 1):
            raise RuntimeError(
                f"The Process noise vector w_k must be a column vector with shape ({x_0_shape}, 1), got {self.w_k.shape}!"
            )
        if self.P_0.shape != (x_0_shape, x_0_shape):
            raise RuntimeError(
                f"The Initial error covariance P_0 must be a square matrix with shape ({x_0_shape}, {x_0_shape}), got {self.P_0.shape}!"
            )
        return True
    
    def _set_none(self, u_k: jnp.ndarray = None) -> None:
        """
        Placeholder function for cases where Jacobians are precomputed and do not need updating.

        Args:
            u_k (jnp.ndarray, optional): Control input vector (m x 1). Defaults to None.
        """
        pass

    def _jacobian(
        self, f: callable, x: jnp.ndarray, u: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Computes the Jacobian of a vector-valued function f at x.

        Args:
            f (callable): Function to compute the Jacobian for.
            x (jnp.ndarray): State vector (n x 1).
            u (jnp.ndarray, optional): Control input vector (m x 1). Defaults to None.

        Returns:
            jnp.ndarray: Jacobian matrix of f evaluated at x (and u if provided).
        """
        jac_F = jacfwd(f)  # Forward-mode Jacobian
        return jnp.array(jac_F(x)) if u is None else jnp.array(jac_F(x, u))

    def _matrix_f(self, x, u):
        """
        Computes the Jacobian of the state transition function f(x, u).

        Args:
            x (jnp.ndarray): State vector (n x 1).
            u (jnp.ndarray): Control input vector (m x 1).

        Returns:
            jnp.ndarray: Jacobian matrix of f (n x n).
        """
        return self._jacobian(self.f, x, u)  # State transition Jacobian

    def _matrix_h(self, x):
        """
        Computes the Jacobian of the measurement function h(x).

        Args:
            x (jnp.ndarray): State vector (n x 1).

        Returns:
            jnp.ndarray: Jacobian matrix of h (p x n).
        """
        return self._jacobian(self.h, x)  # Measurement Jacobian

    def _step_estimation(self, u_k: jnp.ndarray) -> jnp.ndarray:
        """
        Predicts the next state using the nonlinear system dynamics f and control input u_k.

        Args:
            u_k (jnp.ndarray): Control input vector (m x 1).

        Returns:
            jnp.ndarray: Predicted state vector (n x 1).

        Raises:
            RuntimeError: If an error occurs during the prediction.
        """
        try:
            self._set_matrix_f(u_k)
            self._set_matrix_h()
            new_x_k = self.f(self.x_k, u_k) + self.w_k
            return new_x_k
        except Exception as e:
            raise RuntimeError(f"Error in EKF step estimation:{e}") from e

    def _current_state_and_process(
        self, x_km: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Updates the state estimate using a measurement and the Kalman gain.

        This function recomputes the linearization matrices A and H based on the current state.

        Args:
            x_km (jnp.ndarray): Noisy measurement vector (p x 1).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                - Corrected state estimate (n x 1).
                - Updated error covariance matrix (n x n).

        Raises:
            RuntimeError: If an error occurs during state update.
        """
        try:
            # Recompute linearization at current state
            self.A = self._function_f(self.x_k)
            self.H = self._function_h(self.x_k)
            x_k = self.x_k + self.K @ (x_km - self.h(self.x_k) + self.Z)
            p_k = (jnp.eye(self.K.shape[0]) - self.K @ self.H) @ self.P
            return x_k, p_k
        except Exception as e:
            raise RuntimeError(f"Error in EKF state and process update:{e}") from e


"########################################################################################### Kalman Filter + RL ###########################################################################################"


class KalmanRLWrapper(gym.Env):
    """
    A wrapper for combining a reinforcement learning environment with a Kalman Filter.

    Parameters:
        env: The underlying RL environment to be wrapped.
        kalman_filter: An instance of the KalmanFilter class for state estimation.
    """

    def __init__(self, env: gym.Env, kalman_filter: KalmanFilter):
        super(KalmanRLWrapper, self).__init__()
        self.env = env
        self.kf = kalman_filter
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(
        self, seed: Optional[int] = None, **kwargs
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Reset the environment and initialize the Kalman Filter state.

        Args:
            seed: Seed for random number generation in the environment.
            **kwargs: Additional arguments for the environment's reset method.

        Returns:
            Tuple[jnp.ndarray, Dict[str, Any]]: Corrected initial observation and info dictionary.
        """
        noisy_obs = self.env.reset(seed=seed, **kwargs)[0]
        self.kf.x_k = noisy_obs
        return self.kf.x_k, {}

    def step(
        self, action: int
    ) -> Tuple[jnp.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment and update the Kalman Filter state.

        Args:
            action: The action to take in the environment.

        Returns:
            Tuple[jnp.ndarray, float, bool, bool, Dict[str, Any]]: Corrected observation, reward, done flag, truncation flag, and info dictionary.
        """
        noisy_obs, reward, done, truncuated, info = self.env.step(action)
        action = -1 if action == 0 else 1
        u_k = jnp.array([[action]])
        self.kf.predict(u_k)
        corrected_obs = self.kf.update(noisy_obs)
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
        observations: jnp.ndarray,
        labels: Union[List[Any], jnp.ndarray],
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


"########################################################################################### CHMM(not completed) ###########################################################################"


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
        save_to_csv: bool = True,
        log_file: str = "ppo_custom_logs.csv",
        log_interval: int = 1000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_file = log_file
        self.save_to_csv = save_to_csv
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_reward = 0
        self.current_length = 0

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
        rewards = jnp.array(self.training_env.get_attr("reward"))
        if rewards:
            self.current_reward += jnp.mean(rewards)
            self.current_length += 1
        if self.num_timesteps % self.log_interval == 0:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            if self.save_to_csv:
                import pandas as pd  # type: ignore

                df = pd.DataFrame(
                    {
                        "episode_rewards": self.episode_rewards,
                        "episode_lengths": self.episode_lengths,
                    }
                )
                df.to_csv(self.log_file, index=False)
            self.current_reward = 0
            self.current_length = 0
            if self.verbose > 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(
                    f"Step {self.num_timesteps} | Avg Reward: {avg_reward:.2f} | Current Length: {self.current_length}"
                )
        return True


"########################################################################################### DQN ###########################################################################################"


class ModifiedTensorBoard(TensorBoard):
    """
    Custom TensorBoard callback for logging training metrics.

    Overrides default behavior to allow for custom logging and step management.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self._train_dir = self.log_dir
        self._train_step = self.step
        self._should_write_train_graph = False
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def set_model(self, model: Any) -> None:
        """
        Overrides the default set_model method to prevent creating a default log writer.
        """
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of each epoch to update training statistics.

        Args:
            epoch (int): The current epoch number.
            logs (Optional[Dict[str, Any]]): Dictionary of metrics to log.
        """
        self.update_stats(**logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of each batch. No action is taken here.
        """
        pass

    def on_train_end(self, _: Any) -> None:
        """
        Called at the end of training. No action is taken here.
        """
        pass

    def update_stats(self, **stats: Any) -> None:
        """
        Updates TensorBoard with custom statistics.

        Args:
            **stats (Any): Key-value pairs of statistics to log.
        """
        with self.writer.as_default():
            for name, value in stats.items():
                tf.summary.scalar(name, value, step=self.step)
            self.writer.flush()
        self.step += 1


class Agent:
    """
    A Deep Q-Network (DQN) agent for reinforcement learning.

    Attributes:
        env (gym.Env): The environment in which the agent operates.
        episodes (int): The total number of episodes for training.
        epsilon_decay (float): The decay rate for the exploration factor (epsilon).
        min_epsilon (float): The minimum value for epsilon.
        min_train_size (int): The minimum number of experiences required to start training.
        minibatch_size (int): The size of the minibatch used for training.
        discount (float): The discount factor for future rewards.
        update_every (int): The frequency (in episodes) at which the target model is updated.
        aggregate_stats_every (int): The frequency (in episodes) at which statistics are logged.
        model (Sequential): The main neural network model.
        target_model (Sequential): The target neural network model.
        replay_memory (deque): The replay memory for storing experiences.
        update_counter (int): Counter for target model updates.
        tensorboard (ModifiedTensorBoard): TensorBoard for logging.
    """

    def __init__(
        self,
        env: gym.Env,
        input_shape: Tuple[int, ...],
        output_shape: int,
        episodes: int,
        epsilon_decay: float = 0.9995,
        min_epsilon: float = 0.001,
        replay_memory_size: int = 500_000,
        min_train_size: int = 130,
        minibatch_size: int = 64,
        discount: float = 0.995,
        update_every: int = 10,
        aggregate_stats_every: int = 50,
    ):
        """
        Initializes the Agent with the given parameters.

        Args:
            env (gym.Env): The environment in which the agent operates.
            input_shape (Tuple[int, ...]): The shape of the input state.
            output_shape (int): The shape of the output action space.
            episodes (int): The total number of episodes for training.
            epsilon_decay (float): The decay rate for the exploration factor (epsilon).
            min_epsilon (float): The minimum value for epsilon.
            replay_memory_size (int): The maximum size of the replay memory.
            min_train_size (int): The minimum number of experiences required to start training.
            minibatch_size (int): The size of the minibatch used for training.
            discount (float): The discount factor for future rewards.
            update_every (int): The frequency (in episodes) at which the target model is updated.
            aggregate_stats_every (int): The frequency (in episodes) at which statistics are logged.
        """
        self.env = env
        self.episodes = episodes
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.min_train_size = min_train_size
        self.minibatch_size = minibatch_size
        self.discount = discount
        self.update_every = update_every
        self.aggregate_stats_every = aggregate_stats_every
        self.model = self.create_model(input_shape, output_shape)
        self.target_model = self.create_model(input_shape, output_shape)
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.update_counter = 0
        self.tensorboard = ModifiedTensorBoard(
            log_dir="logs/{}-{}".format("agent", int(time()))
        )

    def create_model(
        self, input_shape: Tuple[int, ...], output_shape: int
    ) -> Sequential:
        """
        Creates a neural network model for the agent.

        Args:
            input_shape (Tuple[int, ...]): The shape of the input state.
            output_shape (int): The shape of the output action space.

        Returns:
            Sequential: A compiled Keras model.
        """
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Dense(24, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(output_shape, activation="linear"))
        model.compile(
            loss="mse", optimizer=Adam(learning_rate=0.005), metrics=["accuracy"]
        )
        return model

    def add_observation(
        self, observation: Tuple[jnp.ndarray, int, float, bool, jnp.ndarray]
    ) -> None:
        """
        Adds a new observation (experience) to the replay memory.

        Args:
            observation (Tuple[jnp.ndarray, int, float, bool, jnp.ndarray]): A tuple containing (state, action, reward, done, new_state).
        """
        self.replay_memory.append(observation)

    def train(self, final_state: bool) -> None:
        """
        Trains the agent using experiences from the replay memory.

        Args:
            final_state (bool): Whether the current episode has ended.
        """
        if len(self.replay_memory) < self.min_train_size:
            return
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        current_states = []
        new_states = []
        for obs in minibatch:
            current_states.append(obs[0])
            new_states.append(obs[4])
        current_states, new_states = jnp.array(current_states), jnp.array(new_states)
        current_qs = self.model.predict(current_states, verbose=0)
        new_qs = self.target_model.predict(new_states, verbose=0)
        x = []
        y = []
        for id, (state, action, reward, done, new_state) in enumerate(minibatch):
            new_q = reward
            if not done:
                max_new_q = jnp.max(new_qs[id])
                new_q += self.discount * max_new_q
            current_q = current_qs[id]
            current_q[action] = new_q
            x.append(state)
            y.append(current_q)
        x, y = jnp.array(x), jnp.array(y)
        self.model.fit(
            x,
            y,
            batch_size=self.minibatch_size,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard] if final_state else None,
        )
        if final_state:
            self.update_counter += 1
        if self.update_counter > self.update_every:
            self.update_counter = 0
            self.target_model.set_weights(self.model.get_weights())

    def get_best_action(self, state: jnp.ndarray) -> int:
        """
        Returns the best action for the given state based on the current model.

        Args:
            state (jnp.ndarray): The current state.

        Returns:
            int: The best action (index with the highest Q-value).
        """
        return jnp.argmax(
            self.model.predict(
                jnp.array(state).reshape((-1, state.shape[0])), verbose=0
            )[0]
        )

    def learn(self) -> None:
        """
        Trains the agent over multiple episodes using the Q-learning algorithm.
        """
        epsilon = 1
        ep_rewards = []
        for episode in tqdm(range(1, self.episodes + 1), ascii=True, unit="episodes"):
            self.tensorboard.step = episode
            episode_reward = 0
            if isinstance(self.env.observation_space, Discrete):
                current_state = jnp.array([self.env.reset()[0]])
            else:
                current_state = self.env.reset()[0]
            while True:
                if np.random.random() < epsilon:
                    action = self.get_best_action(current_state)
                else:
                    action = self.env.action_space.sample()
                new_state, reward, done, truncuated, _ = self.env.step(int(action))
                if isinstance(self.env.observation_space, Discrete):
                    new_state = jnp.array([new_state])
                episode_reward += reward
                self.add_observation(
                    (current_state, action, reward, done or truncuated, new_state)
                )
                self.train(done)
                current_state = new_state
                ep_rewards.append(episode_reward)
                if done or truncuated:
                    break
            ep_rewards.append(episode_reward)
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
                self.model.save(
                    f"models/agent__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time())}.keras"
                )
            if epsilon > self.min_epsilon:
                epsilon *= self.epsilon_decay
                epsilon = max(self.min_epsilon, epsilon)
