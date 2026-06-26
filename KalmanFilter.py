from jax import jacfwd  # type: ignore
from typing import Tuple, Optional, List
import jax.numpy as jnp  # type: ignore
import jax.random as jrandom  # type: ignore
from time import time as ttime  # type: ignore


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
        msg = "Unstable"
    else:
        msg = "Stable"
    print(msg + " matrix with eigen values : " + jnp.real(eigenvalues))
    return stable


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


"################################################################################# Kalman Filter #################################################################################"


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
    1. Prediction: propagates the current state estimate using the system dynamics.
    2. Update: corrects the predicted estimate using noisy measurements.

    The filter supports automatic generation of process noise vectors,
    measurement noise vectors, and the initial covariance matrix when
    they are not explicitly provided.

    Notation:
        n : Dimension of the state vector.
        m : Dimension of the control input vector.
        p : Dimension of the measurement (observation) vector.

    Attributes:
        x_0 (jnp.ndarray):
            Initial state estimate vector (n × 1).

        x_k (jnp.ndarray):
            Current state estimate vector (n × 1).

        A (jnp.ndarray):
            State transition matrix (n × n).

        B (jnp.ndarray):
            Control input matrix (n × m).

        H (jnp.ndarray):
            Observation matrix (p × n).

        C (jnp.ndarray):
            Measurement output matrix (p × n).

        R (jnp.ndarray):
            Measurement noise covariance matrix (p × p).

        Q (jnp.ndarray):
            Process noise covariance matrix (n × n).

        Z (jnp.ndarray):
            Measurement noise vector (p × 1).
            Randomly generated from N(0, R) when not provided.

        w_k (jnp.ndarray):
            Process noise vector (n × 1).
            Randomly generated from N(0, Q) when not provided.

        P_0 (jnp.ndarray):
            Initial error covariance matrix (n × n).
            Initialized with ones when not provided.

        P (jnp.ndarray):
            Current error covariance matrix (n × n).

        K (jnp.ndarray):
            Kalman gain matrix from the latest update step.

        __Zrandom (bool):
            Indicates whether measurement noise is generated
            automatically at each update step.

        __w_krandom (bool):
            Indicates whether process noise is generated
            automatically at each prediction step.
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
        Z: Optional[jnp.ndarray] = None,
        w_k: Optional[jnp.ndarray] = None,
        P_0: Optional[jnp.ndarray] = None,
    ) -> None:
        """
        Initialize a Kalman Filter instance.

        Args:
            x_0 (jnp.ndarray):
                Initial state estimate (n × 1).

            A (jnp.ndarray):
                State transition matrix (n × n).

            B (jnp.ndarray):
                Control input matrix (n × m).

            H (jnp.ndarray):
                Observation matrix (p × n).

            C (jnp.ndarray):
                Measurement output matrix (p × n).

            R (jnp.ndarray):
                Measurement noise covariance matrix (p × p).

            Q (jnp.ndarray):
                Process noise covariance matrix (n × n).

            Z (jnp.ndarray, optional):
                Measurement noise vector (p × 1).
                If None, a random sample is drawn from N(0, R).

            w_k (jnp.ndarray, optional):
                Process noise vector (n × 1).
                If None, a random sample is drawn from N(0, Q).

            P_0 (jnp.ndarray, optional):
                Initial error covariance matrix (n × n).
                If None, a matrix of ones with shape (n, n) is used.

        Raises:
            RuntimeError:
                If any matrix or vector has an invalid shape.
        """
        self.x_0: jnp.ndarray = x_0.reshape((-1, 1))
        self.x_k: jnp.ndarray = self.x_0

        self.A: jnp.ndarray = A
        self.B: jnp.ndarray = B
        self.H: jnp.ndarray = H
        self.C: jnp.ndarray = C

        self.R: jnp.ndarray = R
        self.Q: jnp.ndarray = Q

        if Z is None:
            self.Z: jnp.ndarray = self.__MVNrandom(
                mean=jnp.zeros(self.R.shape[0]),
                cov=self.R,
            )
            self.__Zrandom: bool = True
        else:
            self.Z: jnp.ndarray = Z
            self.__Zrandom: bool = False

        if w_k is None:
            self.w_k: jnp.ndarray = self.__MVNrandom(
                mean=jnp.zeros(self.A.shape[0]),
                cov=self.Q,
            )
            self.__w_krandom: bool = True
        else:
            self.w_k: jnp.ndarray = w_k
            self.__w_krandom: bool = False

        if P_0 is None:
            self.P_0: jnp.ndarray = jnp.ones(self.A.shape)
        else:
            self.P_0: jnp.ndarray = P_0

        self.P: jnp.ndarray = self.P_0
        if type(self) is KalmanFilter:
            assert self.__verify_matrices()

    def __MVNrandom(self, mean: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
        return jrandom.multivariate_normal(
            jrandom.PRNGKey(int(ttime())), mean, cov
        ).reshape((-1, 1))

    def __verify_matrices(self) -> bool:
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
                f"The State transition matrix A must be a square matrix with shape ({x_0_shape}, {x_0_shape}) as the initial state estimate, got {self.A.shape}!"
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
            if self.__w_krandom:
                self.w_k = self.__MVNrandom(
                    mean=jnp.zeros(self.A.shape[0]),
                    cov=self.Q,
                )
            new_x_k = self.A @ self.x_k + self.B @ u_k + self.w_k
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
            x_km (jnp.ndarray): Noisy measurement vector with shape (n x 1) or (n, ).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                - Corrected state estimate (shape n x 1).
                - Updated error covariance matrix (shape n x n).

        Raises:
            RuntimeError: If an error occurs during state update.
        """
        try:
            if self.__Zrandom:
                self.Z = self.__MVNrandom(
                    mean=jnp.zeros(self.R.shape[0]),
                    cov=self.R,
                )
            self.measurements = self.C @ x_km.reshape((-1, 1)) + self.Z
            x_k = self.x_k + self.K @ (self.measurements - self.H @ self.x_k)
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
            u_k (jnp.ndarray): Control input vector (m x 1) or (m, ).

        Returns:
            jnp.ndarray: Updated state estimate vector (n x 1).

        Raises:
            RuntimeError: If the control input vector has an invalid shape or an error occurs during prediction.
        """
        if type(self) is KalmanFilter and u_k.shape[0] != self.B.shape[1]:
            raise RuntimeError(
                f"The Control input vector u_k must be a column vector with shape ({self.B.shape[1]}, 1) or ({self.B.shape[1]},), got {u_k.shape}!"
            )
        try:
            u_k = u_k.reshape((-1, 1))
            self.x_k = self._step_estimation(u_k)
            self.P = self._process_covariance()
            return self.x_k
        except Exception as e:
            raise RuntimeError(f"Error in the predict method: {e}") from e

    def update(self, x_km: jnp.ndarray) -> jnp.ndarray:
        """
        Update the state estimate based on the new measurement and the Kalman Gain matrix K (n x p).

        Args:
            x_km (jnp.ndarray): Measured state vector (n x 1) or (n, ).

        Returns:
            jnp.ndarray: Updated state estimate vector (n x 1).

        Raises:
            RuntimeError: If the measurement vector has an invalid shape or an error occurs during update.
        """
        if type(self) is KalmanFilter and x_km.shape[0] != self.C.shape[1]:
            raise RuntimeError(
                f"The Measured state vector x_km must be a column vector with shape ({self.C.shape[1]}, 1) or ({self.C.shape[1]},), got {x_km.shape}!"
            )
        try:
            x_km = x_km.reshape((-1, 1))
            self.K = self._kalman_function()
            self.x_k, self.P = self._current_state_and_process(x_km)
            return self.x_k
        except Exception as e:
            raise RuntimeError(f"Error in the update method: {e}") from e


"################################################################################# Extended Kalman Filter #################################################################################"


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
        __function_f (callable): Jacobian of the state transition function, f(x, u).
        __function_h (callable): Jacobian of the measurement function, h(x).
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
            self.__function_f = jaccobian_f
            self.__set_matrix_f = self.__set_none
            self.A = self.__function_f(self.x_k)

        else:
            self.__set_matrix_f = self.__matrix_f

        self.f = f  # Nonlinear state transition function: f(x, u)

        if jaccobian_h is not None:
            self.__function_h = jaccobian_h
            self.__set_matrix_h = self.__set_none
            self.H = self.__function_h(self.x_k)
        else:
            self.__set_matrix_h = self.__matrix_h

        self.h = h  # Nonlinear measurement function: h(x)
        assert self.__verify_matrices()

    def __verify_matrices(self):
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

    def __set_none(self, u_k: jnp.ndarray = None) -> None:
        """
        Placeholder function for cases where Jacobians are precomputed and do not need updating.

        Args:
            u_k (jnp.ndarray, optional): Control input vector (m x 1). Defaults to None.
        """
        pass

    def __jacobian(
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

    def __matrix_f(self, x, u):
        """
        Computes the Jacobian of the state transition function f(x, u).

        Args:
            x (jnp.ndarray): State vector (n x 1).
            u (jnp.ndarray): Control input vector (m x 1).

        Returns:
            jnp.ndarray: Jacobian matrix of f (n x n).
        """
        return self.__jacobian(self.f, x, u)  # State transition Jacobian

    def __matrix_h(self, x):
        """
        Computes the Jacobian of the measurement function h(x).

        Args:
            x (jnp.ndarray): State vector (n x 1).

        Returns:
            jnp.ndarray: Jacobian matrix of h (p x n).
        """
        return self.__jacobian(self.h, x)  # Measurement Jacobian

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
            self.__set_matrix_f(u_k)
            self.__set_matrix_h()
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
            self.A = self.__function_f(self.x_k)
            self.H = self.__function_h(self.x_k)
            x_k = self.x_k + self.K @ (x_km - self.h(self.x_k) + self.Z)
            p_k = (jnp.eye(self.K.shape[0]) - self.K @ self.H) @ self.P
            return x_k, p_k
        except Exception as e:
            raise RuntimeError(f"Error in EKF state and process update:{e}") from e
