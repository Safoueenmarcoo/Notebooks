import jax.numpy as jnp  # type: ignore
import numpy as np  # type: ignore
from scipy.signal import convolve2d  # type: ignore
import warnings  # type: ignore
from typing import Optional
from jax.image import resize  # type: ignore
import cv2  # type: ignore


class OpticalFlow:
    """
    Estimate dense optical flow between two grayscale images.

    This class implements three optical flow estimation methods:

    1. Lucas-Kanade (local differential method).
    2. Coarse-to-Fine pyramidal Lucas-Kanade.
    3. Robust Feature Descriptor (RFD)-based optical flow.

        Notation:
            H : Image height (number of rows).
            W : Image width (number of columns).

        Attributes:
            image1 (np.ndarray):
                First grayscale image with shape (H, W).

            image2 (np.ndarray):
                Second grayscale image with shape (H, W).

            __lk_u_flow (Optional[np.ndarray]):
                Horizontal optical flow estimated using Lucas-Kanade.

            __lk_v_flow (Optional[np.ndarray]):
                Vertical optical flow estimated using Lucas-Kanade.

            __ctf_u_flow (Optional[np.ndarray]):
                Horizontal optical flow estimated using the
                Coarse-to-Fine method.

            __ctf_v_flow (Optional[np.ndarray]):
                Vertical optical flow estimated using the
                Coarse-to-Fine method.

            __rfd_u_flow (Optional[np.ndarray]):
                Horizontal optical flow estimated using the
                Robust Feature Descriptor method.

            __rfd_v_flow (Optional[np.ndarray]):
                Vertical optical flow estimated using the
                Robust Feature Descriptor method.
    """

    def __init__(self, image1: np.ndarray, image2: np.ndarray) -> None:
        """
        Initialize an Optical Flow estimator.

        Args:
            image1 (np.ndarray):
                First grayscale image with shape (H, W).

            image2 (np.ndarray):
                Second grayscale image with shape (H, W).

        Attributes:
            image1 (np.ndarray):
                First image converted to float32.

            image2 (np.ndarray):
                Second image converted to float32.

            __lk_u_flow (Optional[np.ndarray]):
                Cached horizontal Lucas-Kanade flow.

            __lk_v_flow (Optional[np.ndarray]):
                Cached vertical Lucas-Kanade flow.

            __ctf_u_flow (Optional[np.ndarray]):
                Cached horizontal Coarse-to-Fine flow.

            __ctf_v_flow (Optional[np.ndarray]):
                Cached vertical Coarse-to-Fine flow.

            __rfd_u_flow (Optional[np.ndarray]):
                Cached horizontal Robust Feature Descriptor flow.

            __rfd_v_flow (Optional[np.ndarray]):
                Cached vertical Robust Feature Descriptor flow.

        Raises:
            ValueError:
                If the input images do not have identical dimensions.
        """
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same shape")

        self.image1: np.ndarray = np.asarray(image1, dtype=np.float32)
        self.image2: np.ndarray = np.asarray(image2, dtype=np.float32)

        self.__lk_u_flow: Optional[np.ndarray] = None
        self.__lk_v_flow: Optional[np.ndarray] = None

        self.__ctf_u_flow: Optional[np.ndarray] = None
        self.__ctf_v_flow: Optional[np.ndarray] = None

        self.__rfd_u_flow: Optional[np.ndarray] = None
        self.__rfd_v_flow: Optional[np.ndarray] = None

    def _compute_gradients_sobel(
        self, image1: np.ndarray, image2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spatial and temporal gradients using Sobel operator.

        Args:
            image1: First image (H, W)
            image2: Second image (H, W)

        Returns:
            tuple: (Ix, Iy, It) - Spatial gradients in x, y directions and temporal gradient
        """
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        # Symmetric spatial gradients
        I_avg = 0.5 * (image1 + image2)
        Ix = convolve2d(I_avg, kernel_x, mode="same", boundary="symm")
        Iy = convolve2d(I_avg, kernel_y, mode="same", boundary="symm")

        It = image2 - image1
        return Ix, Iy, It

    def _downsample_image(self, image: np.ndarray, level: int) -> np.ndarray:
        """
        Downsample hierarchically: apply blur+stride level times.

        Args:
            image: Input image (H, W)
            level: Number of downsampling levels to apply

        Returns:
            Downsampled image with shape (H/(2^level), W/(2^level))
        """
        kernel = np.ones((2, 2)) / 4  # fixed small box filter
        for _ in range(level):
            image = convolve2d(image, kernel, mode="same")
            image = image[::2, ::2]
        return image

    def _upsample_flow_cv(
        self,
        u_coarse: np.ndarray,
        v_coarse: np.ndarray,
        new_shape: tuple[int, int],
        method: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Upsample flow fields and scale flow magnitudes appropriately.

        Args:
            u_coarse: Horizontal flow component (Hc, Wc)
            v_coarse: Vertical flow component (Hc, Wc)
            new_shape: Target shape (Hf, Wf), e.g. (250, 250)
            method: Interpolation method ('bilinear', 'nearest', etc.)

        Returns:
            tuple: (u_fine, v_fine) - Upsampled flow fields with shape (Hf, Wf)
        """
        Hc, Wc = u_coarse.shape
        Hf, Wf = new_shape
        sx = Wf / Wc
        sy = Hf / Hc

        u_resized = resize(u_coarse.astype(np.float32), (Hf, Wf), method)
        v_resized = resize(v_coarse.astype(np.float32), (Hf, Wf), method)

        u_resized *= sx
        v_resized *= sy

        return u_resized, v_resized

    def _warp_image(
        self,
        I2: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        interpolation: int,
        borderMode: int,
        borderValue: int,
    ) -> np.ndarray:
        """
        Backward warp of I2 using flow fields (u, v).

        Args:
            I2: Image to warp, shape (H, W) or (H, W, 3)
            u: Horizontal optical flow field, shape (H, W)
            v: Vertical optical flow field, shape (H, W)
            interpolation: OpenCV interpolation flag (e.g., cv2.INTER_LINEAR)
            borderMode: OpenCV border mode flag (e.g., cv2.BORDER_CONSTANT)
            borderValue: Value for border pixels

        Returns:
            I2_warped: Warped image, same shape as I2
        """
        H, W = u.shape

        # create base grid
        x, y = np.meshgrid(np.arange(W), np.arange(H))

        # destination coordinates in I2
        x2 = x + np.asarray(u)
        y2 = y + np.asarray(v)
        # OpenCV remap wants float32 maps
        map_x = x2.astype(np.float32)
        map_y = y2.astype(np.float32)

        I2_warped = cv2.remap(
            I2,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=borderMode,
            borderValue=borderValue,
        )

        return I2_warped

    def _gaussian_blur(self, Img, sigma=1.0) -> np.ndarray:
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(Img, (ksize, ksize), sigma)

    def _compute_descriptor(self, image: np.ndarray) -> np.ndarray:
        Ix, Iy, _ = self._compute_gradients_sobel(image, image)
        mag = np.sqrt(Ix**2 + Iy**2)
        mag = mag / (self._gaussian_blur(mag) + 1e-6)
        return mag

    def LucasKanade(
        self,
        image1: np.ndarray = None,
        image2: np.ndarray = None,
        window_size: int = 5,
        eigen_threshold: float = 1e-3,
        warn_ill_conditioned: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute optical flow using Lucas-Kanade method.

        Args:
            window_size: Size of the window for local computation (must be odd), default 5
            eigen_threshold: Minimum eigenvalue threshold for matrix invertibility, default 1e-3
            warn_ill_conditioned: Whether to emit warnings for ill-conditioned systems, default False

        Returns:
            tuple: (flow_u, flow_v) - Horizontal and vertical optical flow fields, shape (H, W)
        """
        if window_size % 2 == 0:
            raise ValueError("Window_size must b odd")

        if image1 is None:
            image1 = self.image1
        if image2 is None:
            image2 = self.image2

        Ix, Iy, It = self._compute_gradients_sobel(image1, image2)

        h, w = image1.shape
        half = window_size // 2

        flow_u = np.zeros((h, w), dtype=np.float32)
        flow_v = np.zeros((h, w), dtype=np.float32)

        for i in range(half, h - half):
            for j in range(half, w - half):
                Ix_w = Ix[i - half : i + half + 1, j - half : j + half + 1].ravel()
                Iy_w = Iy[i - half : i + half + 1, j - half : j + half + 1].ravel()
                It_w = It[i - half : i + half + 1, j - half : j + half + 1].ravel()

                A = np.stack((Ix_w, Iy_w), axis=1)
                b = -It_w

                ATA = A.T @ A
                min_eig = np.min(np.linalg.eigvals(ATA))

                if warn_ill_conditioned:
                    if min_eig < eigen_threshold:
                        warnings.warn(
                            f"Lucas–Kanade ill-conditioned at pixel ({i}, {j}): "
                            f"min eigenvalue = {min_eig:.2e}",
                            RuntimeWarning,
                        )
                if min_eig < eigen_threshold:
                    continue

                v = np.linalg.solve(ATA, A.T @ b)
                flow_u[i, j], flow_v[i, j] = v

        self.__lk_u_flow = flow_u
        self.__lk_v_flow = flow_v
        return self.__lk_u_flow, self.__lk_v_flow

    def CoarseToFine(
        self,
        image1: np.ndarray = None,
        image2: np.ndarray = None,
        max_level: int = 3,
        base_window_size: int = 10,
        remap_interpolation: int = cv2.INTER_LINEAR,
        remap_borderMode: int = cv2.BORDER_CONSTANT,
        remap_borderValue: int = 0,
        resize_method: str = "bilinear",
        inner_iterations: int = 3,
        eigen_threshold: float = 1e-3,
        warn_ill_conditioned: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute optical flow using coarse-to-fine pyramidal Lucas-Kanade approach.

        Args:
            max_level: Number of pyramid levels, default 3
            base_window_size: Base window size for Lucas-Kanade at finest level, default 10
            remap_interpolation: OpenCV interpolation method for image warping (e.g., cv2.INTER_LINEAR), default cv2.INTER_LINEAR
            remap_borderMode: OpenCV border mode for image warping (e.g., cv2.BORDER_CONSTANT), default cv2.BORDER_CONSTANT
            remap_borderValue: Border value for constant border mode, default 0
            resize_method: Interpolation method for flow upsampling ('bilinear', 'nearest', etc.), default 'bilinear'
            inner_iterations: Number of refinement iterations per pyramid level, default 3
            eigen_threshold: Minimum eigenvalue threshold for Lucas-Kanade solver, default 1e-3
            warn_ill_conditioned: Whether to emit warnings for ill-conditioned systems, default False

        Returns:
            tuple: (u_flow, v_flow) - Horizontal and vertical optical flow fields at original resolution, shape (H, W)
        """
        if image1 is None:
            image1 = self.image1
        if image2 is None:
            image2 = self.image2
        # Coarsest level
        new_image1 = self._downsample_image(image1, max_level)
        new_image2 = self._downsample_image(image2, max_level)

        u_flow = np.zeros_like(new_image1)
        v_flow = np.zeros_like(new_image1)

        for level in range(max_level, 0, -1):
            # Scale-aware window size
            window_size = max(3, base_window_size // (2 ** (level - 1)))
            if window_size % 2 == 0:
                window_size += 1

            for _ in range(inner_iterations):
                warped_image2 = self._warp_image(
                    np.asarray(new_image2),
                    u_flow,
                    v_flow,
                    remap_interpolation,
                    remap_borderMode,
                    remap_borderValue,
                )

                # Create temporary OpticalFlow instance for this level
                temp_flow = OpticalFlow(new_image1, warped_image2)
                du, dv = temp_flow.LucasKanade(
                    window_size=window_size,
                    eigen_threshold=eigen_threshold,
                    warn_ill_conditioned=warn_ill_conditioned,
                )

                u_flow += du
                v_flow += dv

            # Move to next finer level
            if level > 1:
                new_image1 = self._downsample_image(self.image1, level - 1)
                new_image2 = self._downsample_image(self.image2, level - 1)

                u_flow, v_flow = self._upsample_flow_cv(
                    u_flow,
                    v_flow,
                    new_image1.shape,
                    resize_method,
                )

                # Safety check
                assert u_flow.shape == new_image1.shape

        self.__ctf_u_flow = u_flow
        self.__ctf_v_flow = v_flow
        return self.__ctf_u_flow, self.__ctf_v_flow

    def RobustFeatureDescriptor(
        self, image1: np.ndarray = None, image2: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if image1 is None:
            image1 = self.image1
        if image2 is None:
            image2 = self.image2
        descriptor1 = self._compute_descriptor(image1)
        descriptor2 = self._compute_descriptor(image2)

        u_flow, v_flow = self.CoarseToFine(descriptor1, descriptor2)

        self.__rfd_u_flow = u_flow
        self.__rfd_v_flow = v_flow

        return self.__rfd_u_flow, self.__rfd_v_flow


class OpticalFlow2:
    def __init__(self, image1: np.ndarray, image2: np.ndarray):
        """
        Initialize Optical Flow estimator with two images.

        Args:
            image1: First image (H, W), grayscale numpy array
            image2: Second image (H, W), grayscale numpy array

        Raises:
            ValueError: If images have different shapes
        """
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same shape")

        self.image1 = np.asarray(image1, dtype=np.float32)
        self.image2 = np.asarray(image2, dtype=np.float32)

        self.__lk_u_flow = None
        self.__lk_v_flow = None
        self.__ctf_u_flow = None
        self.__ctf_v_flow = None

    def _compute_gradients_sobel(
        self, image1: np.ndarray, image2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spatial and temporal gradients using Sobel operator.

        Args:
            image1: First image (H, W)
            image2: Second image (H, W)

        Returns:
            tuple: (Ix, Iy, It) - Spatial gradients in x, y directions and temporal gradient
        """
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        # Symmetric spatial gradients
        I_avg = 0.5 * (image1 + image2)
        Ix = convolve2d(I_avg, kernel_x, mode="same", boundary="symm")
        Iy = convolve2d(I_avg, kernel_y, mode="same", boundary="symm")

        It = image2 - image1
        return Ix, Iy, It

    def _downsample_image(self, image: jnp.ndarray, level: int) -> jnp.ndarray:
        """
        Downsample hierarchically: apply blur+stride level times.

        Args:
            image: Input image (H, W)
            level: Number of downsampling levels to apply

        Returns:
            Downsampled image with shape (H/(2^level), W/(2^level))
        """
        kernel = jnp.ones((2, 2)) / 4  # fixed small box filter
        for _ in range(level):
            image = convolve2d(image, kernel, mode="same")
            image = image[::2, ::2]
        return image

    def _upsample_flow_cv(
        self,
        u_coarse: np.ndarray,
        v_coarse: np.ndarray,
        new_shape: tuple[int, int],
        method: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Upsample flow fields and scale flow magnitudes appropriately.

        Args:
            u_coarse: Horizontal flow component (Hc, Wc)
            v_coarse: Vertical flow component (Hc, Wc)
            new_shape: Target shape (Hf, Wf), e.g. (250, 250)
            method: Interpolation method ('bilinear', 'nearest', etc.)

        Returns:
            tuple: (u_fine, v_fine) - Upsampled flow fields with shape (Hf, Wf)
        """
        Hc, Wc = u_coarse.shape
        Hf, Wf = new_shape
        sx = Wf / Wc
        sy = Hf / Hc

        u_resized = resize(u_coarse.astype(jnp.float32), (Hf, Wf), method)
        v_resized = resize(v_coarse.astype(jnp.float32), (Hf, Wf), method)

        u_resized *= sx
        v_resized *= sy

        return u_resized, v_resized

    def _warp_image(
        self,
        I2: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        interpolation: int,
        borderMode: int,
        borderValue: int,
    ) -> np.ndarray:
        """
        Backward warp of I2 using flow fields (u, v).

        Args:
            I2: Image to warp, shape (H, W) or (H, W, 3)
            u: Horizontal optical flow field, shape (H, W)
            v: Vertical optical flow field, shape (H, W)
            interpolation: OpenCV interpolation flag (e.g., cv2.INTER_LINEAR)
            borderMode: OpenCV border mode flag (e.g., cv2.BORDER_CONSTANT)
            borderValue: Value for border pixels

        Returns:
            I2_warped: Warped image, same shape as I2
        """
        H, W = u.shape

        # create base grid
        x, y = np.meshgrid(np.arange(W), np.arange(H))

        # destination coordinates in I2
        x2 = x + np.asarray(u)
        y2 = y + np.asarray(v)
        # OpenCV remap wants float32 maps
        map_x = x2.astype(np.float32)
        map_y = y2.astype(np.float32)

        I2_warped = cv2.remap(
            I2,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=borderMode,
            borderValue=borderValue,
        )

        return I2_warped

    def LucasKanade(
        self,
        window_size: int = 5,
        eigen_threshold: float = 1e-3,
        warn_ill_conditioned: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute optical flow using Lucas-Kanade method.

        Args:
            window_size: Size of the window for local computation (must be odd), default 5
            eigen_threshold: Minimum eigenvalue threshold for matrix invertibility, default 1e-3
            warn_ill_conditioned: Whether to emit warnings for ill-conditioned systems, default False

        Returns:
            tuple: (flow_u, flow_v) - Horizontal and vertical optical flow fields, shape (H, W)
        """

        Ix, Iy, It = self._compute_gradients_sobel(self.image1, self.image2)

        h, w = self.image1.shape
        half = window_size // 2

        flow_u = np.zeros((h, w), dtype=np.float32)
        flow_v = np.zeros((h, w), dtype=np.float32)

        for i in range(half, h - half):
            for j in range(half, w - half):
                Ix_w = Ix[i - half : i + half + 1, j - half : j + half + 1].ravel()
                Iy_w = Iy[i - half : i + half + 1, j - half : j + half + 1].ravel()
                It_w = It[i - half : i + half + 1, j - half : j + half + 1].ravel()

                A = np.stack((Ix_w, Iy_w), axis=1)
                b = -It_w

                ATA = A.T @ A
                min_eig = np.min(np.linalg.eigvals(ATA))

                if warn_ill_conditioned:
                    if min_eig < eigen_threshold:
                        warnings.warn(
                            f"Lucas–Kanade ill-conditioned at pixel ({i}, {j}): "
                            f"min eigenvalue = {min_eig:.2e}",
                            RuntimeWarning,
                        )
                if min_eig < eigen_threshold:
                    continue

                v = np.linalg.solve(ATA, A.T @ b)
                flow_u[i, j], flow_v[i, j] = v

        self.__lk_u_flow = flow_u
        self.__lk_v_flow = flow_v
        return self.__lk_u_flow, self.__lk_v_flow

    def CoarseToFine(
        self,
        max_level: int = 3,
        base_window_size: int = 10,
        remap_interpolation: int = cv2.INTER_LINEAR,
        remap_borderMode: int = cv2.BORDER_CONSTANT,
        remap_borderValue: int = 0,
        resize_method: str = "bilinear",
        inner_iterations: int = 3,
        eigen_threshold: float = 1e-3,
        warn_ill_conditioned: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute optical flow using coarse-to-fine pyramidal Lucas-Kanade approach.

        Args:
            max_level: Number of pyramid levels, default 3
            base_window_size: Base window size for Lucas-Kanade at finest level, default 10
            remap_interpolation: OpenCV interpolation method for image warping (e.g., cv2.INTER_LINEAR), default cv2.INTER_LINEAR
            remap_borderMode: OpenCV border mode for image warping (e.g., cv2.BORDER_CONSTANT), default cv2.BORDER_CONSTANT
            remap_borderValue: Border value for constant border mode, default 0
            resize_method: Interpolation method for flow upsampling ('bilinear', 'nearest', etc.), default 'bilinear'
            inner_iterations: Number of refinement iterations per pyramid level, default 3
            eigen_threshold: Minimum eigenvalue threshold for Lucas-Kanade solver, default 1e-3
            warn_ill_conditioned: Whether to emit warnings for ill-conditioned systems, default False

        Returns:
            tuple: (u_flow, v_flow) - Horizontal and vertical optical flow fields at original resolution, shape (H, W)
        """
        # Coarsest level
        new_image1 = self._downsample_image(self.image1, max_level)
        new_image2 = self._downsample_image(self.image2, max_level)

        u_flow = jnp.zeros_like(new_image1)
        v_flow = jnp.zeros_like(new_image1)

        for level in range(max_level, 0, -1):
            # Scale-aware window size
            window_size = max(3, base_window_size // (2 ** (level - 1)))

            for _ in range(inner_iterations):
                warped_image2 = self._warp_image(
                    np.asarray(new_image2),
                    u_flow,
                    v_flow,
                    remap_interpolation,
                    remap_borderMode,
                    remap_borderValue,
                )

                # Create temporary OpticalFlow instance for this level
                temp_flow = OpticalFlow2(new_image1, warped_image2)
                du, dv = temp_flow.LucasKanade(
                    window_size=window_size,
                    eigen_threshold=eigen_threshold,
                    warn_ill_conditioned=warn_ill_conditioned,
                )

                u_flow += du
                v_flow += dv

            # Move to next finer level
            if level > 1:
                new_image1 = self._downsample_image(self.image1, level - 1)
                new_image2 = self._downsample_image(self.image2, level - 1)

                u_flow, v_flow = self._upsample_flow_cv(
                    u_flow,
                    v_flow,
                    new_image1.shape,
                    resize_method,
                )

                # Safety check
                assert u_flow.shape == new_image1.shape

        self.__ctf_u_flow = u_flow
        self.__ctf_v_flow = v_flow
        return self.__ctf_u_flow, self.__ctf_v_flow
