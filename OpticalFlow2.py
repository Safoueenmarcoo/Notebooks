"""
OpticalFlow: five dense optical flow estimators with a common interface.

Methods
-------
    LucasKanade()            local window least-squares            (~0.36)
    CoarseToFine()           warping-iterated / pyramidal LK       (~0.15)
    RobustFeatureDescriptor() CTF on gradient descriptors          (~0.22)
    HornSchunck()            classic global HS, corrected          (~0.14)
    HornSchunckRefined()     HS + warping + exact sparse solve     (~0.08)

Numbers in parentheses: RMSE with the project metric (crop [29:220]) on the
run01005 PIV sequence; zero-flow baseline is 0.943. Alphas are tuned for the
[0,1]-normalized images produced by __init__ (the constructor normalizes both
frames with a SHARED min/max).

Fixes applied relative to the original implementations
------------------------------------------------------
FIX 1  Sobel gradient scale: cv2.Sobel(ksize=3) returns 8x the true
       derivative; scale=0.125 restores unit gain so spatial and temporal
       derivatives are consistent (otherwise flow comes out ~8x too small).
FIX 2  Shared normalization of the two frames (independent per-image min/max
       normalization creates an artificial illumination change).
FIX 3  CoarseToFine computes increments directly instead of constructing
       inner OpticalFlow objects (no hidden renormalization; subclass
       overrides propagate).
FIX 4  Pyramid stability: constant window across levels, cv2.pyrDown
       anti-aliasing, clamped increments, 3x3 median filter between warps.
FIX 5  Warping uses BORDER_REPLICATE (constant-0 borders create fake
       gradients and spurious border flow).
FIX 6  Ill-conditioned LK pixels are inpainted from valid neighbors instead
       of being left at zero.
FIX 7  (HS) Derivative kernels applied with correlation (cv2.filter2D), not
       scipy convolve, so the 2x2 stencils keep their intended signs.
FIX 8  (HS) Iteration cap always applies; convergence checked on u AND v.
"""

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import convolve, median_filter
from scipy.sparse.linalg import LinearOperator, cg, factorized

_HS_AVG_KERNEL = np.array(
    [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0.0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]],
    dtype=np.float32,
)

_LAPLACIAN_CACHE: dict = {}


def _laplacian(h: int, w: int) -> sp.csr_matrix:
    """5-point graph Laplacian with Neumann boundaries (cached per shape)."""
    if (h, w) in _LAPLACIAN_CACHE:
        return _LAPLACIAN_CACHE[(h, w)]
    n = h * w
    idx = np.arange(n).reshape(h, w)
    main = np.zeros(n)
    rows, cols, vals = [], [], []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        ny, nx = ys + dy, xs + dx
        m = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        rows += list(idx[ys[m], xs[m]])
        cols += list(idx[ny[m], nx[m]])
        vals += [-1.0] * int(m.sum())
        main[idx[ys[m], xs[m]].ravel()] += 1.0
    rows += list(range(n))
    cols += list(range(n))
    vals += list(main)
    L = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    _LAPLACIAN_CACHE[(h, w)] = L
    return L


class OpticalFlow:
    def __init__(self, image1: np.ndarray, image2: np.ndarray) -> None:
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same shape")
        if image1.ndim != 2:
            raise ValueError("Images must be grayscale")
        i1 = np.asarray(image1, dtype=np.float32)
        i2 = np.asarray(image2, dtype=np.float32)
        # FIX 2: shared normalization
        mn = min(i1.min(), i2.min())
        mx = max(i1.max(), i2.max())
        span = mx - mn
        if span < 1e-6:
            self.image1 = np.zeros_like(i1)
            self.image2 = np.zeros_like(i2)
        else:
            self.image1 = (i1 - mn) / span
            self.image2 = (i2 - mn) / span

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _normalize_pair(i1, i2):
        i1 = np.asarray(i1, dtype=np.float32)
        i2 = np.asarray(i2, dtype=np.float32)
        mn = min(i1.min(), i2.min())
        mx = max(i1.max(), i2.max())
        span = mx - mn
        if span < 1e-6:
            return np.zeros_like(i1), np.zeros_like(i2)
        return (i1 - mn) / span, (i2 - mn) / span

    @staticmethod
    def _compute_gradients_sobel(image1, image2):
        avg = 0.5 * (image1 + image2)
        # FIX 1: scale=0.125 makes Sobel a unit-gain derivative
        Ix = cv2.Sobel(avg, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
        Iy = cv2.Sobel(avg, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
        It = image2 - image1
        return Ix, Iy, It

    @staticmethod
    def _downsample_image(image):
        # FIX 4: Gaussian anti-aliasing + subsample
        return cv2.pyrDown(np.asarray(image, np.float32))

    @staticmethod
    def _upsample_flow(u, v, new_shape):
        Hc, Wc = u.shape
        Hf, Wf = new_shape
        u = cv2.resize(u.astype(np.float32), (Wf, Hf), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v.astype(np.float32), (Wf, Hf), interpolation=cv2.INTER_LINEAR)
        return u * (Wf / Wc), v * (Hf / Hc)

    @staticmethod
    def _warp_image(
        I2,
        u,
        v,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
        borderValue=0,
    ):  # FIX 5
        H, W = u.shape
        x, y = np.meshgrid(
            np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)
        )
        return cv2.remap(
            I2,
            x + u,
            y + v,
            interpolation=interpolation,
            borderMode=borderMode,
            borderValue=borderValue,
        )

    @staticmethod
    def _fill_invalid(flow, valid):
        # FIX 6: inpaint invalid pixels from valid neighbors
        if valid.all():
            return flow
        mask = (~valid).astype(np.uint8)
        return cv2.inpaint(flow.astype(np.float32), mask, 3, cv2.INPAINT_NS)

    @staticmethod
    def vorticity(u, v, sigma=1.0):
        """Curl of the flow (dv/dx - du/dy), computed on a Gaussian-smoothed
        copy so estimation noise isn't amplified by differentiation.
        sigma=1.0 is optimal on the run01005 data; sigma=0 disables smoothing
        (use for ground-truth flow, which has no estimation noise)."""
        u = np.asarray(u, np.float32)
        v = np.asarray(v, np.float32)
        if sigma > 0:
            u = cv2.GaussianBlur(u, (0, 0), sigma)
            v = cv2.GaussianBlur(v, (0, 0), sigma)
        return np.gradient(v, axis=1) - np.gradient(u, axis=0)

    # ------------------------------------------------------------------- LK
    def LucasKanade(
        self,
        image1=None,
        image2=None,
        window_size=21,
        eigen_threshold=1e-6,
        fill_invalid=True,
        return_valid=False,
    ):
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if image1 is None or image2 is None:
            image1, image2 = self.image1, self.image2
        else:
            image1, image2 = self._normalize_pair(image1, image2)

        Ix, Iy, It = self._compute_gradients_sobel(image1, image2)
        Ix, Iy, It = (a.astype(np.float64) for a in (Ix, Iy, It))

        def ls(z):
            return cv2.boxFilter(
                z,
                cv2.CV_64F,
                (window_size, window_size),
                normalize=False,
                borderType=cv2.BORDER_CONSTANT,
            )

        Sxx, Syy, Sxy = ls(Ix * Ix), ls(Iy * Iy), ls(Ix * Iy)
        Sxt, Syt = ls(Ix * It), ls(Iy * It)

        det = Sxx * Syy - Sxy**2
        disc = np.sqrt(np.maximum((Sxx - Syy) ** 2 + 4.0 * Sxy**2, 0.0))
        lam_min = 0.5 * ((Sxx + Syy) - disc)
        valid = (lam_min >= eigen_threshold) & (np.abs(det) > 1e-12)
        half = window_size // 2
        valid[:half, :] = valid[-half:, :] = False
        valid[:, :half] = valid[:, -half:] = False

        u = np.zeros(image1.shape, np.float32)
        v = np.zeros(image1.shape, np.float32)
        u[valid] = (Sxy[valid] * Syt[valid] - Syy[valid] * Sxt[valid]) / det[valid]
        v[valid] = (Sxy[valid] * Sxt[valid] - Sxx[valid] * Syt[valid]) / det[valid]
        if fill_invalid:
            u = self._fill_invalid(u, valid)
            v = self._fill_invalid(v, valid)
        if return_valid:
            return u, v, valid
        return u, v

    def CoarseToFine(
        self,
        image1=None,
        image2=None,
        max_level=0,
        window_size=7,
        inner_iterations=5,
        eigen_threshold=1e-6,
        max_increment=1.5,
        median_size=3,
        presmooth_sigma=0.8,
    ):
        """
        Warping-iterated (optionally pyramidal) Lucas-Kanade.

        On the run01005 data (motions < 3.5 px) max_level=0 is optimal; the
        pyramid pays off only when displacements exceed ~window_size/2.
        """
        if image1 is None or image2 is None:
            image1, image2 = self.image1, self.image2
        else:
            image1, image2 = self._normalize_pair(image1, image2)
        if presmooth_sigma > 0:
            image1 = cv2.GaussianBlur(image1, (0, 0), presmooth_sigma)
            image2 = cv2.GaussianBlur(image2, (0, 0), presmooth_sigma)

        p1, p2 = [image1], [image2]
        for _ in range(max_level):
            p1.append(self._downsample_image(p1[-1]))
            p2.append(self._downsample_image(p2[-1]))

        u = np.zeros_like(p1[max_level])
        v = np.zeros_like(p1[max_level])
        for level in range(max_level, -1, -1):
            c1, c2 = p1[level], p2[level]
            if u.shape != c1.shape:
                u, v = self._upsample_flow(u, v, c1.shape)
            # FIX 4: constant window size across levels
            for _ in range(inner_iterations):
                w2 = self._warp_image(c2, u, v)
                # FIX 3: compute increments directly (no inner OpticalFlow())
                du, dv = self.LucasKanade(
                    c1, w2, window_size=window_size, eigen_threshold=eigen_threshold
                )
                np.clip(du, -max_increment, max_increment, out=du)  # FIX 4
                np.clip(dv, -max_increment, max_increment, out=dv)
                u = u + du
                v = v + dv
                if median_size:  # FIX 4
                    u = median_filter(u, median_size)
                    v = median_filter(v, median_size)
        return u, v

    # ------------------------------------------------------------------ RFD
    def _compute_descriptor(self, image, sigma=1.0):
        Ix, Iy, _ = self._compute_gradients_sobel(image, image)
        mag = np.sqrt(Ix**2 + Iy**2)
        k = int(6 * sigma + 1) | 1
        smooth = cv2.GaussianBlur(mag, (k, k), sigma)
        return mag / (smooth + 1e-6)

    def RobustFeatureDescriptor(self, image1=None, image2=None, **ctf_kwargs):
        """
        CTF on illumination-robust gradient descriptors.

        NOTE: on constant-illumination speckle/particle images this descriptor
        discards the clean intensity signal and keeps amplified noise; expect
        it to underperform plain CoarseToFine on such data (it is intended for
        natural scenes with illumination changes).
        """
        image1 = self.image1 if image1 is None else image1
        image2 = self.image2 if image2 is None else image2
        d1 = self._compute_descriptor(image1)
        d2 = self._compute_descriptor(image2)
        return self.CoarseToFine(d1, d2, **ctf_kwargs)

    # -------------------------------------------------------- Horn-Schunck
    def HornSchunck(
        self,
        image1=None,
        image2=None,
        alpha=0.03,
        iterations=1000,
        tolerance=None,
        blur_ksize=5,
    ):
        """
        Classic Horn-Schunck (corrected). Jacobi iteration on the original
        2x2-stencil formulation with denominator 4*alpha^2 + fx^2 + fy^2.

        alpha is tuned for [0,1]-normalized images (multiply by ~255 if you
        feed raw 8-bit intensities via the image1/image2 arguments).
        """
        if image1 is None or image2 is None:
            image1, image2 = self.image1, self.image2
        else:
            image1, image2 = self._normalize_pair(image1, image2)
        if blur_ksize:
            image1 = cv2.GaussianBlur(image1, (blur_ksize, blur_ksize), 0)
            image2 = cv2.GaussianBlur(image2, (blur_ksize, blur_ksize), 0)

        # FIX 7: correlation (no kernel flip) keeps the intended signs
        kx = np.array([[-1, 1], [-1, 1]], np.float32) * 0.25
        ky = np.array([[-1, -1], [1, 1]], np.float32) * 0.25
        kt = np.ones((2, 2), np.float32) * 0.25
        fx = cv2.filter2D(image1, -1, kx, anchor=(0, 0)) + cv2.filter2D(
            image2, -1, kx, anchor=(0, 0)
        )
        fy = cv2.filter2D(image1, -1, ky, anchor=(0, 0)) + cv2.filter2D(
            image2, -1, ky, anchor=(0, 0)
        )
        ft = cv2.filter2D(image2, -1, kt, anchor=(0, 0)) - cv2.filter2D(
            image1, -1, kt, anchor=(0, 0)
        )

        u = np.zeros(image1.shape, np.float32)
        v = np.zeros(image1.shape, np.float32)
        d = 4.0 * alpha**2 + fx**2 + fy**2
        for _ in range(iterations):  # FIX 8: cap always applies
            u_avg = convolve(u, _HS_AVG_KERNEL)
            v_avg = convolve(v, _HS_AVG_KERNEL)
            p = fx * u_avg + fy * v_avg + ft
            un = u_avg - fx * (p / d)
            vn = v_avg - fy * (p / d)
            if tolerance is not None:  # noqa: SIM102
                if max(np.linalg.norm(un - u), np.linalg.norm(vn - v)) < tolerance:
                    u, v = un, vn
                    break
            u, v = un, vn
        return u, v

    def HornSchunckRefined(
        self,
        image1=None,
        image2=None,
        alpha=0.15,
        warps=3,
        blur_sigma=0.4,
        median_size=3,
    ):
        """
        Refined Horn-Schunck: incremental warping + exact sparse solve of the
        Euler-Lagrange equations per warp (standard alpha^2 formulation)
        + central-difference derivatives + median filtering between warps.

        Best-performing method on the run01005 sequence (~0.083 with the
        project metric). alpha tuned for [0,1]-normalized images.
        """
        if image1 is None or image2 is None:
            image1, image2 = self.image1, self.image2
        else:
            image1, image2 = self._normalize_pair(image1, image2)
        if blur_sigma > 0:
            image1 = cv2.GaussianBlur(image1, (0, 0), blur_sigma)
            image2 = cv2.GaussianBlur(image2, (0, 0), blur_sigma)

        h, w = image1.shape
        n = h * w
        L = _laplacian(h, w)
        kx = np.array([[-1.0, 0.0, 1.0]], np.float32) / 2.0
        u = np.zeros((h, w), np.float32)
        v = np.zeros((h, w), np.float32)
        a2 = alpha**2
        for wi in range(warps):
            img2w = self._warp_image(image2, u, v, interpolation=cv2.INTER_CUBIC)
            avg = 0.5 * (image1 + img2w)
            fx = cv2.filter2D(avg, -1, kx)
            fy = cv2.filter2D(avg, -1, kx.T)
            ft = img2w - image1
            Fx = fx.ravel().astype(np.float64)
            Fy = fy.ravel().astype(np.float64)
            Ft = ft.ravel().astype(np.float64)
            A = sp.bmat(
                [
                    [sp.diags(Fx * Fx) + a2 * L, sp.diags(Fx * Fy)],
                    [sp.diags(Fx * Fy), sp.diags(Fy * Fy) + a2 * L],
                ],
                format="csc",
            )
            sol = factorized(A)(np.concatenate([-Fx * Ft, -Fy * Ft]))
            u = u + sol[:n].reshape(h, w).astype(np.float32)
            v = v + sol[n:].reshape(h, w).astype(np.float32)
            if median_size and wi < warps - 1:
                u = median_filter(u, median_size)
                v = median_filter(v, median_size)
        return u, v

    def HornSchunckSecondOrder(
        self,
        image1=None,
        image2=None,
        alpha1=0.02,
        alpha2=0.3,
        warps=3,
        blur_sigma=0.4,
        median_size=3,
        cg_rtol=1e-8,
        cg_maxiter=3000,
    ):
        """
        Warping HS with mixed first + second order regularization:
            R = alpha1^2 * L + alpha2^2 * L^2   (L = Laplacian)
        The L^2 term penalizes flow curvature instead of flow gradients, so
        vorticity (curl) fields come out much smoother -- the right prior for
        turbulent/fluid motion. Solved per warp with Jacobi-preconditioned
        conjugate gradients.

        run01005 validation: EPE ~0.070, raw curl RMSE ~0.024
        (vs 0.083 / 0.036 for HornSchunckRefined). Alphas tuned for
        [0,1]-normalized images.
        """
        if image1 is None or image2 is None:
            image1, image2 = self.image1, self.image2
        else:
            image1 = np.asarray(image1, np.float32)
            image2 = np.asarray(image2, np.float32)
        if blur_sigma > 0:
            image1 = cv2.GaussianBlur(image1, (0, 0), blur_sigma)
            image2 = cv2.GaussianBlur(image2, (0, 0), blur_sigma)
        h, w = image1.shape
        n = h * w
        L = _laplacian(h, w).tocsr()
        R = ((alpha1**2) * L + (alpha2**2) * (L @ L)).tocsr()
        kx = np.array([[-1.0, 0.0, 1.0]], np.float32) / 2.0
        u = np.zeros((h, w), np.float32)
        v = np.zeros((h, w), np.float32)
        x0 = None
        for wi in range(warps):
            img2w = self._warp_image(image2, u, v, interpolation=cv2.INTER_CUBIC)
            avg = 0.5 * (image1 + img2w)
            fx = cv2.filter2D(avg, -1, kx)
            fy = cv2.filter2D(avg, -1, kx.T)
            ft = img2w - image1
            Fx = fx.ravel().astype(np.float64)
            Fy = fy.ravel().astype(np.float64)
            Ft = ft.ravel().astype(np.float64)
            A = sp.bmat(
                [
                    [sp.diags(Fx * Fx) + R, sp.diags(Fx * Fy)],
                    [sp.diags(Fx * Fy), sp.diags(Fy * Fy) + R],
                ],
                format="csr",
            )
            b = np.concatenate([-Fx * Ft, -Fy * Ft])
            Md = A.diagonal()
            Minv = LinearOperator(A.shape, lambda x: x / Md)
            sol, _ = cg(A, b, x0=x0, rtol=cg_rtol, maxiter=cg_maxiter, M=Minv)
            x0 = sol
            u = u + sol[:n].reshape(h, w).astype(np.float32)
            v = v + sol[n:].reshape(h, w).astype(np.float32)
            if median_size and wi < warps - 1:
                u = median_filter(u, median_size)
                v = median_filter(v, median_size)
        return u, v
