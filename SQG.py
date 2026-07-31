import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class SQGModel:
    def __init__(self, theta_0, Nx, Ny, Tmax, Xmax, Ymax, cfl=0.1, dt_max=1e-3):
        self.Tmax, self.t = Tmax, 0.0
        self.cfl, self.dt_max = cfl, dt_max

        self.x = jnp.linspace(0, Xmax, Nx, endpoint=False)
        self.y = jnp.linspace(0, Ymax, Ny, endpoint=False)
        self.X, self.Y = jnp.meshgrid(self.x, self.y)
        delta_x, delta_y = Xmax / Nx, Ymax / Ny
        self.delta_x, self.delta_y = delta_x, delta_y

        kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, d=delta_x)
        ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, d=delta_y)
        self.Kx, self.Ky = jnp.meshgrid(kx, ky, indexing="xy")
        K_abs = jnp.sqrt(self.Kx**2 + self.Ky**2)
        self.K_abs_safe = jnp.where(K_abs == 0.0, 1.0, K_abs)

        k_max = jnp.max(jnp.abs(kx))
        self.dealias = (jnp.abs(self.Kx) < (2 / 3) * k_max) & (
            jnp.abs(self.Ky) < (2 / 3) * k_max
        )
        self.filt = jnp.where(
            self.dealias, jnp.exp(-36.0 * (K_abs / ((2 / 3) * k_max)) ** 36), 0.0
        )

        self.theta_hat = jnp.fft.fft2(theta_0) * self.dealias
        self._step_jit = jax.jit(self._rk3_core)
        self._dt_jit = jax.jit(self._compute_dt_core)

    def _psi(self, theta_hat):
        return (-theta_hat / self.K_abs_safe).at[0, 0].set(0.0)

    def _velocity(self, theta_hat):
        psi_hat = self._psi(theta_hat)
        vx = jnp.fft.ifft2(1j * self.Ky * psi_hat).real
        vy = jnp.fft.ifft2(-1j * self.Kx * psi_hat).real
        return vx, vy

    def _rhs(self, theta_hat, f_hat):
        vx, vy = self._velocity(theta_hat)
        tx = jnp.fft.ifft2(1j * self.Kx * theta_hat).real
        ty = jnp.fft.ifft2(1j * self.Ky * theta_hat).real
        return -jnp.fft.fft2(vx * tx + vy * ty) * self.dealias + f_hat

    def _rk3_core(self, theta_hat, f_hat, dt):
        r1 = self._rhs(theta_hat, f_hat)
        t1 = theta_hat + dt * r1
        r2 = self._rhs(t1, f_hat)
        t2 = 0.75 * theta_hat + 0.25 * (t1 + dt * r2)
        r3 = self._rhs(t2, f_hat)
        return self.filt * ((1 / 3) * theta_hat + (2 / 3) * (t2 + dt * r3))

    def _compute_dt_core(self, theta_hat):
        vx, vy = self._velocity(theta_hat)
        rate = jnp.max(jnp.abs(vx)) / self.delta_x + jnp.max(jnp.abs(vy)) / self.delta_y
        return jnp.minimum(self.dt_max, self.cfl / (rate + 1e-12))

    def ssp_rk3_step(self, f):
        f_hat = jnp.fft.fft2(f) * self.dealias
        dt = float(self._dt_jit(self.theta_hat))
        dt = min(dt, self.Tmax - self.t)
        self.t += dt
        self.theta_hat = self._step_jit(self.theta_hat, f_hat, dt)
        return self.theta_hat
