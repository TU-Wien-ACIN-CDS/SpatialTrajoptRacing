from .config import Config
import cv2
import numpy as np
import threading
from functools import partial
import logging
from cma import CMAOptions, CMAEvolutionStrategy
from tqdm import tqdm

class RacelineOptim:
    def __init__(self, config: Config = None):

        if config is None:
            self.cfg = Config()
        else:
            self.cfg = config

        self.image = None
        self.metadata = None

        self.u_eval = np.linspace(0, 1, self.cfg.optim.u_eval).reshape(-1, 1)

        self.NURBS_center = None
        self.dist_field = None
        self.friction_map = None
        self.uncertainty_map = None

        self.max_curvature = (
            1 / self.cfg.vehicle.wheelbase * np.tan(self.cfg.vehicle.delta_max)
        )

        self.p_init = None
        self.W_init = None
        self.U_init = None
        self.NURBS_best = None

        self.opti_initializied = False
        self._stop_event: threading.Event | None = None

    def generate_distance_field(self):

        if self.image is None:
            raise ValueError("Image not loaded. Call load_image() first.")

        image = self.image.copy()
        image = image[::-1, :]

        image[self.image <= 210.0] = 0
        image[self.image > 210.0] = 1
        dist_field1 = cv2.distanceTransform(image, distanceType=cv2.DIST_L2, maskSize=5)
        dist_field2 = cv2.distanceTransform(
            1 - image, distanceType=cv2.DIST_L2, maskSize=5
        )
        dist_field = dist_field1 - dist_field2

        dist_field = cv2.resize(
            dist_field,
            None,
            fx=self.cfg.dist_scaling,
            fy=self.cfg.dist_scaling,
            interpolation=cv2.INTER_LINEAR,
        )
        self.dist_field = dist_field * self.metadata["resolution"]
        self.dist_field = cv2.flip(self.dist_field, 0)

    def generate_friction_map(self, friction_map: np.ndarray = None):

        if self.image is None:
            raise ValueError("Image not loaded. Call load_image() first.")

        if friction_map is not None:
            self.friction_map = friction_map
        else:
            friction_map = self.image.copy()
            friction_map = friction_map[::-1, :]

            friction_map[self.image <= 210.0] = 1.0
            friction_map[self.image > 210.0] = 1.0

            self.friction_map = cv2.resize(
                friction_map,
                None,
                fx=self.cfg.friction_scaling,
                fy=self.cfg.friction_scaling,
                interpolation=cv2.INTER_LINEAR,
            )
            self.friction_map = cv2.flip(self.friction_map, 0)
            self.uncertainty_map = self.friction_map.copy()

    def set_image(self, image: np.ndarray, metadata: dict = None):
        self.image = cv2.flip(image, 0)
        self.metadata = metadata

    def set_centerline(self, spline):
        self.NURBS_center = spline

    def optimize(self, callback=None):

        if self.NURBS_center is None:
            raise ValueError(
                "Centerline not generated. Call generate_centerline() first."
            )
        if not self.opti_initializied:
            self.p_init = self.NURBS_center.control_points[
                : -self.cfg.nurbs.n_deriv, :
            ].flatten()
            self.W_init = self.NURBS_center.weights.flatten()[
                self.cfg.nurbs.n_deriv : -self.cfg.nurbs.n_deriv
            ]
            self.U_init = np.array(self.NURBS_center.knot_vectors).flatten()[
                self.cfg.nurbs.p + 1 : -self.cfg.nurbs.p - 1
            ]

            mean = np.hstack(
                (
                    self.p_init,
                    np.zeros_like(self.W_init),
                    np.zeros_like(self.U_init),
                )
            )
            sigma = (
                [self.cfg.optim.sigma_P] * len(self.p_init)
                + [self.cfg.optim.sigma_W] * len(self.W_init)
                + [self.cfg.optim.sigma_U] * len(self.U_init)
            )

            spline_partial = self.NURBS_center.copy()
            self.eval_candidate_partial = partial(
                self.eval_candidate, spline=spline_partial
            )

            print(f"Starting optimization...")
            print(
                f"Parameters:\n"
                f"Constraints:\n"
                f"maxIter = {self.cfg.optim.maxiter},\n"
                f"a_lat = {self.cfg.vehicle.a_max_lat},\n"
                f"a_long = {self.cfg.vehicle.a_max_long},\n"
                f"v_max = {self.cfg.vehicle.v_max},\n"
                f"Friction Map = {'True' if self.friction_map is not None else 'False'}"
            )

            options = {
                "maxiter": 10000,
                "CMA_elitist": "initial",
                "CMA_stds": sigma,
                "tolfun": self.cfg.optim.tolfun,
                "tolfunhist": self.cfg.optim.tolfunhist,
                "popsize": self.cfg.optim.popsize,
                "CMA_diagonal": False,
                "verbose": self.cfg.optim.verbose,
            }

            CMAOptions().check_attributes(options)
            opts = CMAOptions(options.copy()).complement()

            es = CMAEvolutionStrategy(mean, 1, opts)
            x = es.gp.pheno(
                es.mean,
                copy=True,
                into_bounds=es.boundary_handler.repair,
                archive=es.sent_solutions,
            )
            es.f0 = self.eval_candidate_partial(x)

            self._stop_event = threading.Event()

            it = 0
            pbar = tqdm(total=self.cfg.optim.maxiter, desc="Optimizing Raceline")
            while (
                (True if self.cfg.optim.maxiter is None else it < self.cfg.optim.maxiter)
                and not self._stop_event.is_set()
            ):
                it += 1
                X, fit = es.ask_and_eval(
                    self.eval_candidate_partial, aggregation=np.median
                )
                es.tell(X, fit)
                es.manage_plateaus()
                best_x = es.best.x

                new_fitness = self.eval_candidate_partial(best_x, debug=self.cfg.debug)
                es.best.f = new_fitness

                pbar.update(1)
                if it % 50 == 0:
                    callback(es)
                    pbar.set_postfix({"Best Fitness": f"{new_fitness:.4f}"})

            pbar.close()
            if callback is not None:
                callback(es)

    def stop(self):
        if self._stop_event is not None:
            self._stop_event.set()

    def sample_NURBS(self, x, spline):
        x_P = np.array(x[: len(self.p_init)]).reshape(
            (-1, self.cfg.nurbs.n_DOF)
        )
        x_W = self.W_init + np.array(
            x[
                len(self.p_init) : len(self.p_init)
                + len(self.W_init)
            ]
        )
        x_W = x_W[:, None]
        x_U = self.U_init + np.array(x[len(self.p_init) + len(self.W_init) :])
        x_U = x_U[None, :]
        x_U = self.ensure_unique(np.sort(x_U))

        spline = self.close_NURBS(x_P, x_W, x_U, spline)

        return spline

    def eval_candidate(self, x, spline, debug=False):

        spline = self.sample_NURBS(x, spline)
        return self.loss(spline, debug=debug)

    def loss(self, spline, debug=False):

        qu = spline.evaluate(self.u_eval)
        dqu = spline.derivative(self.u_eval, 1)
        ddqu = spline.derivative(self.u_eval, 2)

        dist_scaling = self.metadata["resolution"] / self.cfg.dist_scaling
        x = ((qu - self.metadata["origin"][:2]) / dist_scaling).astype(int)
        x0 = np.clip(x[:, 0], 0, self.dist_field.shape[1] - 1)
        x1 = np.clip(x[:, 1], 0, self.dist_field.shape[0] - 1)

        distance = self.dist_field[x1, x0]
        mask_distance = distance < self.cfg.vehicle.car_width
        max_distance_all = np.sum(np.abs(distance[mask_distance]))

        def get_signed_curvature(dqu, ddqu):
            curvature = (dqu[:, 0] * ddqu[:, 1] - dqu[:, 1] * ddqu[:, 0]) / (
                dqu[:, 0] ** 2 + dqu[:, 1] ** 2
            ) ** (3 / 2)
            return curvature

        curvature = get_signed_curvature(dqu, ddqu)
        mask_curvature = np.abs(curvature) > self.max_curvature
        curvature_loss = np.sum(np.abs(curvature[mask_curvature]))

        T = self.get_T_constraint(qu, dqu, ddqu)
        cost = (
            T
            + 1e6 * max_distance_all
            + 1e3 * curvature_loss
        )
        if debug:
            print(f"Total: {cost:.4f}, Time: T={T:.4f}, Distance Cost={max_distance_all:.4f}, Curvature Cost={curvature_loss:.4f}")

        return cost

    def get_T_constraint(self, qu, dqu, ddqu):

        if self.friction_map is not None:
            friction_scaling = self.metadata["resolution"] / self.cfg.friction_scaling
            x = ((qu - self.metadata["origin"][:2]) / friction_scaling).astype(int)
            x0 = np.clip(x[:, 0], 0, self.friction_map.shape[1] - 1)
            x1 = np.clip(x[:, 1], 0, self.friction_map.shape[0] - 1)

            a_max_long = self.friction_map[x1, x0] * self.cfg.vehicle.a_max_long
            a_max_lat = self.friction_map[x1, x0] * self.cfg.vehicle.a_max_lat
        else:
            a_max_long = self.cfg.vehicle.a_max_long
            a_max_lat = self.cfg.vehicle.a_max_lat

        v, a_long, a_lat, a = self.get_vel_acc(qu, dqu, ddqu)
        a = np.sqrt(a_long**2 + a_lat**2)

        T_v = np.max(np.max(np.abs(v)) / self.cfg.vehicle.v_max)
        T_long = np.sqrt(np.max((np.max(np.abs(a_long) / a_max_long))))
        T_lat = np.sqrt(np.max((np.max(np.abs(a_lat) / a_max_lat))))
        T = np.max(np.array([T_lat, T_long, T_v]))
        return T

    def close_NURBS(self, P, W, U, spline):

        P_spline = np.ones_like(self.NURBS_center.control_points)
        P_spline[: -self.cfg.nurbs.n_deriv, :] = P

        W_spline = np.ones_like(self.NURBS_center.weights)
        W_spline[self.cfg.nurbs.n_deriv : -self.cfg.nurbs.n_deriv, :] = W

        n = P_spline.shape[1]
        p = self.cfg.nurbs.p

        U = np.sort(U)
        U = np.hstack(
            [np.zeros(p + 1)[None, :], self.ensure_unique(U), np.ones(p + 1)[None, :]]
        )

        dC_0 = p / U[:, p + 1] * (P[1, :] - P[0, :])
        ddC_0 = (
            p
            * (p - 1)
            / U[:, p + 1]
            * (
                P[0, :] / U[:, p + 1]
                - (U[:, p + 1] + U[:, p + 2]) * P[1, :] / (U[:, p + 1] * U[:, p + 2])
                + P[2, :] / U[:, p + 2]
            )
        )

        P_last = P[0, :]
        P_prev = P_last - (1 - U[:, -p - 2]) / p * dC_0

        P_prev_prev = (
            ddC_0 * (1 - U[:, -p - 2]) / (p * (p - 1))
            - P_last / (1 - U[:, -p - 2])
            + (2 - U[:, -p - 2] - U[:, -p - 3])
            * P_prev
            / ((1 - U[:, -p - 2]) * (1 - U[:, -p - 3]))
        ) * (1 - U[:, -p - 3])

        P_spline[-1, :] = P_last
        P_spline[-2, :] = P_prev
        P_spline[-3, :] = P_prev_prev

        spline.knot_vectors = U
        spline.control_points = P_spline
        spline.weights = W_spline

        return spline

    def get_vel_acc(self, qu, dqu, ddqu):
        v_x = dqu[:, 0]
        v_y = dqu[:, 1]
        v = np.sqrt(v_x**2 + v_y**2)

        a_x = ddqu[:, 0]
        a_y = ddqu[:, 1]
        a_long = (v_x * a_x + v_y * a_y) / v
        a_lat = (v_x * a_y - v_y * a_x) / v

        a = np.sqrt(a_long**2 + a_lat**2)

        return v, a_long, a_lat, a

    def ensure_unique(self, U):
        U = np.array(U).reshape(-1, 1)
        n = U.shape[0]

        # Get sorted indices and reverse indices
        indices = np.argsort(U[:, 0])
        reverse_indices = np.zeros(n, dtype=int)
        reverse_indices[indices] = np.arange(n)

        # Sort U
        sorted_U = U[indices]

        # Adjust values to ensure uniqueness
        adjusted = np.zeros((n, 1))
        val = np.clip(sorted_U[0, 0], self.cfg.nurbs.eps, 1.0 - self.cfg.nurbs.eps)
        adjusted[0, 0] = val

        for i in range(1, n):
            val = np.clip(
                max(val + self.cfg.nurbs.eps, sorted_U[i, 0]),
                self.cfg.nurbs.eps,
                1.0 - self.cfg.nurbs.eps,
            )
            adjusted[i, 0] = val

        for i in range(n - 2, -1, -1):
            adjusted[i, 0] = min(
                adjusted[i, 0], adjusted[i + 1, 0] - self.cfg.nurbs.eps
            )
            adjusted[i, 0] = np.clip(
                adjusted[i, 0], self.cfg.nurbs.eps, 1.0 - self.cfg.nurbs.eps
            )

        # Reorder to original order
        U_unique = adjusted[reverse_indices]

        return U_unique.reshape(1, -1)
