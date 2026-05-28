from __future__ import annotations

import os
import queue
import threading
from pathlib import Path

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
import json
import logging
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection
from scipy import interpolate
from splinepy import nurbs

from .optimizer import RacelineOptim
from .utils import gen_centerline_from_img
from .config import Config

class RacelineGUI:
    DELETE_THRESHOLD_FRAC = 0.02
    DEGREE = 3

    # ------------------------------------------------------------------ #
    # Init & helpers                                                     #
    # ------------------------------------------------------------------ #
    def __init__(self, image: str | os.PathLike):
        # ---------- load map & metadata --------------------------------

        self.save_path = Path(image).with_suffix(".json")
        self.image_path = Path(image).expanduser().resolve()
        if not self.image_path.exists():
            raise FileNotFoundError(self.image_path)
        self.yaml_path = self.image_path.with_suffix(".yaml")
        # TODO: change to textbox
        self.save_path = Path(image).with_suffix(".json")
        self._load_image_and_metadata()

        self.extent = [
            self.metadata["origin"][0],
            self.metadata["width"] * self.metadata["resolution"]
            + self.metadata["origin"][0],
            self.metadata["origin"][1],
            self.metadata["height"] * self.metadata["resolution"]
            + self.metadata["origin"][1],
        ]

        # ---------- create figure & main image axes ------------------
        self.fig = plt.figure(figsize=(8, 6))
        gs = self.fig.add_gridspec(1, 1, left=0.00, right=0.75, bottom=0.0, top=1.0)
        self.ax = self.fig.add_subplot(gs[0])
        # self.ax = self.fig.add_axes([0.00, 0.00, 0.75, 1])  # [left, bottom, width, height]
        self.ax.imshow(self.image[::-1, :], cmap="gray", extent=self.extent)
        self.ax.axis("off")

        # ---------- button-strip layout parameters ------------
        strip_left = 0.80  # x-position of the strip
        strip_width = 0.18  # width of each button axes
        btn_height = 0.12  # height of each button axes
        vspace = 0.04  # vertical space between buttons

        # --- dict: label -> callback --------------------------
        button_actions = {
            "Centerline": self._centerline,
            "Spline": self._spline_points,
            "Optimize": self._optimize,
            "Clear": self._clear_points,
            "Save JSON": self._save,
        }

        # ----- compute centred y-positions --------------------
        n = len(button_actions)
        total_height = n * btn_height + (n - 1) * vspace
        start_y = (1 - total_height) / 2
        y_positions = [start_y + i * (btn_height + vspace) for i in range(n)][::-1]

        # ----- create buttons & store in dict -----------------
        self.buttons = {}  # label -> Button object
        for (label, callback), y in zip(button_actions.items(), y_positions):
            ax_btn = self.fig.add_axes([strip_left, y, strip_width, btn_height])
            btn_widget = Button(ax_btn, label)
            btn_widget.on_clicked(callback)
            self.buttons[label] = btn_widget  # save the widget by label

        # state
        self.points: np.ndarray = np.empty((0, 2))
        self._artists: list[plt.Line2D] = []
        self._spline_artist: plt.Line2D | None = None
        self._optim_artist: plt.Line2D | None = None
        self._control_artists: list[plt.Line2D] = []
        self._spline_artists: list[plt.Line2D] = []
        self.spline = None
        self.spline_optim = None
        self.res_optim = None

        # Config
        self.cfg = None

        # ---------- async infrastructure -------------------------------
        self.opt: RacelineOptim | None = None
        self._optim_queue: queue.Queue = queue.Queue()
        self._optim_thread: threading.Thread | None = None
        self._optim_timer = self.fig.canvas.new_timer(interval=100)  # ms
        self._optim_timer.add_callback(self._check_optim_queue)
        self._callback = self._update

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    # ------------------------------------------------------------------ #
    # I/O helpers                                                        #
    # ------------------------------------------------------------------ #
    def _load_image_and_metadata(self):
        image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.flip(image, 0)
        if image is None:
            raise ValueError(f"Could not read image: {self.image_path}")

        with open(self.yaml_path, "r") as fh:
            data = yaml.safe_load(fh)

        self.image = image
        self.metadata = {
            "height": image.shape[0],
            "width": image.shape[1],
            "resolution": data["resolution"],
            "origin": np.array(data["origin"]),
        }

    # ------------------------------------------------------------------ #
    # Button callbacks                                                   #
    # ------------------------------------------------------------------ #
    def _optimize(self, event=None):
        """Launch or stop optimisation in a background thread."""

        # --- toggle: stop if already running ---------------------------
        if self._optim_thread and self._optim_thread.is_alive():
            if self.opt is not None:
                self.opt.stop()
            self.buttons["Optimize"].label.set_text("Stopping…")
            self.fig.canvas.draw_idle()
            return

        # --- start -----------------------------------------------------
        if self.spline_optim is not None:
            spline = self.spline_optim
            print("Optimizing existing spline.")
        elif self.spline is not None:
            spline = self.spline
            print("Using drawn spline for optimisation.")
        else:
            print("Draw a spline first.")
            return

        self.buttons["Optimize"].label.set_text("Stop (0)")

        # --- worker ----------------------------------------------------
        def _worker(image, meta, spline, callback, out_q):
            self.opt = RacelineOptim(self.cfg)
            self.opt.set_image(image, meta)
            self.opt.generate_distance_field()
            if self.cfg.friction_map is True:
                self.opt.generate_friction_map()
            self.opt.set_centerline(spline)
            self.opt.optimize(callback)
            out_q.put(None)  # signal completion

        self._optim_thread = threading.Thread(
            target=_worker,
            args=(self.image, self.metadata, spline, self._callback, self._optim_queue),
            daemon=True,
        )
        self._optim_thread.start()
        self._optim_timer.start()

    def _check_optim_queue(self):
        try:
            item = self._optim_queue.get_nowait()
            if item is None:
                self.buttons["Optimize"].label.set_text("Optimize")
                self._optim_timer.stop()
                self.fig.canvas.draw_idle()
                return
            self.res_optim = item
            self.spline_optim = item[0]
            self._draw_optimized()
        except queue.Empty:
            return

    def _update(self, es):
        self.buttons["Optimize"].label.set_text(f"Stop ({es.countiter})")
        
        best_x = es.best.x
        
        if self.opt:
            current_best_spline = self.opt.sample_NURBS(best_x, self.opt.NURBS_center.copy())
            
            qu = current_best_spline.evaluate(self.opt.u_eval)
            dqu = current_best_spline.derivative(self.opt.u_eval, 1)
            ddqu = current_best_spline.derivative(self.opt.u_eval, 2)
            T = self.opt.get_T_constraint(qu, dqu, ddqu)
            
            self._optim_queue.put((current_best_spline, T, es))

    def _draw_optimized(self):
        self._clear_points()
        if self._optim_artist is not None:
            self._optim_artist.remove()
            self._optim_artist = None

        u = np.linspace(0.0, 1.0, 600)[:, None]

        pts = self.spline_optim.evaluate(u)
        dpts = self.spline_optim.derivative(u, 1)
        speed = np.linalg.norm(dpts, axis=1)

        points = pts.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments, cmap="rainbow", norm=plt.Normalize(speed.min(), speed.max())
        )
        lc.set_array(speed[:-1])
        lc.set_linewidth(2.5)

        # ----------- draw & remember the artist -------------------
        self._optim_artist = self.ax.add_collection(lc)
        self._spline_artists.append(self._optim_artist)

        self.fig.canvas.draw_idle()
        # print("Optimized spline with velocity bar drawn.")

    def _clear_points(self, event=None):
        for art in self._artists:
            if art.axes is not None:
                art.remove()
        self._artists.clear()

        for art in self._control_artists:
            if art.axes is not None:
                art.remove()
        self._control_artists.clear()

        for art in self._spline_artists:
            if art.axes is not None:
                art.remove()
        self._spline_artists.clear()

        for special in ("_spline_artist", "_optim_artist"):
            art = getattr(self, special)
            if art is not None and art.axes is not None:
                art.remove()
            setattr(self, special, None)

        self.points = np.empty((0, 2))
        self.spline = None
        self.fig.canvas.draw_idle()
        # print("Cleared everything.")

    def _save(self, event=None):

        if self.spline_optim is None:
            print("No optimized spline to save.")
            return

        spline, T, es = self.res_optim

        P = np.array(spline.control_points).T
        W = np.array(spline.weights).flatten()
        U = np.array(spline.knot_vectors).flatten()

        data = {
            "P": P.tolist(),
            "W": W.tolist(),
            "U": U.tolist(),
            "p": self.DEGREE,
            "T": T,
        }

        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved optimized spline to {self.save_path}")

    def _spline_points(self, event=None):
        n = self.points.shape[0]
        if n < 3:
            print("Need at least 3 points (currently: %d)", n)
            return

        points = np.vstack([self.points, self.points[0]])

        tck_per, _ = interpolate.splprep(points.T, s=self.cfg.nurbs.s_smooth, k=self.DEGREE, per=1)
        spl_per = interpolate.BSpline(tck_per[0], np.array(tck_per[1]).T, self.DEGREE)
        (dx0, dy0) = interpolate.splev(0, tck_per, der=1)

        bc_x = [(1, dx0)]
        bc_y = [(1, dy0)]

        n_ctrl = self.cfg.nurbs.n_ctrl
        u = np.linspace(0, 1, n_ctrl, endpoint=True)
        pts = spl_per(u)
        spl_x = interpolate.make_interp_spline(
            u, pts[:, 0], k=self.DEGREE, bc_type=(bc_x, bc_x)
        )
        spl_y = interpolate.make_interp_spline(
            u, pts[:, 1], k=self.DEGREE, bc_type=(bc_y, bc_y)
        )

        u_dense = np.linspace(0.0, 1.0, 500)
        x_dense, y_dense = spl_x(u_dense), spl_y(u_dense)

        if self._spline_artist is not None:
            self._spline_artist.remove()
        (self._spline_artist,) = self.ax.plot(
            x_dense, y_dense, "b-", lw=2, label="spline"
        )
        self._spline_artists.append(self._spline_artist)

        P = np.vstack([spl_x.c, spl_y.c]).T
        for art in self._control_artists:
            art.remove()
        self._control_artists.clear()
        (ctrl_line,) = self.ax.plot(P[:, 0], P[:, 1], "g--", lw=1, zorder=3)
        (ctrl_pts,) = self.ax.plot(P[:, 0], P[:, 1], "gs", ms=5, mfc="none", zorder=4)
        self._control_artists.extend([ctrl_line, ctrl_pts])

        self.spline = nurbs.NURBS(
            degrees=[self.DEGREE],
            knot_vectors=[spl_x.t],
            control_points=P,
            weights=np.ones(P.shape[0]),
        )

        self.fig.canvas.draw_idle()
        print("Spline drawn – control points: %d", P.shape[0])

    def _centerline(self, event=None):
        self._clear_points()

        # TODO: Magic Threshold Number
        points = gen_centerline_from_img(self.image, 0.4)
        # TODO: Magic Sparse Number
        points = (
            points[::15, :] * self.metadata["resolution"] + self.metadata["origin"][:2]
        )

        for x, y in points:
            self._add_point(x, y)

    # ------------------------------------------------------------------ #
    # Mouse click handlers                                               #
    # ------------------------------------------------------------------ #
    def _on_click(self, event):
        manager = getattr(self.fig.canvas, "manager", None)
        toolbar = getattr(manager, "toolbar", None) if manager else None
        if toolbar and getattr(toolbar, "mode", ""):
            return

        if event.inaxes is not self.ax:
            return
        if event.button == 1:
            self._add_point(event.xdata, event.ydata)
        elif event.button == 3:
            self._delete_near(event.xdata, event.ydata)

    def _add_point(self, x, y):
        (art,) = self.ax.plot(x, y, "ro", ms=6)
        self._artists.append(art)
        self.points = np.vstack([self.points, [x, y]])
        self.fig.canvas.draw_idle()
        # print("Added point #%d at (%.1f, %.1f)", len(self.points), x, y)

    def _delete_near(self, x, y):
        if self.points.size == 0:
            return
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        thresh = self.DELETE_THRESHOLD_FRAC * max(
            abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0])
        )
        dists = np.hypot(*(self.points.T - np.array([x, y])[:, None]))
        idx = np.argmin(dists)
        if dists[idx] < thresh:
            self._artists[idx].remove()
            del self._artists[idx]
            self.points = np.delete(self.points, idx, axis=0)
            self.fig.canvas.draw_idle()
            print("Deleted point #%d", idx + 1)
        else:
            print("No point close enough to delete.")

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def show(self):
        plt.show()

    def get_points(self) -> np.ndarray:
        return self.points.copy()

    def set_callback(self, callback):
        self._callback = callback

    def set_config(self, path: str = None):
        if path is None:
            self.cfg = Config()
        else:
            self.cfg = Config.from_yaml(path)


def main():

    parser = argparse.ArgumentParser(description="SpatialTrajoptRacing Raceline Optimization GUI")

    parser.add_argument(
        "map_path",
        type=str,
        nargs="?",
        default="maps/f1_aut.png",
        help="Path to the map PNG file (e.g., maps/track.png)",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the optimization config YAML file",
    )
    
    args = parser.parse_args()

    try:
        gui = RacelineGUI(args.map_path)
        gui.set_config(args.config)
        gui.show()
    except FileNotFoundError as e:
        print(f"Failed to load map: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
