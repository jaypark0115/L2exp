from __future__ import annotations

import csv
import argparse
import json
import math
import os
import pickle
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from struct import unpack
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - server dependency may vary
    njit = None


if njit is not None:
    @njit(cache=False)
    def _update_weight_single_numba(w: float, w_del: float) -> float:
        if w_del < 0.0:
            delta = 0.008 * w_del * (w ** 0.9)
            if w < abs(delta):
                return 0.0
            return w + delta
        if w_del > 0.0:
            return w + 0.008 * w_del * ((1.0 - w) ** 0.9)
        return w
else:
    def _update_weight_single_numba(w: float, w_del: float) -> float:
        if w_del < 0.0:
            delta = 0.008 * w_del * (w ** 0.9)
            if w < abs(delta):
                return 0.0
            return w + delta
        if w_del > 0.0:
            return w + 0.008 * w_del * ((1.0 - w) ** 0.9)
        return w


@dataclass
class SNNConfig:
    mode: str
    n_e: int = 625
    n_input: int = 784
    sim_time: float = 0.350
    time_step: float = 0.001
    spike_rate_per_time: int = 350
    train_steps_per_image: int = 300
    test_steps_per_image: int = 350
    relax_steps: int = 300
    train_relax_steps: int = 50
    min_spikes: float = 5.0
    initial_interval: float = 2.0
    interval_increment: float = 1.0
    v_rest_e: float = -0.065
    v_rest_i: float = -0.060
    v_reset_e: float = -0.080
    v_reset_i: float = -0.075
    v_thresh_i: float = -0.055
    tau_e: float = 0.1
    tau_i: float = 0.01
    tau_syn_E: float = 0.001
    tau_syn_I: float = 0.002
    gL: float = 1.0
    vE_E: float = 0.0
    vE_I: float = 0.0
    vI_E: float = -0.240
    inter_i_gE_max: float = 300.0
    sensory_chi: float = 0.9993
    sensory_tau_exp: float = 8.1
    vth_theta_inc: float = 0.00005
    vth_theta_tau_exp: float = 6.1
    initial_theta: float = 0.02
    weight_norm_target: float = 85.5
    seed: int = 1234
    use_gpu: bool = True
    max_interval_attempts: int = 100
    max_input_interval: float = 100.0
    winner_top_k: int = 0
    log_interval: int = 1000
    save_interval: int = 10000

    @property
    def n_i(self) -> int:
        return self.n_e

    @property
    def th(self) -> int:
        return 0 if self.mode.lower() in {"sen", "sensory", "sa"} else 1

    @property
    def canonical_mode(self) -> str:
        return "sen" if self.th == 0 else "vth"


@dataclass
class SampleResult:
    label: int
    predictions: Dict[str, int]
    total_spikes: float
    exc_spikes: float
    inh_spikes: float
    noi: int
    wta_size: int
    interval: float


def default_workspace_root() -> Path:
    return Path(__file__).resolve().parent


def default_snn_root() -> Path:
    env_root = os.environ.get("L1EXP_SNN_ROOT")
    if env_root:
        return Path(env_root)
    candidates = [
        default_workspace_root() / "snn",
        default_workspace_root().parent / "snn",
        Path.cwd() / "snn",
        Path.cwd() / "L1exp" / "snn",
        Path.cwd().parent / "L1exp" / "snn",
        Path(r"C:\Users\jaypa\Desktop\L1exp\snn"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_results_root() -> Path:
    return default_workspace_root() / "results"


def find_snn_dirs(data_root: Path) -> Tuple[Path, Path]:
    sen_dir = data_root / "SNN_Sensory-main - saving - Sen"
    if not sen_dir.exists():
        candidates = [p for p in data_root.iterdir() if p.is_dir() and "Sen" in p.name]
        if not candidates:
            raise FileNotFoundError(f"Could not find Sensory directory under {data_root}")
        sen_dir = candidates[0]

    vth_dir = data_root / "SNN_Sensory-main - saving - VTH - 복사본 - 복사본"
    if not vth_dir.exists():
        candidates = [p for p in data_root.iterdir() if p.is_dir() and "VTH" in p.name]
        if not candidates:
            raise FileNotFoundError(f"Could not find VTH directory under {data_root}")
        vth_dir = candidates[0]
    return sen_dir, vth_dir


def find_training_mnist_dir(data_root: Path) -> Path:
    simple_mnist = data_root / "mnist"
    if simple_mnist.exists():
        return simple_mnist
    sen_dir, _vth_dir = find_snn_dirs(data_root)
    return sen_dir / "mnist"


def get_labeled_data(mnist_dir: Path, name: str, b_train: bool = True) -> Dict[str, np.ndarray]:
    pickle_path = mnist_dir / f"{name}.pickle"
    if pickle_path.exists():
        with pickle_path.open("rb") as f:
            return pickle.load(f)

    if b_train:
        image_path = mnist_dir / "train-images-idx3-ubyte"
        label_path = mnist_dir / "train-labels-idx1-ubyte"
    else:
        image_path = mnist_dir / "t10k-images-idx3-ubyte"
        label_path = mnist_dir / "t10k-labels-idx1-ubyte"

    with image_path.open("rb") as images, label_path.open("rb") as labels:
        images.read(4)
        number_of_images = unpack(">I", images.read(4))[0]
        rows = unpack(">I", images.read(4))[0]
        cols = unpack(">I", images.read(4))[0]

        labels.read(4)
        number_of_labels = unpack(">I", labels.read(4))[0]
        if number_of_images != number_of_labels:
            raise ValueError("number of labels did not match the number of images")

        x = np.zeros((number_of_images, rows, cols), dtype=np.uint16)
        y = np.zeros((number_of_images, 1), dtype=np.uint16)
        for i in range(number_of_images):
            x[i] = [[unpack(">B", images.read(1))[0] for _ in range(cols)] for _ in range(rows)]
            y[i] = unpack(">B", labels.read(1))[0]

    data = {"x": x, "y": y, "rows": rows, "cols": cols}
    with pickle_path.open("wb") as f:
        pickle.dump(data, f)
    return data


def create_random_connections(random_dir: Path, n_e: int, seed: int) -> None:
    random_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    x_to_sen = (rng.random((784, n_e, 1), dtype=np.float64) + 0.01) * 0.3
    sen_in_e = np.ones(n_e, dtype=np.float64) * 8.0
    e_to_i = np.ones(n_e, dtype=np.float64) * 4.0
    i_to_x = np.ones((n_e, n_e), dtype=np.float64) * 4.0
    np.fill_diagonal(i_to_x, 0.0)

    np.save(random_dir / "X_to_Sen.npy", x_to_sen)
    np.save(random_dir / "Sen_in_E.npy", sen_in_e)
    np.save(random_dir / "E_to_I.npy", e_to_i)
    np.save(random_dir / "I_to_X.npy", i_to_x)


def copy_mnist_if_needed(src_mnist: Path, dst_mnist: Path) -> None:
    dst_mnist.mkdir(parents=True, exist_ok=True)
    for name in [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]:
        src = src_mnist / name
        dst = dst_mnist / name
        if not dst.exists():
            shutil.copy2(src, dst)


def readout_names(topk_values: Iterable[int] = (5, 10, 25)) -> List[str]:
    names = ["winner", "class_mean", "class_sum", "class_mean_nonzero"]
    names.extend([f"top{k}_class_sum" for k in topk_values])
    names.extend([f"top{k}_class_mean" for k in topk_values])
    return names


def predict_readouts(count_cpu: np.ndarray, neuron_expect: np.ndarray, names: Iterable[str]) -> Dict[str, int]:
    count_cpu = np.asarray(count_cpu, dtype=np.float32).reshape(-1)
    neuron_expect = np.asarray(neuron_expect).reshape(-1).astype(np.int64)
    predictions: Dict[str, int] = {}

    maxv = np.max(count_cpu)
    winner_idx = np.where(count_cpu == maxv)[0]
    winner_labels = neuron_expect[winner_idx]
    if winner_labels.size:
        vals, counts = np.unique(winner_labels, return_counts=True)
        predictions["winner"] = int(vals[np.argmax(counts)])
    else:
        predictions["winner"] = 0

    for name in names:
        if name == "winner":
            continue
        if name.startswith("top") and "_class_" in name:
            k = int(name[3:].split("_", 1)[0])
            idx = np.argsort(count_cpu)[-k:]
            local_counts = count_cpu[idx]
            local_labels = neuron_expect[idx]
            use_mean = name.endswith("_mean")
            scores = np.full(10, -np.inf, dtype=np.float32)
            for c in range(10):
                class_counts = local_counts[local_labels == c]
                if class_counts.size:
                    scores[c] = float(np.mean(class_counts) if use_mean else np.sum(class_counts))
            predictions[name] = int(np.argmax(scores))
            continue

        scores = np.full(10, -np.inf, dtype=np.float32)
        for c in range(10):
            idx = np.where(neuron_expect == c)[0]
            if idx.size == 0:
                continue
            vals = count_cpu[idx]
            if name == "class_sum":
                scores[c] = float(np.sum(vals))
            elif name == "class_mean":
                scores[c] = float(np.mean(vals))
            elif name == "class_mean_nonzero":
                nonzero = vals[vals > 0.0]
                scores[c] = float(np.mean(nonzero)) if nonzero.size else 0.0
            else:
                raise ValueError(f"Unknown readout mode: {name}")
        predictions[name] = int(np.argmax(scores))

    return predictions


class SNNRunner:
    def __init__(self, config: SNNConfig):
        self.cfg = config
        self._init_backend()
        self._init_random()

        self.Sensory_gI_max = self.xp.ones(self.cfg.n_e, dtype=self.xp.float32)
        self.Inter_I_gE_max = self.xp.ones(self.cfg.n_e, dtype=self.xp.float32) * self.cfg.inter_i_gE_max

    def _init_backend(self) -> None:
        self.use_gpu = False
        if self.cfg.use_gpu:
            try:
                import cupy as cp  # type: ignore

                self.xp = cp
                self.cp = cp
                self.use_gpu = True
            except Exception:
                self.xp = np
                self.cp = None
        else:
            self.xp = np
            self.cp = None

    def _init_random(self) -> None:
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        if self.use_gpu:
            self.xp.random.seed(self.cfg.seed)

    def asnumpy(self, arr):
        if self.use_gpu:
            return self.cp.asnumpy(arr)
        return arr

    def synchronize(self) -> None:
        if self.use_gpu:
            self.cp.cuda.Stream.null.synchronize()

    def initial_state(self):
        xp = self.xp
        excitation = xp.ones(self.cfg.n_e, dtype=xp.float32) * self.cfg.v_rest_e
        inhibition = xp.ones(self.cfg.n_e, dtype=xp.float32) * self.cfg.v_rest_i
        sensory_gE = xp.zeros(self.cfg.n_e, dtype=xp.float32)
        sensory_gI = xp.zeros(self.cfg.n_e, dtype=xp.float32)
        inter_i_gE = xp.zeros(self.cfg.n_e, dtype=xp.float32)
        return excitation, inhibition, sensory_gE, sensory_gI, inter_i_gE

    def poisson_spike_train(self, rate_cpu: np.ndarray, interval: float):
        xp = self.xp
        lam = rate_cpu.astype(np.float32) * self.cfg.time_step / 8.0 * interval
        p = xp.random.uniform(
            0.0,
            1.0,
            (int(self.cfg.n_input), int(self.cfg.spike_rate_per_time)),
        ).astype(xp.float32)
        return xp.where(p < xp.asarray(lam)[:, None], 1.0, 0.0).astype(xp.float32)

    def input_spike(self, weight_xp, spike_xp):
        return self.xp.matmul(
            weight_xp,
            spike_xp.reshape(int(self.cfg.n_input), 1, int(self.cfg.spike_rate_per_time)),
        ).astype(self.xp.float32)

    def e_spike_gen(
        self,
        excitation,
        inhibition,
        sensory_gE,
        sensory_gI,
        inter_i_gE,
        sensory_ge_spike,
        weight_ei,
        weight_ie,
        sensory_gE_max,
        times: int,
        theta,
        adapt: bool,
    ):
        xp = self.xp
        cfg = self.cfg
        sensory_spike = xp.zeros((cfg.n_e, cfg.spike_rate_per_time), dtype=xp.uint16)
        inter_i_spike = xp.zeros((cfg.n_e, cfg.spike_rate_per_time + 1), dtype=xp.uint16)
        zero_input = sensory_ge_spike is None

        for t in range(int(times)):
            v_thresh_e = -0.055 * xp.ones(cfg.n_e, dtype=xp.float32)

            if t < 5:
                spike_part_sum = xp.sum(sensory_spike[:, 0:t], axis=1).astype(xp.uint16)
            else:
                spike_part_sum = xp.sum(sensory_spike[:, t - 5:t], axis=1)

            s_not_neuron = xp.where(spike_part_sum != 0)
            s_neuron = xp.where(spike_part_sum == 0)
            v_thresh_e = v_thresh_e + theta - 0.02

            i_to_e_spike_data = xp.sum(weight_ie * inter_i_spike[:, t], axis=1)
            if zero_input:
                ge_input = 0.0
            else:
                ge_input = xp.sum(sensory_ge_spike[:, :, t], axis=0) * sensory_gE_max

            sensory_gE = sensory_gE * (1 - cfg.time_step / cfg.tau_syn_E) + ge_input
            sensory_gI = sensory_gI * (1 - cfg.time_step / cfg.tau_syn_I) + i_to_e_spike_data * self.Sensory_gI_max

            sensory_dv = (
                -(excitation - cfg.v_rest_e)
                - (sensory_gE / cfg.gL) * (excitation - cfg.vE_E)
                - (sensory_gI / cfg.gL) * (excitation - cfg.vI_E)
            ) * (cfg.time_step / cfg.tau_e)

            excitation = excitation.copy()
            excitation[s_neuron] = excitation[s_neuron] + sensory_dv[s_neuron]
            excitation[s_not_neuron] = cfg.v_rest_e

            sensory_spike[:, t] = xp.where(v_thresh_e < excitation, 1.0, 0.0).astype(xp.float32)
            excitation = xp.where(v_thresh_e < excitation, cfg.v_reset_e, excitation).astype(xp.float32)
            sensory_spike[s_not_neuron, t] = 0.0

            if adapt:
                if cfg.th == 1:
                    theta = xp.where(
                        sensory_spike[:, t] == 1.0,
                        theta + cfg.vth_theta_inc,
                        theta,
                    ).astype(xp.float32)
                    theta = theta + (-theta / (10 ** cfg.vth_theta_tau_exp))
                else:
                    sensory_gE_max = xp.where(
                        sensory_spike[:, t] == 1.0,
                        sensory_gE_max * cfg.sensory_chi,
                        sensory_gE_max,
                    ).astype(xp.float32)
                    sensory_gE_max = sensory_gE_max + (sensory_gE_max / (10 ** cfg.sensory_tau_exp))

            e_to_i_spike_data = weight_ei * sensory_spike[:, t]
            if t < 2:
                i_spike_part_sum = xp.sum(inter_i_spike[:, 0:t], axis=1).astype(xp.uint16)
            else:
                i_spike_part_sum = xp.sum(inter_i_spike[:, t - 2:t], axis=1)
            i_not_neuron = xp.where(i_spike_part_sum != 0)
            i_neuron = xp.where(i_spike_part_sum == 0)

            inter_i_gE = inter_i_gE * (1 - cfg.time_step / cfg.tau_syn_E) + e_to_i_spike_data * self.Inter_I_gE_max
            inter_dv_i = (
                -(inhibition - cfg.v_rest_i)
                - (inter_i_gE / cfg.gL) * (inhibition - cfg.vE_I)
            ) * (cfg.time_step / cfg.tau_e)

            inhibition = inhibition.copy()
            inhibition[i_neuron] = inhibition[i_neuron] + inter_dv_i[i_neuron]
            inhibition[i_not_neuron] = cfg.v_rest_i

            inter_i_spike[:, t + 1] = xp.where(cfg.v_thresh_i < inhibition, 1.0, 0.0).astype(xp.float32)
            inhibition = xp.where(cfg.v_thresh_i < inhibition, cfg.v_reset_i, inhibition).astype(xp.float32)
            inter_i_spike[i_not_neuron, t + 1] = 0.0

        return (
            excitation,
            inhibition,
            sensory_gE,
            sensory_gI,
            inter_i_gE,
            sensory_spike,
            sensory_gE_max,
            theta,
            inter_i_spike,
        )

    def spike_count(self, spike):
        return self.xp.sum(spike, axis=1)

    def winner(self, count_xp) -> np.ndarray:
        count_cpu = np.asarray(self.asnumpy(count_xp)).reshape(-1)
        if count_cpu.size == 0:
            return np.array([], dtype=np.uint16)
        max_count = float(np.max(count_cpu))
        if max_count <= 0.0:
            return np.array([], dtype=np.uint16)
        active_idx = np.where(count_cpu > 0.0)[0]
        if self.cfg.winner_top_k > 0:
            k = min(int(self.cfg.winner_top_k), active_idx.size)
            top_idx = active_idx[np.argpartition(count_cpu[active_idx], -k)[-k:]]
            order = np.argsort(count_cpu[top_idx])[::-1]
            return top_idx[order].astype(np.uint16)
        return np.where(count_cpu == max_count)[0].astype(np.uint16)

    def update_neuron_expect(
        self,
        neuron_fire_num: np.ndarray,
        class_seen: np.ndarray,
        current_expect: np.ndarray,
    ) -> np.ndarray:
        scores = neuron_fire_num / np.maximum(class_seen, 1)
        active = np.sum(neuron_fire_num, axis=1) > 0
        updated = np.asarray(current_expect).copy().astype(np.uint16)
        if np.any(active):
            updated[active] = np.argmax(scores[active], axis=1).astype(np.uint16)
        return updated

    def normalization(self, weight_xp):
        colsums = self.xp.sum(weight_xp, axis=0)
        factors = self.cfg.weight_norm_target / colsums
        return weight_xp * factors

    def load_random_weights(self, random_dir: Path):
        xp = self.xp
        weight_input = xp.asarray(np.load(random_dir / "X_to_Sen.npy").astype(np.float32))
        weight_ei = xp.asarray(np.load(random_dir / "Sen_in_E.npy").astype(np.float32))
        weight_ie = xp.asarray(np.load(random_dir / "I_to_X.npy").astype(np.float32))
        return weight_input, weight_ei, weight_ie

    def load_checkpoint_or_arrays(self, model_dir: Path):
        xp = self.xp
        weight_file = model_dir / "weight.npy"
        expect_file = model_dir / "neuron_expect.npy"
        gmax_file = model_dir / "Sensory_gE_max.npy"
        theta_file = model_dir / "theta.npy"
        if weight_file.exists() and expect_file.exists() and gmax_file.exists():
            weight = xp.asarray(np.load(weight_file).astype(np.float32))
            neuron_expect = np.load(expect_file)
            sensory_gE_max = xp.asarray(np.load(gmax_file).astype(np.float32))
            if theta_file.exists():
                theta = xp.asarray(np.load(theta_file).astype(np.float32))
            else:
                theta = xp.ones(self.cfg.n_e, dtype=xp.float32) * self.cfg.initial_theta
            return weight, neuron_expect, sensory_gE_max, theta

        checkpoint_file = model_dir / "checkpoint.npz"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"No checkpoint or npy arrays found in {model_dir}")
        data = np.load(checkpoint_file, allow_pickle=True)
        weight = xp.asarray(data["weight"].astype(np.float32))
        neuron_expect = data["neuron_expect"]
        sensory_gE_max = xp.asarray(data["Sensory_gE_max"].astype(np.float32))
        theta = xp.asarray(data["theta"].astype(np.float32))
        return weight, neuron_expect, sensory_gE_max, theta

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        iteration: int,
        weight_input,
        neuron_expect: np.ndarray,
        sensory_gE_max,
        theta,
    ) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            checkpoint_path,
            iter=iteration,
            weight=np.asarray(self.asnumpy(weight_input), dtype=np.float32),
            neuron_expect=neuron_expect,
            Sensory_gE_max=np.asarray(self.asnumpy(sensory_gE_max), dtype=np.float32),
            theta=np.asarray(self.asnumpy(theta), dtype=np.float32),
        )

    def _find_nearest_over(self, array: np.ndarray, value: int) -> Optional[int]:
        over_array = array[np.where(array >= value)]
        if over_array.size == 0:
            return None
        return int(over_array[(np.abs(over_array - value)).argmin()])

    def _find_nearest_under(self, array: np.ndarray, value: int, neuron_train_count: np.ndarray) -> Optional[int]:
        under_array = array[np.where(array <= value)]
        if under_array.size == 0:
            return None
        under_value = int(under_array[(np.abs(under_array - value)).argmin()])
        idx = np.where(array == under_value)
        if neuron_train_count[idx].size and neuron_train_count[idx][0] == 0:
            neuron_train_count[idx] = 1
        elif neuron_train_count[idx].size and neuron_train_count[idx][0] == 1:
            return None
        return under_value

    def stdp_cpu(self, pre_xp, post_xp, winner_idx: np.ndarray, weight_xp):
        pre = np.asarray(self.asnumpy(pre_xp))
        post = np.asarray(self.asnumpy(post_xp))
        weight = np.asarray(self.asnumpy(weight_xp)).copy()

        a, b, c = np.where(pre == 1)
        d, e = np.where(post == 1)
        pre_time = np.stack((a, b, c)).T if a.size else np.empty((0, 3), dtype=int)
        post_time = np.stack((d, e)).T if d.size else np.empty((0, 2), dtype=int)

        for neuron in np.asarray(winner_idx, dtype=np.int64):
            pre_time_arr = pre_time[np.where(pre_time[:, 1] == neuron)] if pre_time.size else np.empty((0, 3), dtype=int)
            post_time_arr = post_time[np.where(post_time[:, 0] == neuron)] if post_time.size else np.empty((0, 2), dtype=int)
            pre_inputs = np.unique(pre_time_arr[:, 0]) if pre_time_arr.size else np.array([], dtype=int)
            if post_time_arr.size == 0:
                continue

            for input_idx in pre_inputs:
                pre_for_input = pre_time_arr[np.where(pre_time_arr[:, 0] == input_idx)]
                setting = np.zeros(len(pre_for_input))
                for post_t in post_time_arr[:, 1]:
                    under = self._find_nearest_under(pre_for_input[:, 2], int(post_t), setting)
                    if post_t == under:
                        w_del = 0.0
                    elif under is None:
                        over = self._find_nearest_over(pre_for_input[:, 2], int(post_t))
                        if post_t == over:
                            w_del = 0.0
                        elif over is None:
                            w_del = -math.exp(-1.0 / 5.0)
                        else:
                            w_del = -math.exp((int(post_t) - over) / 40.0)
                    else:
                        w_del = math.exp(-(int(post_t) - under) / 20.0)

                    if w_del != 0.0:
                        if weight.ndim == 3:
                            weight_index = (int(input_idx), int(neuron), 0)
                        else:
                            weight_index = (int(input_idx), int(neuron))
                        wij = float(weight[weight_index])
                        weight[weight_index] = _update_weight_single_numba(wij, float(w_del))

        return self.xp.asarray(weight) if self.use_gpu else weight

    def test(
        self,
        mnist_dir: Path,
        random_dir: Path,
        model_dir: Path,
        output_dir: Path,
        readouts: Optional[List[str]] = None,
        test_limit: int = 10000,
        write_predictions: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        readouts = readouts or readout_names()
        output_dir.mkdir(parents=True, exist_ok=True)

        testing = get_labeled_data(mnist_dir, "testing", b_train=False)
        testing_data = np.array(testing["x"]).astype(np.float32)
        testing_label = np.array(testing["y"]).astype(np.uint16)

        weight_input, weight_ei, weight_ie = self.load_random_weights(random_dir)
        weight_input, neuron_expect, sensory_gE_max, theta = self.load_checkpoint_or_arrays(model_dir)

        excitation, inhibition, sensory_gE, sensory_gI, inter_i_gE = self.initial_state()
        initial_excitation = self.xp.copy(excitation)
        initial_inhibition = self.xp.copy(inhibition)

        correct = {name: 0 for name in readouts}
        spike_sum = 0.0
        noi_sum = 0.0
        wta_size_sum = 0.0
        single_wta = 0
        sample_count = 0
        rows: List[Dict[str, object]] = []
        interval = self.cfg.initial_interval
        start = time.time()

        for i in range(test_limit):
            start_excitation = self.xp.copy(excitation)
            start_inhibition = self.xp.copy(inhibition)
            start_theta = self.xp.copy(theta)
            count = self.xp.zeros(self.cfg.n_e, dtype=self.xp.float32)

            data_cpu = testing_data[i % 10000, :, :].reshape(self.cfg.n_input).astype(np.float32)
            label = int(testing_label[i % 10000, 0])
            used_interval = interval

            interval_attempts = 0
            while (
                float(self.xp.sum(count)) < self.cfg.min_spikes
                and interval_attempts < self.cfg.max_interval_attempts
            ):
                interval_attempts += 1
                if i == 0:
                    excitation = self.xp.copy(initial_excitation)
                    inhibition = self.xp.copy(initial_inhibition)
                    theta = self.xp.copy(start_theta)
                else:
                    excitation = self.xp.copy(start_excitation)
                    inhibition = self.xp.copy(start_inhibition)
                    theta = self.xp.copy(start_theta)

                current_spike = self.poisson_spike_train(data_cpu, interval)
                sensory_ge_spike = self.input_spike(weight_input, current_spike)
                (
                    excitation,
                    inhibition,
                    sensory_gE,
                    sensory_gI,
                    inter_i_gE,
                    sensory_spike,
                    sensory_gE_max,
                    theta,
                    inter_i_spike,
                ) = self.e_spike_gen(
                    excitation,
                    inhibition,
                    sensory_gE,
                    sensory_gI,
                    inter_i_gE,
                    sensory_ge_spike,
                    weight_ei,
                    weight_ie,
                    sensory_gE_max,
                    self.cfg.test_steps_per_image,
                    theta,
                    adapt=False,
                )

                count = self.spike_count(sensory_spike)
                used_interval = interval
                if float(self.xp.sum(count)) < self.cfg.min_spikes:
                    interval = min(interval + self.cfg.interval_increment, self.cfg.max_input_interval)

            count_cpu = np.asarray(self.asnumpy(count), dtype=np.float32)
            predictions = predict_readouts(count_cpu, neuron_expect, readouts)
            for name, pred in predictions.items():
                if pred == label:
                    correct[name] += 1

            winner_idx = self.winner(count)
            exc_spikes = float(self.xp.sum(count))
            inh_spikes = float(self.xp.sum(inter_i_spike))
            total_spikes = exc_spikes + inh_spikes
            noi = int(self.xp.count_nonzero(count))
            wta_size = int(winner_idx.size)

            spike_sum += total_spikes
            noi_sum += noi
            wta_size_sum += wta_size
            single_wta += 1 if wta_size == 1 else 0
            sample_count += 1

            rows.append(
                {
                    "sample": i,
                    "label": label,
                    "total_spikes": total_spikes,
                    "exc_spikes": exc_spikes,
                    "inh_spikes": inh_spikes,
                    "noi": noi,
                    "wta_size": wta_size,
                    "interval": used_interval,
                    **{f"pred_{name}": pred for name, pred in predictions.items()},
                }
            )

            (
                excitation,
                inhibition,
                sensory_gE,
                sensory_gI,
                inter_i_gE,
                _sensory_spike,
                sensory_gE_max,
                theta,
                _inter_i_spike,
            ) = self.e_spike_gen(
                excitation,
                inhibition,
                sensory_gE,
                sensory_gI,
                inter_i_gE,
                None,
                weight_ei,
                weight_ie,
                sensory_gE_max,
                self.cfg.relax_steps,
                theta,
                adapt=False,
            )

            if float(self.xp.sum(count)) > self.cfg.min_spikes:
                interval = self.cfg.initial_interval

            if (i + 1) % 1000 == 0:
                self.synchronize()
                elapsed = time.time() - start
                best = max(correct, key=lambda name: correct[name])
                best_acc = correct[best] / float(i + 1) * 100.0
                print(f"[TEST {self.cfg.canonical_mode} {i + 1}/{test_limit}] best={best} acc={best_acc:.2f}% elapsed={elapsed/60:.1f} min")

        self.synchronize()
        elapsed = time.time() - start
        common = {
            "avg_spikes": spike_sum / max(sample_count, 1),
            "avg_noi": noi_sum / max(sample_count, 1),
            "avg_wta_size": wta_size_sum / max(sample_count, 1),
            "single_wta_ratio": 100.0 * single_wta / max(sample_count, 1),
            "samples": sample_count,
            "elapsed_sec": elapsed,
        }
        summary: Dict[str, Dict[str, float]] = {}
        for name in readouts:
            summary[name] = {
                "accuracy": correct[name] / max(sample_count, 1) * 100.0,
                **common,
            }

        metadata = {
            "config": asdict(self.cfg),
            "mnist_dir": str(mnist_dir),
            "random_dir": str(random_dir),
            "model_dir": str(model_dir),
            "output_dir": str(output_dir),
            "readouts": readouts,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._write_summary_csv(output_dir / "summary.csv", summary)
        self._write_summary_txt(output_dir / "summary.txt", summary)
        if write_predictions:
            self._write_predictions_csv(output_dir / "predictions.csv", rows)
        return summary

    def train(
        self,
        mnist_dir: Path,
        random_dir: Path,
        model_dir: Path,
        train_iters: int = 60000,
        resume: bool = True,
    ) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        training = get_labeled_data(mnist_dir, "training", b_train=True)
        training_data = np.array(training["x"]).astype(np.float32)
        training_label = np.array(training["y"]).astype(np.uint16)

        weight_input, weight_ei, weight_ie = self.load_random_weights(random_dir)
        sensory_gE_max = self.xp.ones(self.cfg.n_e, dtype=self.xp.float32)
        theta = self.xp.ones(self.cfg.n_e, dtype=self.xp.float32) * self.cfg.initial_theta
        neuron_expect = np.zeros(self.cfg.n_e, dtype=np.uint16)
        neuron_fire_num = np.zeros((self.cfg.n_e, 10), dtype=np.uint16)
        class_seen = np.zeros(10, dtype=np.uint16)
        start_iter = 0

        checkpoint_path = model_dir / "checkpoint.npz"
        if resume and checkpoint_path.exists():
            data = np.load(checkpoint_path, allow_pickle=True)
            start_iter = int(data["iter"]) + 1
            weight_input = self.xp.asarray(data["weight"].astype(np.float32))
            neuron_expect = data["neuron_expect"]
            sensory_gE_max = self.xp.asarray(data["Sensory_gE_max"].astype(np.float32))
            theta = self.xp.asarray(data["theta"].astype(np.float32))
            print(f"[RESUME] {self.cfg.canonical_mode} from iter {start_iter}")

        excitation, inhibition, sensory_gE, sensory_gI, inter_i_gE = self.initial_state()
        initial_excitation = self.xp.copy(excitation)
        initial_inhibition = self.xp.copy(inhibition)
        interval = self.cfg.initial_interval

        performance_count = 0.0
        window_spike_sum = 0.0
        window_noi_sum = 0.0
        window_wta_size_sum = 0.0
        window_single_wta = 0
        window_sample_count = 0
        log_performance_count = 0.0
        log_spike_sum = 0.0
        log_noi_sum = 0.0
        log_wta_size_sum = 0.0
        log_single_wta = 0
        log_sample_count = 0
        start = time.time()

        log_path = model_dir / "train_log.txt"
        if start_iter == 0:
            log_path.write_text("", encoding="utf-8")

        for i in range(start_iter, train_iters):
            label = int(training_label[i % 60000][0])
            data_cpu = training_data[i % 60000, :, :].reshape(self.cfg.n_input)
            count = self.xp.zeros(self.cfg.n_e, dtype=self.xp.float32)
            start_excitation = self.xp.copy(excitation).astype(self.xp.float32)
            start_inhibition = self.xp.copy(inhibition).astype(self.xp.float32)
            start_theta = self.xp.copy(theta).astype(self.xp.float32)

            weight_input = self.normalization(weight_input)

            interval_attempts = 0
            while (
                float(self.xp.sum(count)) < self.cfg.min_spikes
                and interval_attempts < self.cfg.max_interval_attempts
            ):
                interval_attempts += 1
                if i == 0:
                    excitation = self.xp.copy(initial_excitation)
                    inhibition = self.xp.copy(initial_inhibition)
                    theta = self.xp.copy(start_theta)
                else:
                    excitation = self.xp.copy(start_excitation)
                    inhibition = self.xp.copy(start_inhibition)
                    theta = self.xp.copy(start_theta)

                current_spike = self.poisson_spike_train(data_cpu, interval)
                sensory_ge_spike = self.input_spike(weight_input, current_spike)
                (
                    excitation,
                    inhibition,
                    sensory_gE,
                    sensory_gI,
                    inter_i_gE,
                    sensory_spike,
                    sensory_gE_max,
                    theta,
                    inter_i_spike,
                ) = self.e_spike_gen(
                    excitation,
                    inhibition,
                    sensory_gE,
                    sensory_gI,
                    inter_i_gE,
                    sensory_ge_spike,
                    weight_ei,
                    weight_ie,
                    sensory_gE_max,
                    self.cfg.train_steps_per_image,
                    theta,
                    adapt=True,
                )

                count = self.spike_count(sensory_spike)
                if float(self.xp.sum(count)) < self.cfg.min_spikes:
                    interval = min(interval + self.cfg.interval_increment, self.cfg.max_input_interval)

            count_cpu = np.asarray(self.asnumpy(count), dtype=np.float32)
            pred = predict_readouts(count_cpu, neuron_expect, ["winner"])["winner"]
            winner_idx = self.winner(count)

            exc_spikes = float(self.xp.sum(count))
            inh_spikes = float(self.xp.sum(inter_i_spike))
            total_spikes = exc_spikes + inh_spikes
            noi = int(self.xp.count_nonzero(count))
            wta_size = int(winner_idx.size)

            window_spike_sum += total_spikes
            window_noi_sum += noi
            window_wta_size_sum += wta_size
            window_single_wta += 1 if wta_size == 1 else 0
            window_sample_count += 1
            log_spike_sum += total_spikes
            log_noi_sum += noi
            log_wta_size_sum += wta_size
            log_single_wta += 1 if wta_size == 1 else 0
            log_sample_count += 1

            neuron_fire_num[winner_idx, label] += 1
            class_seen[label] += 1
            if float(self.xp.sum(count)) != 0.0:
                interval = self.cfg.initial_interval

            pre_spike = self.xp.where(sensory_ge_spike != 0.0, 1.0, 0.0)
            weight_input = self.stdp_cpu(pre_spike, sensory_spike, winner_idx, weight_input)

            (
                excitation,
                inhibition,
                sensory_gE,
                sensory_gI,
                inter_i_gE,
                _sensory_spike,
                sensory_gE_max,
                theta,
                _inter_i_spike,
            ) = self.e_spike_gen(
                excitation,
                inhibition,
                sensory_gE,
                sensory_gI,
                inter_i_gE,
                None,
                weight_ei,
                weight_ie,
                sensory_gE_max,
                self.cfg.train_relax_steps,
                theta,
                adapt=True,
            )

            if pred == label:
                performance_count += 1.0
                log_performance_count += 1.0

            should_log = i % self.cfg.log_interval == self.cfg.log_interval - 1 or i == train_iters - 1
            should_save = i % self.cfg.save_interval == self.cfg.save_interval - 1 or i == train_iters - 1

            if should_save:
                neuron_expect = self.update_neuron_expect(neuron_fire_num, class_seen, neuron_expect)

            if should_log:
                acc = log_performance_count / max(log_sample_count, 1) * 100.0
                avg_spikes = log_spike_sum / max(log_sample_count, 1)
                avg_noi = log_noi_sum / max(log_sample_count, 1)
                avg_wta_size = log_wta_size_sum / max(log_sample_count, 1)
                single_wta_ratio = log_single_wta / max(log_sample_count, 1) * 100.0
                self.synchronize()
                elapsed = time.time() - start
                msg = (
                    f"[iter {i + 1}] accuracy = {acc:.2f} %\n"
                    f"  avg spikes per image (exc+inh): {avg_spikes:.2f}\n"
                    f"  avg NOI: {avg_noi:.2f}\n"
                    f"  avg WTA winner size: {avg_wta_size:.2f}\n"
                    f"  single-WTA ratio: {single_wta_ratio:.2f} %\n"
                    f"train time = {elapsed:.1f} s\n"
                )
                print(msg.rstrip())
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(msg)

                log_performance_count = 0.0
                log_spike_sum = 0.0
                log_noi_sum = 0.0
                log_wta_size_sum = 0.0
                log_single_wta = 0
                log_sample_count = 0

            if should_save:
                self.save_checkpoint(checkpoint_path, i, weight_input, neuron_expect, sensory_gE_max, theta)
                self.save_checkpoint(
                    model_dir / f"checkpoint_{i + 1:08d}.npz",
                    i,
                    weight_input,
                    neuron_expect,
                    sensory_gE_max,
                    theta,
                )

                performance_count = 0.0
                neuron_fire_num = np.zeros((self.cfg.n_e, 10), dtype=np.uint16)
                class_seen = np.zeros(10, dtype=np.uint16)
                window_spike_sum = 0.0
                window_noi_sum = 0.0
                window_wta_size_sum = 0.0
                window_single_wta = 0
                window_sample_count = 0

        np.save(model_dir / "weight.npy", np.asarray(self.asnumpy(weight_input), dtype=np.float32))
        np.save(model_dir / "neuron_expect.npy", neuron_expect)
        np.save(model_dir / "neuron_fire_num.npy", neuron_fire_num)
        np.save(model_dir / "Sensory_gE_max.npy", np.asarray(self.asnumpy(sensory_gE_max), dtype=np.float32))
        if self.cfg.th == 1:
            np.save(model_dir / "theta.npy", np.asarray(self.asnumpy(theta), dtype=np.float32))
        (model_dir / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2), encoding="utf-8")
        print(f"[TRAIN DONE] {self.cfg.canonical_mode} output={model_dir}")

    def assign_labels(
        self,
        mnist_dir: Path,
        random_dir: Path,
        model_dir: Path,
        assignment_limit: int = 10000,
        force: bool = False,
    ) -> None:
        if assignment_limit <= 0:
            return

        marker_path = model_dir / f"label_assignment_{assignment_limit}.json"
        if marker_path.exists() and not force:
            print(f"[SKIP LABEL ASSIGN] {self.cfg.canonical_mode} already complete: {marker_path}")
            return

        training = get_labeled_data(mnist_dir, "training", b_train=True)
        training_data = np.array(training["x"]).astype(np.float32)
        training_label = np.array(training["y"]).astype(np.uint16)

        weight_input, weight_ei, weight_ie = self.load_random_weights(random_dir)
        weight_input, neuron_expect, sensory_gE_max, theta = self.load_checkpoint_or_arrays(model_dir)
        assignment_limit = min(int(assignment_limit), int(training_data.shape[0]))

        excitation, inhibition, sensory_gE, sensory_gI, inter_i_gE = self.initial_state()
        initial_excitation = self.xp.copy(excitation)
        initial_inhibition = self.xp.copy(inhibition)
        response_by_label = np.zeros((self.cfg.n_e, 10), dtype=np.float64)
        class_seen = np.zeros(10, dtype=np.uint32)
        interval = self.cfg.initial_interval
        start = time.time()

        print(f"[LABEL ASSIGN] mode={self.cfg.canonical_mode} samples={assignment_limit}")
        for i in range(assignment_limit):
            start_excitation = self.xp.copy(excitation)
            start_inhibition = self.xp.copy(inhibition)
            start_theta = self.xp.copy(theta)
            count = self.xp.zeros(self.cfg.n_e, dtype=self.xp.float32)

            data_cpu = training_data[i, :, :].reshape(self.cfg.n_input).astype(np.float32)
            label = int(training_label[i, 0])

            interval_attempts = 0
            while (
                float(self.xp.sum(count)) < self.cfg.min_spikes
                and interval_attempts < self.cfg.max_interval_attempts
            ):
                interval_attempts += 1
                if i == 0:
                    excitation = self.xp.copy(initial_excitation)
                    inhibition = self.xp.copy(initial_inhibition)
                    theta = self.xp.copy(start_theta)
                else:
                    excitation = self.xp.copy(start_excitation)
                    inhibition = self.xp.copy(start_inhibition)
                    theta = self.xp.copy(start_theta)

                current_spike = self.poisson_spike_train(data_cpu, interval)
                sensory_ge_spike = self.input_spike(weight_input, current_spike)
                (
                    excitation,
                    inhibition,
                    sensory_gE,
                    sensory_gI,
                    inter_i_gE,
                    sensory_spike,
                    sensory_gE_max,
                    theta,
                    inter_i_spike,
                ) = self.e_spike_gen(
                    excitation,
                    inhibition,
                    sensory_gE,
                    sensory_gI,
                    inter_i_gE,
                    sensory_ge_spike,
                    weight_ei,
                    weight_ie,
                    sensory_gE_max,
                    self.cfg.test_steps_per_image,
                    theta,
                    adapt=False,
                )

                count = self.spike_count(sensory_spike)
                if float(self.xp.sum(count)) < self.cfg.min_spikes:
                    interval = min(interval + self.cfg.interval_increment, self.cfg.max_input_interval)

            count_cpu = np.asarray(self.asnumpy(count), dtype=np.float32)
            response_by_label[:, label] += count_cpu
            class_seen[label] += 1

            (
                excitation,
                inhibition,
                sensory_gE,
                sensory_gI,
                inter_i_gE,
                _sensory_spike,
                sensory_gE_max,
                theta,
                _inter_i_spike,
            ) = self.e_spike_gen(
                excitation,
                inhibition,
                sensory_gE,
                sensory_gI,
                inter_i_gE,
                None,
                weight_ei,
                weight_ie,
                sensory_gE_max,
                self.cfg.relax_steps,
                theta,
                adapt=False,
            )

            if float(self.xp.sum(count)) > self.cfg.min_spikes:
                interval = self.cfg.initial_interval

            if (i + 1) % 1000 == 0 or i == assignment_limit - 1:
                self.synchronize()
                elapsed = time.time() - start
                active = int(np.sum(np.sum(response_by_label, axis=1) > 0))
                print(f"[LABEL ASSIGN {self.cfg.canonical_mode} {i + 1}/{assignment_limit}] active={active} elapsed={elapsed/60:.1f} min")

        scores = response_by_label / np.maximum(class_seen, 1)
        active = np.sum(response_by_label, axis=1) > 0
        updated_expect = np.asarray(neuron_expect).copy().astype(np.uint16)
        if np.any(active):
            updated_expect[active] = np.argmax(scores[active], axis=1).astype(np.uint16)

        np.save(model_dir / "neuron_expect.npy", updated_expect)
        np.save(model_dir / "label_response_by_class.npy", response_by_label)

        checkpoint_path = model_dir / "checkpoint.npz"
        iteration = -1
        if checkpoint_path.exists():
            checkpoint = np.load(checkpoint_path, allow_pickle=True)
            iteration = int(checkpoint["iter"])
        self.save_checkpoint(checkpoint_path, iteration, weight_input, updated_expect, sensory_gE_max, theta)

        dist = {
            int(label): int(count)
            for label, count in zip(*np.unique(updated_expect, return_counts=True))
        }
        marker = {
            "mode": self.cfg.canonical_mode,
            "assignment_limit": assignment_limit,
            "active_neurons": int(np.sum(active)),
            "class_seen": class_seen.astype(int).tolist(),
            "neuron_expect_distribution": dist,
            "elapsed_sec": time.time() - start,
        }
        marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
        print(f"[LABEL ASSIGN DONE] mode={self.cfg.canonical_mode} active={marker['active_neurons']} dist={dist}")

    def _write_summary_csv(self, path: Path, summary: Dict[str, Dict[str, float]]) -> None:
        fields = ["readout", "accuracy", "avg_spikes", "avg_noi", "avg_wta_size", "single_wta_ratio", "samples", "elapsed_sec"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for name, metrics in summary.items():
                writer.writerow({"readout": name, **metrics})

    def _write_summary_txt(self, path: Path, summary: Dict[str, Dict[str, float]]) -> None:
        lines = []
        for name, metrics in sorted(summary.items(), key=lambda item: item[1]["accuracy"], reverse=True):
            lines.append(
                f"{name}: accuracy={metrics['accuracy']:.4f}% "
                f"avg_spikes={metrics['avg_spikes']:.4f} "
                f"avg_noi={metrics['avg_noi']:.4f} "
                f"single_wta={metrics['single_wta_ratio']:.4f}%"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_predictions_csv(self, path: Path, rows: List[Dict[str, object]]) -> None:
        if not rows:
            return
        fields = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def checkpoint_complete(model_dir: Path, train_iters: int) -> bool:
    checkpoint = model_dir / "checkpoint.npz"
    if not checkpoint.exists():
        return False
    try:
        data = np.load(checkpoint, allow_pickle=True)
        return int(data["iter"]) >= train_iters - 1
    except Exception:
        return False


def test_complete(output_dir: Path, test_limit: int, readouts: List[str]) -> bool:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for name in readouts:
        metrics = summary.get(name)
        if not metrics or int(metrics.get("samples", 0)) < test_limit:
            return False
    return True


def run_l1_training_only(
    data_root: Path,
    results_root: Path,
    run_name: str,
    modes: Iterable[str],
    n_e: int,
    train_iters: int,
    test_limit: int,
    seed: int,
    use_gpu: bool,
    readouts: List[str],
    resume: bool,
    v_i_e: float,
    sensory_chi: float,
    sensory_tau_exp: float,
    assignment_limit: int,
    force: bool = False,
) -> None:
    source_mnist = find_training_mnist_dir(data_root)

    for mode in modes:
        mode = mode.lower()
        run_dir = results_root / f"l1_{n_e}" / run_name / mode
        mnist_dir = source_mnist
        random_dir = run_dir / "random"
        model_dir = run_dir / ("weight_Sen" if mode == "sen" else "weight_th")
        test_dir = run_dir / "test"

        if not (random_dir / "X_to_Sen.npy").exists():
            create_random_connections(random_dir, n_e=n_e, seed=seed)

        fast_params = {}
        if mode == "sen":
            fast_params = {
                "min_spikes": 5.0,
                "initial_interval": 2.0,
                "interval_increment": 1.0,
                "max_interval_attempts": 32,
                "max_input_interval": 64.0,
                "winner_top_k": 25,
            }
        elif mode == "vth":
            fast_params = {
                "min_spikes": 5.0,
                "initial_interval": 2.0,
                "interval_increment": 1.0,
                "initial_theta": 0.02,
                "vth_theta_inc": 0.00005,
                "max_interval_attempts": 32,
                "max_input_interval": 64.0,
                "winner_top_k": 10,
            }

        cfg = SNNConfig(
            mode=mode,
            n_e=n_e,
            seed=seed,
            use_gpu=use_gpu,
            vI_E=v_i_e,
            sensory_chi=sensory_chi,
            sensory_tau_exp=sensory_tau_exp,
            **fast_params,
        )
        runner = SNNRunner(cfg)
        if not force and checkpoint_complete(model_dir, train_iters):
            print(f"[SKIP L1 TRAIN] mode={mode} already complete: {model_dir}")
        else:
            print(f"[L1 TRAIN] mode={mode} n_e={n_e} out={model_dir}")
            runner.train(
                mnist_dir=mnist_dir,
                random_dir=random_dir,
                model_dir=model_dir,
                train_iters=train_iters,
                resume=resume,
            )

        if checkpoint_complete(model_dir, train_iters):
            runner.assign_labels(
                mnist_dir=mnist_dir,
                random_dir=random_dir,
                model_dir=model_dir,
                assignment_limit=assignment_limit,
                force=force,
            )

        if not force and test_complete(test_dir, test_limit, readouts):
            print(f"[SKIP L1 TEST] mode={mode} already complete: {test_dir}")
        else:
            print(f"[L1 TEST] mode={mode} n_e={n_e} out={test_dir}")
            runner.test(
                mnist_dir=mnist_dir,
                random_dir=random_dir,
                model_dir=model_dir,
                output_dir=test_dir,
                readouts=readouts,
                test_limit=test_limit,
                write_predictions=True,
            )


def build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone single-file runner for L1 Vth 120k training and test."
    )
    parser.add_argument("--data-root", type=Path, default=default_snn_root(), help="Path to the original snn folder.")
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    parser.add_argument("--run-name", default="server_run_l1_vth_rein_120k_001")
    parser.add_argument("--modes", nargs="+", default=["vth"], choices=["vth"])
    parser.add_argument("--n-e", type=int, default=1250)
    parser.add_argument("--train-iters", type=int, default=120000)
    parser.add_argument("--test-limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu", action="store_true", help="Force NumPy CPU backend.")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch even if checkpoint.npz exists.")
    parser.add_argument("--v-i-e", type=float, default=-0.240)
    parser.add_argument("--sensory-chi", type=float, default=0.9993)
    parser.add_argument("--sensory-tau-exp", type=float, default=8.1)
    parser.add_argument("--assignment-limit", type=int, default=10000, help="Training samples used for post-STDP neuron label assignment. Use 0 to skip.")
    parser.add_argument("--topk", nargs="*", type=int, default=[5, 10, 25])
    parser.add_argument("--force", action="store_true", help="Re-run train/test even if outputs already exist.")
    return parser


def main() -> None:
    args = build_standalone_parser().parse_args()
    print(f"[STANDALONE L1] run_name={args.run_name}")
    print(f"[STANDALONE L1] data_root={args.data_root}")
    print(f"[STANDALONE L1] results_root={args.results_root}")
    run_l1_training_only(
        data_root=args.data_root,
        results_root=args.results_root,
        run_name=args.run_name,
        modes=args.modes,
        n_e=args.n_e,
        train_iters=args.train_iters,
        test_limit=args.test_limit,
        seed=args.seed,
        use_gpu=not args.cpu,
        readouts=readout_names(args.topk),
        resume=not args.no_resume,
        v_i_e=args.v_i_e,
        sensory_chi=args.sensory_chi,
        sensory_tau_exp=args.sensory_tau_exp,
        assignment_limit=args.assignment_limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
