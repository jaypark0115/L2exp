from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from struct import unpack
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - server dependency may vary
    cp = None

try:
    from numba import njit
except Exception:  # pragma: no cover - server dependency may vary
    njit = None


if njit is not None:
    @njit(cache=True)
    def _update_weight_single(w: float, w_del: float) -> float:
        if w_del < 0.0:
            delta = 0.008 * w_del * (w ** 0.9)
            if w < abs(delta):
                return 0.0
            return w + delta
        if w_del > 0.0:
            return w + 0.008 * w_del * ((1.0 - w) ** 0.9)
        return w
else:
    def _update_weight_single(w: float, w_del: float) -> float:
        if w_del < 0.0:
            delta = 0.008 * w_del * (w ** 0.9)
            if w < abs(delta):
                return 0.0
            return w + delta
        if w_del > 0.0:
            return w + 0.008 * w_del * ((1.0 - w) ** 0.9)
        return w


@dataclass
class SNNLayerConfig:
    mode: str
    n_input: int
    n_e: int = 625
    spike_rate_per_time: int = 350
    train_steps_per_image: int = 300
    test_steps_per_image: int = 350
    train_relax_steps: int = 50
    test_relax_steps: int = 300
    min_spikes: float = 5.0
    initial_interval: float = 1.0
    interval_increment: float = 0.5
    time_step: float = 0.001
    v_rest_e: float = -0.065
    v_rest_i: float = -0.060
    v_reset_e: float = -0.080
    v_reset_i: float = -0.075
    v_thresh_i: float = -0.055
    tau_e: float = 0.1
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
    input_gain: float = 1.0
    max_gain_attempts: int = 100
    max_input_gain: float = 100.0
    winner_top_k: int = 0
    seed: int = 1234
    use_gpu: bool = True
    save_interval: int = 1000

    @property
    def th(self) -> int:
        return 0 if self.mode.lower() in {"sen", "sensory", "sa"} else 1

    @property
    def canonical_mode(self) -> str:
        return "sen" if self.th == 0 else "vth"


def workspace_root() -> Path:
    return Path(__file__).resolve().parent


def default_data_root() -> Path:
    return workspace_root() / "snn"


def default_results_root() -> Path:
    return workspace_root() / "results"


def load_idx_dataset(mnist_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    if split == "training":
        image_path = mnist_dir / "train-images-idx3-ubyte"
        label_path = mnist_dir / "train-labels-idx1-ubyte"
    else:
        image_path = mnist_dir / "t10k-images-idx3-ubyte"
        label_path = mnist_dir / "t10k-labels-idx1-ubyte"

    with image_path.open("rb") as images, label_path.open("rb") as labels:
        images.read(4)
        n_images = unpack(">I", images.read(4))[0]
        rows = unpack(">I", images.read(4))[0]
        cols = unpack(">I", images.read(4))[0]
        labels.read(4)
        n_labels = unpack(">I", labels.read(4))[0]
        if n_images != n_labels:
            raise ValueError(f"image/label count mismatch: {n_images} != {n_labels}")

        x = np.zeros((n_images, rows, cols), dtype=np.float32)
        y = np.zeros(n_labels, dtype=np.uint16)
        for i in range(n_images):
            x[i] = np.frombuffer(images.read(rows * cols), dtype=np.uint8).reshape(rows, cols)
            y[i] = unpack(">B", labels.read(1))[0]
    return x, y


def resolve_layout(data_root: Path, mode: str) -> Tuple[Path, Path, Path]:
    mode_root = data_root / f"l1_{mode}"
    if (data_root / "mnist").exists() and mode_root.exists():
        return data_root / "mnist", mode_root / "random", mode_root / "model"
    raise FileNotFoundError(f"Expected packaged L2exp layout under {data_root}")


def write_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def readout_names(topk_values: Iterable[int] = (3, 5)) -> List[str]:
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
        if name.startswith("top"):
            k = int(name.split("_", 1)[0][3:])
            idx = np.argsort(count_cpu)[-k:]
            local_counts = count_cpu[idx]
            local_labels = neuron_expect[idx]
            scores = np.zeros(10, dtype=np.float64)
            for c in range(10):
                vals = local_counts[local_labels == c]
                if vals.size:
                    scores[c] = vals.mean() if name.endswith("mean") else vals.sum()
            predictions[name] = int(np.argmax(scores))
            continue

        scores = np.zeros(10, dtype=np.float64)
        for c in range(10):
            idx = np.where(neuron_expect == c)[0]
            vals = count_cpu[idx]
            if vals.size == 0:
                continue
            if name == "class_mean":
                scores[c] = vals.mean()
            elif name == "class_sum":
                scores[c] = vals.sum()
            elif name == "class_mean_nonzero":
                nonzero = vals[vals > 0]
                scores[c] = nonzero.mean() if nonzero.size else 0.0
        predictions[name] = int(np.argmax(scores))
    return predictions


class SNNRuntime:
    def __init__(self, cfg: SNNLayerConfig):
        self.cfg = cfg
        self.use_gpu = bool(cfg.use_gpu and cp is not None)
        self.cp = cp
        self.xp = cp if self.use_gpu else np
        if self.use_gpu:
            self.xp.random.seed(cfg.seed)
        else:
            np.random.seed(cfg.seed)
        self.sensory_gI_max = self.xp.ones(cfg.n_e, dtype=self.xp.float32)
        self.inter_i_gE_max = self.xp.ones(cfg.n_e, dtype=self.xp.float32) * cfg.inter_i_gE_max

    def asnumpy(self, arr):
        return self.cp.asnumpy(arr) if self.use_gpu else arr

    def synchronize(self) -> None:
        if self.use_gpu:
            self.cp.cuda.Stream.null.synchronize()

    def initial_state(self):
        xp = self.xp
        cfg = self.cfg
        excitation = xp.ones(cfg.n_e, dtype=xp.float32) * cfg.v_rest_e
        inhibition = xp.ones(cfg.n_e, dtype=xp.float32) * cfg.v_rest_i
        sensory_gE = xp.zeros(cfg.n_e, dtype=xp.float32)
        sensory_gI = xp.zeros(cfg.n_e, dtype=xp.float32)
        inter_i_gE = xp.zeros(cfg.n_e, dtype=xp.float32)
        return excitation, inhibition, sensory_gE, sensory_gI, inter_i_gE

    def poisson_spike_train(self, rate_cpu: np.ndarray, interval: float):
        xp = self.xp
        cfg = self.cfg
        lam = rate_cpu.astype(np.float32) * cfg.time_step / 8.0 * interval
        p = xp.random.uniform(0.0, 1.0, (cfg.n_input, cfg.spike_rate_per_time)).astype(xp.float32)
        return xp.where(p < xp.asarray(lam)[:, None], 1.0, 0.0).astype(xp.float32)

    def input_spike(self, weight_xp, spike_xp, gain: float = 1.0):
        cfg = self.cfg
        return self.xp.matmul(
            weight_xp,
            (spike_xp * gain).reshape(cfg.n_input, 1, cfg.spike_rate_per_time),
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
        theta,
        times: int,
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
            ge_input = 0.0 if zero_input else xp.sum(sensory_ge_spike[:, :, t], axis=0) * sensory_gE_max

            sensory_gE = sensory_gE * (1 - cfg.time_step / cfg.tau_syn_E) + ge_input
            sensory_gI = sensory_gI * (1 - cfg.time_step / cfg.tau_syn_I) + i_to_e_spike_data * self.sensory_gI_max
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
                    theta = xp.where(sensory_spike[:, t] == 1.0, theta + cfg.vth_theta_inc, theta).astype(xp.float32)
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

            inter_i_gE = inter_i_gE * (1 - cfg.time_step / cfg.tau_syn_E) + e_to_i_spike_data * self.inter_i_gE_max
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

        return excitation, inhibition, sensory_gE, sensory_gI, inter_i_gE, sensory_spike, sensory_gE_max, theta, inter_i_spike

    def spike_count(self, spike):
        return self.xp.sum(spike, axis=1)

    def winner(self, count_xp) -> np.ndarray:
        count_cpu = np.asarray(self.asnumpy(count_xp)).reshape(-1)
        if count_cpu.size == 0:
            return np.array([], dtype=np.uint16)
        if self.cfg.winner_top_k > 0:
            k = min(int(self.cfg.winner_top_k), count_cpu.size)
            top_idx = np.argpartition(count_cpu, -k)[-k:]
            order = np.argsort(count_cpu[top_idx])[::-1]
            return top_idx[order].astype(np.uint16)
        return np.where(count_cpu == np.max(count_cpu))[0].astype(np.uint16)

    def normalization(self, weight_xp):
        colsums = self.xp.sum(weight_xp, axis=0)
        factors = self.cfg.weight_norm_target / colsums
        return weight_xp * factors


class FrozenL1Extractor(SNNRuntime):
    def load_weights(self, random_dir: Path, model_dir: Path):
        xp = self.xp
        weight_ei = xp.asarray(np.load(random_dir / "Sen_in_E.npy").astype(np.float32))
        weight_ie = xp.asarray(np.load(random_dir / "I_to_X.npy").astype(np.float32))
        ckpt = np.load(model_dir / "checkpoint.npz", allow_pickle=True)
        weight_input = xp.asarray(ckpt["weight"].astype(np.float32))
        sensory_gE_max = xp.asarray(ckpt["Sensory_gE_max"].astype(np.float32))
        theta = xp.asarray(ckpt["theta"].astype(np.float32))
        return weight_input, weight_ei, weight_ie, sensory_gE_max, theta

    def extract_spikes(self, image_cpu: np.ndarray, weights) -> np.ndarray:
        weight_input, weight_ei, weight_ie, sensory_gE_max_base, theta_base = weights
        interval = 2.0
        flat = image_cpu.reshape(self.cfg.n_input).astype(np.float32)
        count = self.xp.zeros(self.cfg.n_e, dtype=self.xp.float32)
        sensory_spike = None
        best_spike = None
        best_spike_total = -1.0

        for _attempt in range(max(1, int(self.cfg.max_gain_attempts))):
            excitation, inhibition, sensory_gE, sensory_gI, inter_i_gE = self.initial_state()
            current_spike = self.poisson_spike_train(flat, interval)
            sensory_ge_spike = self.input_spike(weight_input, current_spike)
            (
                _excitation,
                _inhibition,
                _sensory_gE,
                _sensory_gI,
                _inter_i_gE,
                sensory_spike,
                _sensory_gE_max,
                _theta,
                _inter_i_spike,
            ) = self.e_spike_gen(
                excitation,
                inhibition,
                sensory_gE,
                sensory_gI,
                inter_i_gE,
                sensory_ge_spike,
                weight_ei,
                weight_ie,
                self.xp.copy(sensory_gE_max_base),
                self.xp.copy(theta_base),
                self.cfg.test_steps_per_image,
                adapt=False,
            )
            count = self.spike_count(sensory_spike)
            spike_total = float(self.xp.sum(count))
            if spike_total > best_spike_total:
                best_spike = sensory_spike
                best_spike_total = spike_total
            if spike_total >= self.cfg.min_spikes:
                return sensory_spike
            interval = min(interval + self.cfg.interval_increment, self.cfg.max_input_gain)

        return best_spike if best_spike is not None else sensory_spike


class TrainableL2SNN(SNNRuntime):
    def create_random(self, random_dir: Path) -> None:
        random_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(self.cfg.seed)
        weight = (rng.random((self.cfg.n_input, self.cfg.n_e, 1), dtype=np.float64) + 0.01) * 0.3
        e_to_i = np.ones(self.cfg.n_e, dtype=np.float64) * 4.0
        i_to_x = np.ones((self.cfg.n_e, self.cfg.n_e), dtype=np.float64) * 4.0
        np.save(random_dir / "L1_to_L2.npy", weight)
        np.save(random_dir / "L2_E_to_I.npy", e_to_i)
        np.save(random_dir / "L2_I_to_X.npy", i_to_x)

    def load_random(self, random_dir: Path):
        if not (random_dir / "L1_to_L2.npy").exists():
            self.create_random(random_dir)
        xp = self.xp
        weight_input = xp.asarray(np.load(random_dir / "L1_to_L2.npy").astype(np.float32))
        weight_ei = xp.asarray(np.load(random_dir / "L2_E_to_I.npy").astype(np.float32))
        weight_ie = xp.asarray(np.load(random_dir / "L2_I_to_X.npy").astype(np.float32))
        return weight_input, weight_ei, weight_ie

    def save_checkpoint(
        self,
        path: Path,
        iteration: int,
        weight_input,
        neuron_expect: np.ndarray,
        sensory_gE_max,
        theta,
        neuron_fire_num: np.ndarray,
        class_seen: np.ndarray,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            iter=iteration,
            weight=np.asarray(self.asnumpy(weight_input), dtype=np.float32),
            neuron_expect=neuron_expect,
            Sensory_gE_max=np.asarray(self.asnumpy(sensory_gE_max), dtype=np.float32),
            theta=np.asarray(self.asnumpy(theta), dtype=np.float32),
            neuron_fire_num=neuron_fire_num,
            class_seen=class_seen,
            config=asdict(self.cfg),
        )

    def load_checkpoint_or_init(self, model_dir: Path, random_dir: Path, resume: bool):
        weight_input, weight_ei, weight_ie = self.load_random(random_dir)
        sensory_gE_max = self.xp.ones(self.cfg.n_e, dtype=self.xp.float32)
        theta = self.xp.ones(self.cfg.n_e, dtype=self.xp.float32) * self.cfg.initial_theta
        neuron_expect = np.zeros(self.cfg.n_e, dtype=np.uint16)
        neuron_fire_num = np.zeros((self.cfg.n_e, 10), dtype=np.uint32)
        class_seen = np.zeros(10, dtype=np.uint32)
        start_iter = 0
        checkpoint = model_dir / "checkpoint.npz"
        if resume and checkpoint.exists():
            ckpt = np.load(checkpoint, allow_pickle=True)
            start_iter = int(ckpt["iter"]) + 1
            weight_input = self.xp.asarray(ckpt["weight"].astype(np.float32))
            neuron_expect = ckpt["neuron_expect"].astype(np.uint16)
            sensory_gE_max = self.xp.asarray(ckpt["Sensory_gE_max"].astype(np.float32))
            theta = self.xp.asarray(ckpt["theta"].astype(np.float32))
            if "neuron_fire_num" in ckpt:
                neuron_fire_num = ckpt["neuron_fire_num"].astype(np.uint32)
            if "class_seen" in ckpt:
                class_seen = ckpt["class_seen"].astype(np.uint32)
            print(f"[RESUME L2 {self.cfg.canonical_mode}] from iter {start_iter}")
        return start_iter, weight_input, weight_ei, weight_ie, sensory_gE_max, theta, neuron_expect, neuron_fire_num, class_seen

    def _find_nearest_over(self, array: np.ndarray, value: int) -> Optional[int]:
        over_array = array[np.where(array >= value)]
        if over_array.size == 0:
            return None
        return int(over_array[(np.abs(over_array - value)).argmin()])

    def _find_nearest_under(self, array: np.ndarray, value: int, used: np.ndarray) -> Optional[int]:
        under_array = array[np.where(array <= value)]
        if under_array.size == 0:
            return None
        under_value = int(under_array[(np.abs(under_array - value)).argmin()])
        idx = np.where(array == under_value)
        if used[idx].size and used[idx][0] == 0:
            used[idx] = 1
        elif used[idx].size and used[idx][0] == 1:
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
                used = np.zeros(len(pre_for_input))
                for post_t in post_time_arr[:, 1]:
                    under = self._find_nearest_under(pre_for_input[:, 2], int(post_t), used)
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
                        idx = (int(input_idx), int(neuron), 0) if weight.ndim == 3 else (int(input_idx), int(neuron))
                        weight[idx] = _update_weight_single(float(weight[idx]), float(w_del))

        return self.xp.asarray(weight) if self.use_gpu else weight

    def run_layer_once(
        self,
        l1_spike_xp,
        weight_input,
        weight_ei,
        weight_ie,
        sensory_gE_max,
        theta,
        *,
        train: bool,
    ):
        steps = self.cfg.train_steps_per_image if train else self.cfg.test_steps_per_image
        gain = self.cfg.input_gain
        count = self.xp.zeros(self.cfg.n_e, dtype=self.xp.float32)
        result = None
        best_result = None
        best_l2_ge_spike = None
        best_count = None
        best_spike_total = -1.0
        for _attempt in range(max(1, int(self.cfg.max_gain_attempts))):
            excitation, inhibition, sensory_gE, sensory_gI, inter_i_gE = self.initial_state()
            l2_ge_spike = self.input_spike(weight_input, l1_spike_xp, gain=gain)
            result = self.e_spike_gen(
                excitation,
                inhibition,
                sensory_gE,
                sensory_gI,
                inter_i_gE,
                l2_ge_spike,
                weight_ei,
                weight_ie,
                sensory_gE_max,
                theta,
                steps,
                adapt=train,
            )
            count = self.spike_count(result[5])
            spike_total = float(self.xp.sum(count))
            if spike_total > best_spike_total:
                best_result = result
                best_l2_ge_spike = l2_ge_spike
                best_count = count
                best_spike_total = spike_total
            if spike_total >= self.cfg.min_spikes:
                return result, l2_ge_spike, count
            gain = min(gain + self.cfg.interval_increment, self.cfg.max_input_gain)
        return best_result, best_l2_ge_spike, best_count

    def train(
        self,
        *,
        l1_extractor: FrozenL1Extractor,
        l1_weights,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        random_dir: Path,
        model_dir: Path,
        train_iters: int,
        resume: bool,
    ) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (
            start_iter,
            weight_input,
            weight_ei,
            weight_ie,
            sensory_gE_max,
            theta,
            neuron_expect,
            neuron_fire_num,
            class_seen,
        ) = self.load_checkpoint_or_init(model_dir, random_dir, resume=resume)

        log_path = model_dir / "train_log.txt"
        if start_iter == 0:
            log_path.write_text("", encoding="utf-8")

        performance_count = 0.0
        spike_sum = 0.0
        noi_sum = 0.0
        wta_sum = 0.0
        single_wta = 0
        window_count = 0
        start = time.time()

        for i in range(start_iter, train_iters):
            label = int(train_labels[i % len(train_labels)])
            l1_spike = l1_extractor.extract_spikes(train_images[i % len(train_images)], l1_weights)
            weight_input = self.normalization(weight_input)
            (result, l2_ge_spike, count) = self.run_layer_once(
                l1_spike,
                weight_input,
                weight_ei,
                weight_ie,
                sensory_gE_max,
                theta,
                train=True,
            )
            (
                _excitation,
                _inhibition,
                _sensory_gE,
                _sensory_gI,
                _inter_i_gE,
                l2_spike,
                sensory_gE_max,
                theta,
                inter_i_spike,
            ) = result

            count_cpu = np.asarray(self.asnumpy(count), dtype=np.float32)
            pred = predict_readouts(count_cpu, neuron_expect, ["winner"])["winner"]
            winner_idx = self.winner(count)
            if pred == label:
                performance_count += 1.0

            neuron_fire_num[winner_idx, label] += 1
            class_seen[label] += 1
            pre_spike = self.xp.where(l2_ge_spike != 0.0, 1.0, 0.0)
            weight_input = self.stdp_cpu(pre_spike, l2_spike, winner_idx, weight_input)

            spike_sum += float(self.xp.sum(count)) + float(self.xp.sum(inter_i_spike))
            noi_sum += int(self.xp.count_nonzero(count))
            wta_size = int(np.asarray(winner_idx).size)
            wta_sum += wta_size
            single_wta += 1 if wta_size == 1 else 0
            window_count += 1

            if i % self.cfg.save_interval == self.cfg.save_interval - 1 or i == train_iters - 1:
                neuron_expect = np.argmax((neuron_fire_num / np.maximum(class_seen, 1)), axis=1).astype(np.uint16)
                elapsed = time.time() - start
                acc = performance_count / max(window_count, 1) * 100.0
                msg = (
                    f"[L2 TRAIN {self.cfg.canonical_mode} iter {i + 1}/{train_iters}] accuracy={acc:.2f}% "
                    f"avg_spikes={spike_sum / max(window_count, 1):.2f} "
                    f"avg_noi={noi_sum / max(window_count, 1):.2f} "
                    f"avg_wta={wta_sum / max(window_count, 1):.2f} "
                    f"single_wta={single_wta / max(window_count, 1) * 100.0:.2f}% "
                    f"elapsed={elapsed / 60.0:.1f} min"
                )
                print(msg)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                self.save_checkpoint(
                    model_dir / "checkpoint.npz",
                    i,
                    weight_input,
                    neuron_expect,
                    sensory_gE_max,
                    theta,
                    neuron_fire_num,
                    class_seen,
                )
                self.save_checkpoint(
                    model_dir / f"checkpoint_{i + 1:08d}.npz",
                    i,
                    weight_input,
                    neuron_expect,
                    sensory_gE_max,
                    theta,
                    neuron_fire_num,
                    class_seen,
                )
                performance_count = 0.0
                spike_sum = 0.0
                noi_sum = 0.0
                wta_sum = 0.0
                single_wta = 0
                window_count = 0

        np.save(model_dir / "weight.npy", np.asarray(self.asnumpy(weight_input), dtype=np.float32))
        np.save(model_dir / "neuron_expect.npy", neuron_expect)
        np.save(model_dir / "Sensory_gE_max.npy", np.asarray(self.asnumpy(sensory_gE_max), dtype=np.float32))
        np.save(model_dir / "theta.npy", np.asarray(self.asnumpy(theta), dtype=np.float32))
        print(f"[L2 TRAIN DONE] mode={self.cfg.canonical_mode} out={model_dir}")

    def test(
        self,
        *,
        l1_extractor: FrozenL1Extractor,
        l1_weights,
        test_images: np.ndarray,
        test_labels: np.ndarray,
        random_dir: Path,
        model_dir: Path,
        output_dir: Path,
        test_limit: int,
        force: bool,
    ) -> Dict[str, Dict[str, float]]:
        summary_path = output_dir / "summary.json"
        if summary_path.exists() and not force:
            print(f"[SKIP L2 TEST] mode={self.cfg.canonical_mode} already complete: {summary_path}")
            return json.loads(summary_path.read_text(encoding="utf-8"))

        output_dir.mkdir(parents=True, exist_ok=True)
        weight_input, weight_ei, weight_ie = self.load_random(random_dir)
        ckpt = np.load(model_dir / "checkpoint.npz", allow_pickle=True)
        weight_input = self.xp.asarray(ckpt["weight"].astype(np.float32))
        neuron_expect = ckpt["neuron_expect"].astype(np.uint16)
        sensory_gE_max = self.xp.asarray(ckpt["Sensory_gE_max"].astype(np.float32))
        theta = self.xp.asarray(ckpt["theta"].astype(np.float32))

        readouts = readout_names()
        correct = {name: 0 for name in readouts}
        rows: List[Dict[str, object]] = []
        spike_sum = 0.0
        noi_sum = 0.0
        wta_sum = 0.0
        single_wta = 0
        start = time.time()

        for i in range(test_limit):
            label = int(test_labels[i % len(test_labels)])
            l1_spike = l1_extractor.extract_spikes(test_images[i % len(test_images)], l1_weights)
            result, _l2_ge_spike, count = self.run_layer_once(
                l1_spike,
                weight_input,
                weight_ei,
                weight_ie,
                self.xp.copy(sensory_gE_max),
                self.xp.copy(theta),
                train=False,
            )
            l2_spike = result[5]
            inter_i_spike = result[8]
            count_cpu = np.asarray(self.asnumpy(count), dtype=np.float32)
            preds = predict_readouts(count_cpu, neuron_expect, readouts)
            for name, pred in preds.items():
                if pred == label:
                    correct[name] += 1
            winner_idx = self.winner(count)
            spike_sum += float(self.xp.sum(count)) + float(self.xp.sum(inter_i_spike))
            noi = int(self.xp.count_nonzero(count))
            wta_size = int(np.asarray(winner_idx).size)
            noi_sum += noi
            wta_sum += wta_size
            single_wta += 1 if wta_size == 1 else 0
            rows.append(
                {
                    "sample": i,
                    "label": label,
                    "total_spikes": float(self.xp.sum(count)) + float(self.xp.sum(inter_i_spike)),
                    "l2_exc_spikes": float(self.xp.sum(count)),
                    "noi": noi,
                    "wta_size": wta_size,
                    **{f"pred_{name}": pred for name, pred in preds.items()},
                }
            )
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                best = max(correct, key=lambda name: correct[name])
                print(f"[L2 TEST {self.cfg.canonical_mode} {i + 1}/{test_limit}] best={best} acc={correct[best] / (i + 1) * 100.0:.2f}% elapsed={elapsed/60.0:.1f} min")

        elapsed = time.time() - start
        common = {
            "samples": test_limit,
            "elapsed_sec": elapsed,
            "avg_spikes": spike_sum / max(test_limit, 1),
            "avg_noi": noi_sum / max(test_limit, 1),
            "avg_wta_size": wta_sum / max(test_limit, 1),
            "single_wta_ratio": single_wta / max(test_limit, 1) * 100.0,
        }
        summary = {
            name: {"accuracy": correct[name] / max(test_limit, 1) * 100.0, "correct": correct[name], **common}
            for name in readouts
        }
        write_json(summary_path, summary)
        with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
            for name, metrics in sorted(summary.items(), key=lambda item: item[1]["accuracy"], reverse=True):
                f.write(f"{name}: accuracy={metrics['accuracy']:.4f}% correct={metrics['correct']}/{test_limit}\n")
            f.write(json.dumps(common, indent=2) + "\n")
        with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["sample"])
            writer.writeheader()
            writer.writerows(rows)
        best = max(summary, key=lambda name: summary[name]["accuracy"])
        print(f"[L2 TEST DONE] mode={self.cfg.canonical_mode} best={best} acc={summary[best]['accuracy']:.2f}%")
        return summary


def checkpoint_complete(model_dir: Path, train_iters: int) -> bool:
    ckpt = model_dir / "checkpoint.npz"
    if not ckpt.exists():
        return False
    data = np.load(ckpt, allow_pickle=True)
    return int(data["iter"]) >= train_iters - 1


def run_mode(
    *,
    mode: str,
    data_root: Path,
    results_root: Path,
    run_name: str,
    train_iters: int,
    test_limit: int,
    n_l2: int,
    seed: int,
    save_interval: int,
    force: bool,
    use_gpu: bool,
) -> Dict[str, Dict[str, float]]:
    mnist_dir, l1_random_dir, l1_model_dir = resolve_layout(data_root, mode)
    run_dir = results_root / "l2_snn" / run_name / mode
    l2_random_dir = run_dir / "random"
    l2_model_dir = run_dir / "model"
    l2_test_dir = run_dir / "test"

    train_images, train_labels = load_idx_dataset(mnist_dir, "training")
    test_images, test_labels = load_idx_dataset(mnist_dir, "testing")

    l1_fast_params = {}
    l2_fast_params = {}
    if mode == "vth":
        l1_fast_params = {
            "min_spikes": 1.0,
            "interval_increment": 1.0,
            "max_gain_attempts": 8,
            "max_input_gain": 10.0,
        }
        l2_fast_params = {
            "min_spikes": 1.0,
            "input_gain": 2.5,
            "interval_increment": 1.0,
            "initial_theta": 0.015,
            "vth_theta_inc": 0.000025,
            "max_gain_attempts": 8,
            "max_input_gain": 10.0,
            "winner_top_k": 1,
        }

    l1_cfg = SNNLayerConfig(
        mode=mode,
        n_input=784,
        n_e=625,
        use_gpu=use_gpu,
        seed=seed,
        **l1_fast_params,
    )
    l2_cfg_params = {
        "mode": mode,
        "n_input": 625,
        "n_e": n_l2,
        "use_gpu": use_gpu,
        "seed": seed,
        "save_interval": save_interval,
        "initial_interval": 1.0,
        "interval_increment": 0.5,
    }
    l2_cfg_params.update(l2_fast_params)
    l2_cfg = SNNLayerConfig(**l2_cfg_params)
    l1 = FrozenL1Extractor(l1_cfg)
    l1_weights = l1.load_weights(l1_random_dir, l1_model_dir)
    l2 = TrainableL2SNN(l2_cfg)

    if not force and checkpoint_complete(l2_model_dir, train_iters):
        print(f"[SKIP L2 TRAIN] mode={mode} already complete: {l2_model_dir}")
    else:
        print(f"[L2 SNN TRAIN] mode={mode} n_l2={n_l2} train_iters={train_iters} out={l2_model_dir}")
        l2.train(
            l1_extractor=l1,
            l1_weights=l1_weights,
            train_images=train_images,
            train_labels=train_labels,
            random_dir=l2_random_dir,
            model_dir=l2_model_dir,
            train_iters=train_iters,
            resume=not force,
        )

    print(f"[L2 SNN TEST] mode={mode} test_limit={test_limit} out={l2_test_dir}")
    summary = l2.test(
        l1_extractor=l1,
        l1_weights=l1_weights,
        test_images=test_images,
        test_labels=test_labels,
        random_dir=l2_random_dir,
        model_dir=l2_model_dir,
        output_dir=l2_test_dir,
        test_limit=test_limit,
        force=force,
    )
    write_json(
        run_dir / "metadata.json",
        {
            "mode": mode,
            "mnist_dir": str(mnist_dir),
            "l1_random_dir": str(l1_random_dir),
            "l1_model_dir": str(l1_model_dir),
            "l2_random_dir": str(l2_random_dir),
            "l2_model_dir": str(l2_model_dir),
            "l1_config": asdict(l1_cfg),
            "l2_config": asdict(l2_cfg),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="True two-layer SNN: frozen trained 625-neuron L1 feeds a trainable L2 SNN layer.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    parser.add_argument("--run-name", default="server_run_l2_snn_001")
    parser.add_argument("--modes", nargs="+", choices=["sen", "vth"], default=["sen", "vth"])
    parser.add_argument("--n-l2", type=int, default=625)
    parser.add_argument("--train-iters", type=int, default=60000)
    parser.add_argument("--test-limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true", help="Force NumPy CPU backend.")
    parser.add_argument("--force", action="store_true", help="Rebuild L2 random weights and rerun train/test.")
    args = parser.parse_args()

    print(f"[L2 TRUE SNN] run_name={args.run_name}")
    print(f"[L2 TRUE SNN] data_root={args.data_root}")
    print(f"[L2 TRUE SNN] results_root={args.results_root}")
    print("[L2 TRUE SNN] architecture=frozen trained L1 SNN spike train -> trainable L2 SNN with STDP/readout voting")

    all_summary = {}
    for mode in args.modes:
        all_summary[mode] = run_mode(
            mode=mode,
            data_root=args.data_root,
            results_root=args.results_root,
            run_name=args.run_name,
            train_iters=args.train_iters,
            test_limit=args.test_limit,
            n_l2=args.n_l2,
            seed=args.seed,
            save_interval=args.save_interval,
            force=args.force,
            use_gpu=not args.cpu,
        )

    write_json(args.results_root / "l2_snn" / args.run_name / "summary.json", all_summary)
    print("[L2 TRUE SNN DONE]")


if __name__ == "__main__":
    main()
