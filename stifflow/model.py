"""Reuse ET-Flow Lightning/backbone; replace conditional path and inference source."""

import time

import numpy as np
import torch
from etflow.models import BaseFlow
from etflow.models.loss import batchwise_mse_loss
from torch_cluster import radius_graph
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

from .source import prepare_source
from .targets import draw_target, path_module
from .training import advance_cursor, allocator_budget, gpu_memory_check


def require_finite(tensors, what):
    """Keep all checks, with one host decision instead of one per parameter."""
    flags = [torch.isfinite(tensor).all() for tensor in tensors]
    valid = torch.stack(flags).all() if flags else torch.tensor(True)
    if torch.distributed.is_initialized():
        if torch.distributed.get_backend() == "nccl":
            valid = valid.to(torch.cuda.current_device())
        valid = valid.to(torch.int32)
        torch.distributed.all_reduce(valid, op=torch.distributed.ReduceOp.MIN)
    if not valid:
        raise FloatingPointError(f"Non-finite {what}")


def graph_data(graph):
    """Whitelist graph features: never include target coordinates or labels."""
    return Data(
        atomic_numbers=torch.as_tensor(graph["atomic_numbers"], dtype=torch.long),
        edge_index=torch.as_tensor(graph["bonds"], dtype=torch.long)
        .reshape(-1, 2)
        .T.contiguous(),
        node_attr=torch.as_tensor(graph["node_attr"], dtype=torch.float32),
        edge_attr=torch.as_tensor(graph["bond_attr"], dtype=torch.float32),
        num_nodes=len(graph["atomic_numbers"]),
    )


class StifFlow(BaseFlow):
    def __init__(
        self,
        method="precision",
        seed=42,
        source_config=None,
        target_rng_mode="stream",
        path_clock=None,
        stiffness_start=None,
        locality_report_sha256=None,
        **kwargs,
    ):
        path_module(method)
        from .research_clock import PathClock
        from .research_paths import DELAYED_METHODS, validate_stiffness_start

        self.stiffness_start = validate_stiffness_start(method, stiffness_start)
        if locality_report_sha256 is not None and (
            method not in DELAYED_METHODS
            or not isinstance(locality_report_sha256, str)
            or len(locality_report_sha256) != 64
            or any(c not in "0123456789abcdef" for c in locality_report_sha256)
        ):
            raise ValueError("Invalid delayed-path locality report hash")
        if (method == "precision_clock") != (path_clock is not None):
            raise ValueError("Only precision_clock requires a frozen path_clock")
        self.path_clock = PathClock(path_clock) if path_clock is not None else None
        if target_rng_mode not in ("stream", "global_draw"):
            raise ValueError("Unknown target RNG mode")
        fixed = {
            "so3_equivariant": True,
            "parity_switch": None,
            "sigma": 0.0,
            "sample_time_dist": "uniform",
            "node_attr_dim": 10,
            "edge_attr_dim": 3,
        }
        for key, value in fixed.items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError(f"StifFlow requires {key}={value!r}")
        kwargs.update(fixed)
        super().__init__(**kwargs)
        self.method, self.seed = method, seed
        self.source_config = source_config or {"sigma0": 1.0, "rho": 1.0}
        self.training_signature = None
        self.target_rng = np.random.default_rng(seed)
        self.target_rng_mode = target_rng_mode
        self.consumed_samples = 0
        self.gpu_memory = None
        self.save_hyperparameters(
            {
                "method": method,
                "seed": seed,
                "source_config": self.source_config,
                "target_rng_mode": target_rng_mode,
                **kwargs,
            }
        )
        if path_clock is not None:
            self.save_hyperparameters({"path_clock": path_clock})
        if stiffness_start is not None:
            self.save_hyperparameters(
                {
                    "stiffness_start": self.stiffness_start,
                    "locality_report_sha256": locality_report_sha256,
                }
            )

    def forward(self, z, t, pos, bond_index, edge_attr, node_attr, batch):
        pos = pos - scatter(pos, batch, dim=0, reduce="mean")[batch]
        max_neighbors = int(torch.bincount(batch).max().item())
        radius = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=max_neighbors
        )
        edges, inverse = torch.unique(
            torch.cat([bond_index, radius], dim=1), dim=1, return_inverse=True
        )
        attrs = pos.new_zeros((edges.shape[1], 3))
        attrs.index_add_(0, inverse[: bond_index.shape[1]], edge_attr.to(pos.dtype))
        return self.network(
            z=z,
            t=t[batch],
            pos=pos,
            edge_index=edges,
            edge_attr=attrs,
            node_attr=node_attr,
            batch=batch,
        )

    def generic_step(self, payload, batch_idx, stage):
        graph, supervision = payload
        if stage == "train":
            rank = getattr(self._trainer, "global_rank", 0)
            world = getattr(self._trainer, "world_size", 1)
            if world > 1 and self.target_rng_mode != "global_draw":
                raise ValueError("DDP requires global-draw supervision")
            self.consumed_samples = advance_cursor(
                self.consumed_samples, supervision, rank, world
            )
        positions, targets, times = [], [], []
        rng = (
            self.target_rng
            if stage == "train"
            else np.random.default_rng([self.seed, batch_idx, 1])
        )
        with torch.profiler.record_function("stifflow/float64_target"):
            for record in supervision:
                if self.target_rng_mode == "global_draw":
                    rng = np.random.default_rng(
                        [self.seed, record["draw_id"], 3 if stage == "train" else 4]
                    )
                xt, ut, t = draw_target(
                    record["x1"],
                    record["source"],
                    record["label"],
                    self.method,
                    rng,
                    clock=self.path_clock,
                    stiffness_start=self.stiffness_start,
                )
                positions.append(xt)
                targets.append(ut)
                times.append(t)
        with torch.profiler.record_function("stifflow/target_transfer"):
            pos = torch.as_tensor(
                np.concatenate(positions), dtype=torch.float32, device=self.device
            )
            target = torch.as_tensor(
                np.concatenate(targets), dtype=torch.float32, device=self.device
            )
            t = torch.as_tensor(times, dtype=torch.float32, device=self.device).reshape(
                -1, 1
            )
        with torch.profiler.record_function("stifflow/network_forward"):
            prediction = self(
                z=graph.atomic_numbers,
                t=t,
                pos=pos,
                bond_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                node_attr=graph.node_attr,
                batch=graph.batch,
            )
        loss = batchwise_mse_loss(prediction.float(), target, graph.batch, reduce="sum")
        require_finite([loss], "Euclidean CFM loss")
        if self.gpu_memory and stage == "val":
            gpu_memory_check(self.gpu_memory)
        self.log_helper(f"{stage}/loss", loss, len(supervision))
        return loss

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        with torch.profiler.record_function("stifflow/graph_transfer"):
            return batch[0].to(device, non_blocking=True), batch[1]

    def sample(self, *args, **kwargs):
        raise RuntimeError(
            "Use stifflow.model.generate for the graph-only StifFlow source"
        )

    def on_after_backward(self):
        if self.gpu_memory:
            gpu_memory_check(self.gpu_memory)
        with torch.profiler.record_function("stifflow/gradient_checks"):
            require_finite(
                (p.grad for p in self.parameters() if p.grad is not None), "gradient"
            )

    def on_before_zero_grad(self, optimizer):
        with torch.profiler.record_function("stifflow/parameter_checks"):
            require_finite(self.parameters(), "parameter")

    def on_train_start(self):
        if self.gpu_memory:
            free, total = torch.cuda.mem_get_info()
            fraction = allocator_budget(
                self.gpu_memory, total, total - free - torch.cuda.memory_reserved()
            )
            torch.cuda.set_per_process_memory_fraction(fraction)

    def on_before_backward(self, loss):
        if self.gpu_memory:
            gpu_memory_check(self.gpu_memory)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["training_signature"] = self.training_signature
        checkpoint["stifflow_rng"] = self.target_rng.bit_generator.state
        checkpoint["torch_rng"] = torch.get_rng_state()
        checkpoint["gradient_queue"] = self.gradnorm_queue.items
        checkpoint["consumed_samples"] = self.consumed_samples
        if torch.distributed.is_initialized():
            state = {
                key: checkpoint[key]
                for key in (
                    "stifflow_rng",
                    "torch_rng",
                    "gradient_queue",
                    "consumed_samples",
                )
            }
            state["cuda_rng_local"] = (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            )
            states = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(states, state)
            checkpoint["distributed_states"] = {
                "world_size": len(states),
                "ranks": states,
            }
        elif torch.cuda.is_available():
            checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()

    def on_load_checkpoint(self, checkpoint):
        if "distributed_states" in checkpoint:
            saved = checkpoint["distributed_states"]
            world = (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else 1
            )
            if saved["world_size"] != world:
                raise ValueError("Cannot change world size when resuming DDP")
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            checkpoint = {**checkpoint, **saved["ranks"][rank]}
        self.target_rng.bit_generator.state = checkpoint["stifflow_rng"]
        self.gradnorm_queue.items = checkpoint["gradient_queue"]
        self.consumed_samples = checkpoint.get(
            "consumed_samples", self.consumed_samples
        )
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if checkpoint.get("cuda_rng_local") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_local"].cpu())
        elif "cuda_rng" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])


@torch.no_grad()
def generate(model, graph, source_config=None, sampler_config=None):
    """Core/Alternative A eq:inference. Graph-only, left Euler from 0 to 1."""
    config = {"samples": 1, "steps": 50, "seed": 42, "batch_size": 32}
    if sampler_config:
        if set(sampler_config) - config.keys():
            raise ValueError("Unknown sampler options")
        config.update(sampler_config)
    if any(
        not isinstance(config[k], int) or config[k] < 1
        for k in ("samples", "steps", "batch_size")
    ):
        raise ValueError("samples, steps and batch_size must be positive integers")
    if source_config is not None and source_config != model.source_config:
        raise ValueError("Inference source differs from training source")
    source_started = time.perf_counter()
    source = prepare_source(graph, model.source_config)
    source_seconds = time.perf_counter() - source_started
    rng = np.random.default_rng(config["seed"])
    data = graph_data(graph)
    device = next(model.parameters()).device
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    results = []
    calls = 0
    for offset in range(0, config["samples"], config["batch_size"]):
        size = min(config["batch_size"], config["samples"] - offset)
        graphs = Batch.from_data_list([data] * size).to(device)
        initial = (
            rng.standard_normal((size, source.S.shape[0])) @ source.S.T @ source.Q.T
        ).reshape(-1, 3)
        x = torch.as_tensor(initial, dtype=torch.float32, device=device)
        for step in range(config["steps"]):
            t = x.new_full((size, 1), step / config["steps"])
            x = (
                x
                + model(
                    z=graphs.atomic_numbers,
                    t=t,
                    pos=x,
                    bond_index=graphs.edge_index,
                    edge_attr=graphs.edge_attr,
                    node_attr=graphs.node_attr,
                    batch=graphs.batch,
                )
                / config["steps"]
            )
            calls += 1
        if not torch.isfinite(x).all():
            raise FloatingPointError("Non-finite generated positions")
        results.append(x.reshape(size, -1, 3).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return np.concatenate(results), {
        "source_preparation_seconds": source_seconds,
        "forward_calls": calls,
        "nfe_per_conformer": config["steps"],
        "seconds": time.perf_counter() - start,
        "peak_gpu_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "sampler": config,
        "postprocessing": "none",
    }
