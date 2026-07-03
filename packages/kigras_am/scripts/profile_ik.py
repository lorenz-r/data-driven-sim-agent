"""Profile IK solver: fd-init + L-BFGS scaling characteristics."""

import time
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kigras.kinematics import CTRAKinematics


def _time(name, fn, n_warmup=0, n_trial=2):
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(n_trial):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"  {name}: {avg * 1000:.1f} ms")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Device: {device}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})"
    )
    print(f"PyTorch: {torch.__version__}")

    kin = CTRAKinematics(dt=0.5)
    init = torch.tensor([0.0, 0.0, 1.0, 0.0, 5.0]).unsqueeze(0)

    # ---------- forward pass micro ----------
    print("\n=== Forward kinematics ===")
    for T in [10, 30, 100]:
        acts = torch.randn(1, T, 2)
        _time(f"T={T}", lambda a=acts: kin.forward(init, a))
        _time(
            f"T={T} B=16",
            lambda a=acts: kin.forward(init.expand(16, -1), a.expand(16, -1, -1)),
        )

    # ---------- trajectory length scaling ----------
    print("\n=== Trajectory length scaling (max_iter=5) ===")
    for T in [10, 30, 60, 110]:
        traj = kin.forward(init, torch.randn(1, T, 2))
        s = CTRAKinematics(dt=0.5, max_iter=5)
        _time(f"T={T}", lambda t=traj, solver=s: solver.solve(t, quantize=False))

    # ---------- iter scaling ----------
    print("\n=== Iteration scaling (T=30) ===")
    traj_30 = kin.forward(init, torch.randn(1, 30, 2))
    for iters in [0, 1, 5, 15]:
        s = CTRAKinematics(dt=0.5, max_iter=iters)
        _time(
            f"iter={iters}", lambda t=traj_30, solver=s: solver.solve(t, quantize=False)
        )

    # ---------- batch scaling ----------
    print("\n=== Batch scaling (T=30, iter=5) ===")
    traj_b1 = kin.forward(init, torch.randn(1, 30, 2))
    for B in [1, 8, 32, 64]:
        batch = traj_b1.expand(B, -1, -1).contiguous()
        s = CTRAKinematics(dt=0.5, max_iter=5)
        _time(f"B={B}", lambda t=batch, solver=s: solver.solve(t, quantize=False))

    # ---------- per-step cost breakdown ----------
    print("\n=== Per-step cost breakdown (T=30, iter=5) ===")
    obs = traj_30
    T = obs.shape[-2] - 1
    cur_state = obs[..., 0, :]
    wa_analytic = kin.solve(obs, quantize=False)
    p = torch.nn.Parameter(wa_analytic.detach().clone())

    def closure():
        p.grad = None
        pred = kin.forward(cur_state, p)
        pred_norm, target_norm = kin._normalize(pred, obs)
        loss = F.mse_loss(pred_norm, target_norm, reduction="sum")
        loss.backward()
        return loss

    opt = torch.optim.LBFGS([p], lr=1.0, max_iter=5, line_search_fn="strong_wolfe")
    _time("1 L-BFGS step (5 iters)", lambda: opt.step(closure), n_warmup=1)

    t0 = time.perf_counter()
    for _ in range(100):
        kin.forward(cur_state, p)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fwd_time = (time.perf_counter() - t0) / 100
    print(f"  Single forward (T=29): {fwd_time * 1000:.3f} ms")


if __name__ == "__main__":
    main()
