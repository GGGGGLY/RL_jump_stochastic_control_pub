"""
Resume / checkpoint utilities for merton_pinn_0201_optimizer*.py.

Goals
-----
- Provide a *compatible* resume mechanism:
  - If a training-state file exists (optimizer/scheduler/global_step), restore it exactly.
  - If not, fall back to best-effort inference from ckpt filenames / logs and align schedules.

This module is intentionally self-contained and only touches resume/checkpoint concerns.
"""

from __future__ import annotations

import glob
import math
import os
import re
from typing import Any, Dict, Optional, Tuple

import torch


TRAIN_STATE_PREFIX = "train_state_it"


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def parse_start_it_from_path(path: Optional[str]) -> Optional[int]:
    """
    Best-effort parse the *next* training iteration index (start_it) from a checkpoint path string.

    Supported patterns (basename or full path):
      - *_it{N}.pth   (periodic ckpt naming: actor_it{N}.pth / critic_it{N}.pth), N is 1-based step
                      -> start_it = N
      - *itr{N}*      (eval/top-k ckpt naming)
                      -> start_it = N + 1   (itrN means "after finishing iteration N")

    Returns:
      start_it (0-based loop index to continue training from), or None if not found.
    """
    if not path:
        return None
    s = str(path)
    base = os.path.basename(s)

    # Prefer explicit "_it123" (periodic ckpts)
    m_it = re.search(r"(?:^|[_-])it(\d+)(?:\D|$)", base) or re.search(r"(?:^|[_-])it(\d+)(?:\D|$)", s)
    if m_it is not None:
        return int(m_it.group(1))

    # Fallback: "itr123" tags (saved after finishing itr)
    m_itr = re.search(r"itr(\d+)", base) or re.search(r"itr(\d+)", s)
    if m_itr is not None:
        return _safe_int(m_itr.group(1), default=0) + 1

    return None


def infer_start_it(actor_path: Optional[str], critic_path: Optional[str]) -> int:
    """
    Infer start_it from actor/critic ckpt paths; return 0 if unknown.

    Priority:
    1) parse from filename (it/itr tags)
    2) if missing (e.g. actor.pth), infer from ckpt directory artifacts:
       - last row of loss_history.csv (iter column) => start_it = last_iter + 1
       - max actor_it{N}.pth under every_*_ckpts
    """
    cands = [
        parse_start_it_from_path(actor_path),
        parse_start_it_from_path(critic_path),
    ]
    cands = [c for c in cands if isinstance(c, int) and c >= 0]
    if cands:
        return int(max(cands))

    # Fallback: infer from directory history when filenames don't carry an iteration tag.
    base_dir = None
    try:
        p0 = actor_path or critic_path
        if p0:
            base_dir = os.path.dirname(os.path.abspath(str(p0)))
    except Exception:
        base_dir = None

    inferred = []
    if base_dir and os.path.isdir(base_dir):
        # a) loss_history.csv last iter
        try:
            loss_csv = os.path.join(base_dir, "loss_history.csv")
            if os.path.isfile(loss_csv):
                last = None
                with open(loss_csv, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("iter,"):
                            continue
                        last = line
                if last is not None:
                    it_val = int(str(last).split(",")[0])
                    inferred.append(int(it_val) + 1)
        except Exception:
            pass

        # b) scan periodic ckpt names under every_*_ckpts
        try:
            pat = os.path.join(base_dir, "every_*_ckpts", "actor_it*.pth")
            best = None
            for fp in glob.glob(pat):
                m = re.search(r"(?:^|[_-])it(\d+)(?:\D|$)", os.path.basename(fp))
                if m is None:
                    continue
                v = int(m.group(1))
                best = v if best is None else max(best, v)
            if best is not None:
                inferred.append(int(best))
        except Exception:
            pass

    return int(max(inferred)) if inferred else 0


def estimate_global_actor_update_step(
    start_it: int,
    N: int,
    K_actor: int,
) -> int:
    """
    Schedules (warmup/freeze/mix_ramp) use `_global_actor_update_step` which increments
    once per actor chunk update (roughly ceil(N / K_actor) times per outer iteration).
    We approximate the current actor_update_step from start_it to avoid re-warmup on resume.
    """
    start_it = int(max(0, int(start_it)))
    N = int(max(1, int(N)))
    K_actor = int(max(1, int(K_actor)))
    chunks_per_iter = int(math.ceil(N / float(K_actor)))
    chunks_per_iter = max(1, chunks_per_iter)
    return int(start_it * chunks_per_iter)


def _ckpt_dir_from_paths(actor_path: Optional[str], critic_path: Optional[str]) -> Optional[str]:
    p0 = actor_path or critic_path
    if not p0:
        return None
    try:
        d = os.path.dirname(os.path.abspath(str(p0)))
        return d if os.path.isdir(d) else None
    except Exception:
        return None


def training_state_path_for_start_it(ckpt_dir: str, start_it: int) -> str:
    """Canonical training-state filename for a given start_it (= periodic step)."""
    return os.path.join(ckpt_dir, f"{TRAIN_STATE_PREFIX}{int(start_it)}.pt")


def _find_best_training_state_file(ckpt_dir: str, start_it: int) -> Optional[str]:
    """
    Find a training-state file in ckpt_dir.
    Preference:
      1) exact match train_state_it{start_it}.pt
      2) the largest train_state_it{n}.pt with n <= start_it
      3) otherwise, the largest available
    """
    start_it = int(max(0, int(start_it)))
    exact = training_state_path_for_start_it(ckpt_dir, start_it)
    if os.path.isfile(exact):
        return exact

    files = glob.glob(os.path.join(ckpt_dir, f"{TRAIN_STATE_PREFIX}*.pt"))
    parsed = []
    for fp in files:
        m = re.search(rf"{re.escape(TRAIN_STATE_PREFIX)}(\d+)\.pt$", os.path.basename(fp))
        if m is None:
            continue
        parsed.append((int(m.group(1)), fp))
    if not parsed:
        return None

    le = [t for t in parsed if t[0] <= start_it]
    if le:
        return sorted(le, key=lambda t: t[0])[-1][1]
    return sorted(parsed, key=lambda t: t[0])[-1][1]


def save_training_state(model: Any, ckpt_dir: str, it: int) -> Optional[str]:
    """
    Save resume-relevant training state alongside periodic checkpoints.

    Writes:
      ckpt_dir/train_state_it{step}.pt   where step = it + 1

    Contents include:
      - optimizer_actor, optimizer_critic state_dict
      - scheduler_actor, scheduler_critic state_dict (if present)
      - _global_actor_update_step
      - it/step bookkeeping

    Returns written path, or None if failed.
    """
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        step = int(it) + 1
        path = training_state_path_for_start_it(ckpt_dir, step)
        payload: Dict[str, Any] = {
            "version": 1,
            "it_completed": int(it),
            "step": int(step),          # periodic ckpt step
            "next_it": int(step),       # next outer loop iteration index
            "global_actor_update_step": int(getattr(model, "_global_actor_update_step", 0) or 0),
        }
        # Optimizers
        payload["optimizer_actor"] = getattr(model, "optimizer_actor").state_dict()
        payload["optimizer_critic"] = getattr(model, "optimizer_critic").state_dict()
        # Schedulers (optional)
        sch_a = getattr(model, "scheduler_actor", None)
        sch_c = getattr(model, "scheduler_critic", None)
        payload["scheduler_actor"] = None if sch_a is None else sch_a.state_dict()
        payload["scheduler_critic"] = None if sch_c is None else sch_c.state_dict()

        torch.save(payload, path)
        return path
    except Exception as e:
        try:
            print(f"[save] training state save failed: {e}")
        except Exception:
            pass
        return None


def _apply_global_actor_update_step(model: Any, gstep: int):
    model._global_actor_update_step = int(max(0, int(gstep)))
    try:
        if hasattr(model, "net_actor") and hasattr(model.net_actor, "set_update_step"):
            model.net_actor.set_update_step(int(model._global_actor_update_step))
    except Exception as e:
        print(f"[resume] warning: failed to set actor schedule update_step: {e}")
    try:
        if hasattr(model, "_apply_flow_param_freeze"):
            model._apply_flow_param_freeze()
    except Exception:
        pass


def align_schedules_without_state(model: Any, start_it: int):
    """Fallback when no training-state file exists: align schedule state to start_it."""
    try:
        N = int(getattr(model, "N", 1) or 1)
        K_actor = int(getattr(model, "params_train", {}).get("K_actor", 1) or 1)
    except Exception:
        N, K_actor = 1, 1
    gstep = estimate_global_actor_update_step(start_it=start_it, N=N, K_actor=K_actor)
    _apply_global_actor_update_step(model, gstep=gstep)
    print(f"[resume] inferred start_it={int(start_it)}, global_actor_update_step={int(getattr(model, '_global_actor_update_step', gstep))}")


def align_multisteplr_without_state(model: Any, start_it: int):
    """
    Fallback LR alignment when no scheduler state exists.
    Training calls scheduler.step() once per outer iteration at the END of each iteration.
    Therefore, before starting iteration `start_it`, we need epoch = start_it-1.
    """
    start_it = int(max(0, int(start_it)))
    if start_it <= 0:
        return
    target_epoch = int(start_it - 1)

    def _step_to(sch, name: str):
        if sch is None:
            return
        try:
            sch.step(target_epoch)
        except TypeError:
            for _ in range(start_it):
                sch.step()
        except Exception as e:
            print(f"[resume] warning: failed to set {name} scheduler to epoch={target_epoch}: {e}")

    _step_to(getattr(model, "scheduler_critic", None), "critic")
    _step_to(getattr(model, "scheduler_actor", None), "actor")

    # Log current lrs for sanity
    try:
        lr_c = float(model.optimizer_critic.param_groups[0]["lr"])
    except Exception:
        lr_c = None
    try:
        lr_groups = [float(g.get("lr")) for g in getattr(model.optimizer_actor, "param_groups", [])]
    except Exception:
        lr_groups = None
    print(f"[resume] scheduler aligned to epoch={target_epoch}; lr_critic={lr_c}, lr_actor_groups={lr_groups}")


def try_load_training_state(
    model: Any,
    actor_path: Optional[str],
    critic_path: Optional[str],
    start_it_hint: int,
    device: Optional[torch.device] = None,
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Try to load optimizer/scheduler/global-step state.
    Returns (loaded, next_it, state_path).
    """
    ckpt_dir = _ckpt_dir_from_paths(actor_path, critic_path)
    if not ckpt_dir:
        return False, None, None

    fp = _find_best_training_state_file(ckpt_dir, start_it=int(start_it_hint))
    if not fp or (not os.path.isfile(fp)):
        return False, None, None

    try:
        map_loc = device if device is not None else "cpu"
        state: Dict[str, Any] = torch.load(fp, map_location=map_loc)

        # Optimizers
        try:
            model.optimizer_actor.load_state_dict(state.get("optimizer_actor", {}))
            model.optimizer_critic.load_state_dict(state.get("optimizer_critic", {}))
        except Exception as e:
            print(f"[resume] warning: optimizer state load failed ({fp}): {e}")
            return False, None, fp

        # Schedulers (optional)
        try:
            sa = state.get("scheduler_actor", None)
            sc = state.get("scheduler_critic", None)
            if sa is not None and getattr(model, "scheduler_actor", None) is not None:
                model.scheduler_actor.load_state_dict(sa)
            if sc is not None and getattr(model, "scheduler_critic", None) is not None:
                model.scheduler_critic.load_state_dict(sc)
        except Exception as e:
            print(f"[resume] warning: scheduler state load failed ({fp}): {e}")
            # Scheduler mismatch is not fatal; we can still proceed (LRs in optimizers are restored).

        gstep = int(state.get("global_actor_update_step", 0) or 0)
        _apply_global_actor_update_step(model, gstep=gstep)

        next_it = state.get("next_it", None)
        next_it = None if next_it is None else int(next_it)

        # Log current lrs for sanity
        try:
            lr_c = float(model.optimizer_critic.param_groups[0]["lr"])
        except Exception:
            lr_c = None
        try:
            lr_groups = [float(g.get("lr")) for g in getattr(model.optimizer_actor, "param_groups", [])]
        except Exception:
            lr_groups = None
        print(f"[resume] loaded training state from {fp}")
        print(f"[resume] next_it={next_it}, global_actor_update_step={int(getattr(model,'_global_actor_update_step', gstep))}, lr_critic={lr_c}, lr_actor_groups={lr_groups}")
        return True, next_it, fp
    except Exception as e:
        print(f"[resume] warning: failed to read training state file {fp}: {e}")
        return False, None, fp


def prepare_resume(
    model: Any,
    actor_path: Optional[str],
    critic_path: Optional[str],
    device: Optional[torch.device] = None,
) -> int:
    """
    Main entry point for scripts:
    - Infer start_it from ckpt paths / logs
    - If training state exists: restore optimizer/scheduler/global_step and return next_it
    - Else: align schedules + MultiStepLR to inferred start_it and return inferred start_it
    """
    start_it = int(infer_start_it(actor_path, critic_path))
    loaded, next_it, _fp = try_load_training_state(
        model=model,
        actor_path=actor_path,
        critic_path=critic_path,
        start_it_hint=start_it,
        device=device,
    )
    if loaded and (next_it is not None):
        return int(next_it)

    # Fallback (no state file): align internal schedules + LR schedulers
    align_schedules_without_state(model, start_it=start_it)
    align_multisteplr_without_state(model, start_it=start_it)
    return int(start_it)

