"""OPSD configuration, validated at construction rather than at first use."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..privileged.schema import C_V0_DIM

__all__ = ["OPSDConfig"]

_MODES = ("teacher", "distill")
_PROVIDERS = ("privileged_c", "future_latent")
_WEIGHTINGS = ("uniform", "kl")


def _default_adapter_layers() -> list[int]:
    # Every 5th layer of 30. Privileged information is coarse context, not
    # something needing per-layer refresh; if the learned gates come out
    # uniformly large, revisit this before anything else.
    return [4, 9, 14, 19, 24, 29]


@dataclass
class OPSDConfig:
    enabled: bool = False
    mode: str = "teacher"

    # phi
    phi_provider: str = "future_latent"
    # Width of the raw privileged vector the encoder reads. RoboTwin's c v0 is
    # 28-D; UniVTAC's tactile summary is 22-D. Wrong here means the encoder
    # silently mis-slices every frame, so it is validated against the data.
    phi_c_dim: int = C_V0_DIM
    # Distillation objective. "mse" matches velocities pointwise, whose minimum
    # is the conditional mean; "energy" matches DISTRIBUTIONS with a proper
    # scoring rule, which is what on-policy self-distillation does on discrete
    # vocabularies and what a multi-valued teacher requires. See losses.py.
    loss_type: str = "mse"
    # Student action samples per observation for the energy score. Each costs
    # one extra sampler rollout, so this multiplies step time.
    energy_samples: int = 4
    phi_dim: int = 512
    phi_tokens: int = 8
    phi_heads: int = 8

    # adapters
    adapter_layers: Sequence[int] = field(default_factory=_default_adapter_layers)
    adapter_bottleneck: int = 512
    adapter_heads: int = 8
    adapter_expert: str = "action"
    # See GatedCrossAttnAdapter: alpha_init=0 preserves the model but cannot
    # bootstrap; zero_init_up=True with alpha_init>0 preserves AND trains.
    alpha_init: float = 0.0
    zero_init_up: bool = False

    # Stage A
    action_loss_only: bool = True
    # Control teacher: shuffle `privileged_c` across the batch during training, so
    # the adapters get the same data and capacity but no information from c. The
    # gap to a real-c teacher is what c itself is worth, free of the fine-tuning
    # effect the adapters have on their own.
    shuffle_c: bool = False

    # Stage B
    teacher_ckpt: Optional[str] = None
    num_sigma_points: int = 3
    distill_inference_steps: int = 10
    loss_weighting: str = "uniform"
    lambda_flow: float = 1.0
    lambda_bc: float = 1.0
    # Where the energy score supervises the student: "student" = states its own
    # sampler visits (on-policy, TOPS-WAM); "demo" = the demonstration's action
    # forward-noised to the same sigma points (off-policy ablation).
    state_source: str = "student"

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"`mode` must be one of {_MODES}, got {self.mode!r}")
        if self.lambda_flow < 0.0 or self.lambda_bc < 0.0:
            raise ValueError(
                f"`lambda_flow` and `lambda_bc` must be >= 0, got {self.lambda_flow} and {self.lambda_bc}"
            )
        if self.mode == "distill" and self.lambda_flow == 0.0 and self.lambda_bc == 0.0:
            raise ValueError("`lambda_flow` and `lambda_bc` are both 0: Stage B would train on nothing")
        if self.shuffle_c and (
            self.mode != "teacher" or self.phi_provider not in ("privileged_c", "future_latent")
        ):
            raise ValueError(
                "`shuffle_c` is a Stage A control; it needs mode='teacher' and a "
                f"privileged_c or future_latent provider, got mode={self.mode!r}, "
                f"phi_provider={self.phi_provider!r}"
            )
        if self.phi_provider not in _PROVIDERS:
            raise ValueError(f"`phi_provider` must be one of {_PROVIDERS}, got {self.phi_provider!r}")
        if self.loss_weighting not in _WEIGHTINGS:
            raise ValueError(
                f"`loss_weighting` must be one of {_WEIGHTINGS}, got {self.loss_weighting!r}"
            )
        if self.state_source not in ("student", "demo"):
            raise ValueError(f"`state_source` must be 'student' or 'demo', got {self.state_source!r}")
        if self.state_source == "demo" and self.loss_type not in ("energy", "action_mse"):
            raise ValueError("`state_source='demo'` is implemented for 'energy' and 'action_mse' only")
        if self.loss_type not in ("mse", "energy", "action_mse"):
            raise ValueError(
                f"`loss_type` must be 'mse', 'energy' or 'action_mse', got {self.loss_type!r}"
            )
        if self.loss_type == "energy" and self.energy_samples < 2:
            raise ValueError(
                f"the energy score needs at least 2 samples, got {self.energy_samples}"
            )
        if self.phi_c_dim < 1:
            raise ValueError(f"`phi_c_dim` must be >= 1, got {self.phi_c_dim}")
        if self.phi_dim < 1 or self.phi_tokens < 1:
            raise ValueError("`phi_dim` and `phi_tokens` must be >= 1")
        if self.phi_dim % self.phi_heads != 0:
            raise ValueError(
                f"`phi_dim` ({self.phi_dim}) must be divisible by `phi_heads` ({self.phi_heads})"
            )
        if self.adapter_bottleneck % self.adapter_heads != 0:
            raise ValueError(
                f"`adapter_bottleneck` ({self.adapter_bottleneck}) must be divisible by "
                f"`adapter_heads` ({self.adapter_heads})"
            )
        self.adapter_layers = [int(i) for i in self.adapter_layers]
        if len(set(self.adapter_layers)) != len(self.adapter_layers):
            raise ValueError(f"`adapter_layers` has duplicates: {self.adapter_layers}")
        if self.num_sigma_points < 1:
            raise ValueError(f"`num_sigma_points` must be >= 1, got {self.num_sigma_points}")
        if self.distill_inference_steps < 1:
            raise ValueError(
                f"`distill_inference_steps` must be >= 1, got {self.distill_inference_steps}"
            )
        # A lambda_flow = 0 run is the BC control: it never calls the teacher, so it
        # needs no checkpoint (and the engine does not build one).
        if self.mode == "distill" and self.lambda_flow > 0 and not self.teacher_ckpt:
            raise ValueError(
                "`mode='distill'` requires `teacher_ckpt`: the Stage-A checkpoint whose "
                "adapters and phi provider define the teacher. Without it the 'teacher' "
                "would be the untrained student and distillation would be a no-op."
            )

    @classmethod
    def from_cfg(cls, cfg) -> "OPSDConfig":
        """Build from an OmegaConf node or plain mapping; None/absent -> disabled."""
        if cfg is None:
            return cls(enabled=False)
        if not isinstance(cfg, dict):
            try:
                from omegaconf import OmegaConf

                cfg = OmegaConf.to_container(cfg, resolve=True)
            except Exception as exc:  # pragma: no cover - defensive
                raise TypeError(f"cannot interpret opsd config of type {type(cfg)!r}") from exc
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"unknown opsd config keys: {sorted(unknown)}")
        return cls(**cfg)
