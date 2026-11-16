"""Drives both OPSD stages against a live `FastWAM` instance.

Kept out of `fastwam.py` so the model file stays readable and this logic stays
testable on its own. The engine is an `nn.Module` assigned to `model.opsd`, so
its adapters and phi provider are real submodules: `.to()`, `state_dict()`,
DeepSpeed sharding and the optimizer's `requires_grad` filter all see them
without special cases.

Stage A (`mode="teacher"`)
    Base expert frozen. Train adapters + phi provider on the ordinary action
    flow-matching loss, with phi supplied. Gates start at `tanh(0) = 0`, so the
    teacher begins bit-identical to the student and any gain is attributable to
    phi rather than to extra capacity being shaken loose.

Stage B (`mode="distill"`)
    Teacher frozen (loaded from the Stage-A checkpoint). The student rolls out
    its own action ODE, and is trained to match the teacher's velocity field at
    the highest-`delta^2` points on that trajectory.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

from .config import OPSDConfig
from .losses import action_mse_loss, energy_score_loss, flow_distillation_loss, implied_action
from .providers import build_phi_provider
from .rollout import record_student_trajectory
from .schedule import select_sigma_points

logger = logging.getLogger(__name__)

__all__ = ["OPSDEngine"]


def shuffled_c(sample: dict, generator: torch.Generator, zero_if_single: bool = False) -> dict:
    """Copy of `sample` whose `privileged_c` rows are rotated across the batch.

    A rotation by a random nonzero offset is a derangement: no sample keeps its
    own c, which a plain `randperm` would allow for ~1 sample per batch and so
    leak real information into the control.

    `zero_if_single` covers the trainer's periodic eval, which scores one sample
    at a time: there is no other row to rotate in, so c is zeroed instead. Both
    are uninformative, which is all the control requires. During training a batch
    of 1 stays an error, since it would mean the run is misconfigured.
    """
    c = sample.get("privileged_c")
    if c is None:
        raise KeyError("`shuffle_c` needs `privileged_c` in the batch")
    batch = int(c.shape[0])
    if batch < 2:
        if zero_if_single:
            out = dict(sample)
            out["privileged_c"] = torch.zeros_like(c)
            return out
        raise ValueError(
            "`shuffle_c` needs a per-device batch of at least 2; with 1 there is nothing "
            "to shuffle and the control would see its own c"
        )
    shift = int(torch.randint(1, batch, (1,), generator=generator).item())
    out = dict(sample)
    out["privileged_c"] = torch.roll(c, shifts=shift, dims=0)
    if "privileged_c_is_pad" in sample:
        out["privileged_c_is_pad"] = torch.roll(sample["privileged_c_is_pad"], shifts=shift, dims=0)
    return out


def shuffled_latents(inputs: dict, generator: torch.Generator, zero_if_single: bool = False) -> dict:
    """Copy of `inputs` whose `input_latents` are rotated across the batch.

    The hindsight control. The future-latent teacher reads another episode's
    future: same adapters, same capacity, same training, but no information about
    the clip it is acting in. Distilling from it keeps everything about the
    procedure -- on-policy rollouts, a smooth teacher target, the extra loss term
    -- and removes only the hindsight, which is the one thing a claim about
    privileged information has to be attributed to.

    Only the copy handed to the phi provider is rotated. The model's own forward
    pass keeps the real latents, so the student is trained on its own clip.
    """
    latents = inputs.get("input_latents")
    if latents is None:
        raise KeyError("`shuffle_c` with the future-latent provider needs `input_latents`")
    batch = int(latents.shape[0])
    out = dict(inputs)
    if batch < 2:
        if zero_if_single:
            out["input_latents"] = torch.zeros_like(latents)
            return out
        raise ValueError(
            "`shuffle_c` needs a per-device batch of at least 2; with 1 there is nothing "
            "to shuffle and the control would see its own future"
        )
    shift = int(torch.randint(1, batch, (1,), generator=generator).item())
    out["input_latents"] = torch.roll(latents, shifts=shift, dims=0)
    return out


class OPSDEngine(nn.Module):
    def __init__(self, model, config: OPSDConfig):
        super().__init__()
        self.config = config
        # Plain attribute, deliberately NOT registered as a submodule: the engine
        # must not own the model that owns it, or `state_dict()` recurses forever.
        object.__setattr__(self, "_model", model)

        # Re-entrancy guard. The engine calls back into `model.training_loss`,
        # which checks `model.opsd` -- without this the call bounces straight back
        # here and recurses until the stack dies.
        self.in_inner_call = False
        # Own RNG, so the shuffle does not shift the global stream: a control run
        # then draws the same noise and timesteps as the real-c run it is compared to.
        self._shuffle_generator = torch.Generator().manual_seed(0) if config.shuffle_c else None

        self.phi_provider = build_phi_provider(
            kind=config.phi_provider,
            out_dim=config.phi_dim,
            num_tokens=config.phi_tokens,
            num_heads=config.phi_heads,
            latent_channels=self._infer_latent_channels(model),
            c_dim=config.phi_c_dim,
        )

        added = model.mot.install_phi_adapters(
            expert_name=config.adapter_expert,
            layer_indices=config.adapter_layers,
            phi_dim=config.phi_dim,
            bottleneck=config.adapter_bottleneck,
            num_heads=config.adapter_heads,
            alpha_init=config.alpha_init,
            zero_init_up=config.zero_init_up,
        )
        provider_params = sum(p.numel() for p in self.phi_provider.parameters())
        logger.info(
            "OPSD mode=%s provider=%s: %.2f M adapter params + %.2f M provider params",
            config.mode,
            config.phi_provider,
            added / 1e6,
            provider_params / 1e6,
        )

        self.teacher = None
        # The BC control (lambda_flow = 0) never queries the teacher; skipping it
        # saves a ~2 GB frozen copy of the action expert and the checkpoint it needs.
        if config.mode == "distill" and config.lambda_flow > 0:
            self.teacher = _FrozenTeacher(model, config)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    @classmethod
    def maybe_build(cls, model, cfg) -> Optional["OPSDEngine"]:
        """Return an engine, or None when OPSD is off (the stock training path)."""
        config = OPSDConfig.from_cfg(cfg)
        if not config.enabled:
            return None
        return cls(model, config)

    @staticmethod
    def _infer_latent_channels(model) -> Optional[int]:
        for attr in ("latent_channels", "z_dim"):
            value = getattr(getattr(model, "vae", None), attr, None)
            if value is not None:
                return int(value)
        # Wan2.2 TI2V VAE is 48-channel. Only the future-latent provider needs
        # this, and it raises a shape error immediately if the guess is wrong.
        return 48

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #

    def training_loss(self, sample, tiled: bool = False):
        if self.config.mode == "teacher":
            return self._teacher_loss(sample, tiled=tiled)
        return self._distill_loss(sample, tiled=tiled)

    def _teacher_loss(self, sample, tiled: bool = False):
        model = self._model
        # Built once and shared: `input_latents` (the future-latent phi source)
        # only exists after build_inputs, and re-running it would re-encode video.
        inputs = model.build_inputs(sample, tiled=tiled)
        # The control shuffle lives here, not in the provider: the trainer puts
        # every module except the DiT in eval mode, so a provider gated on
        # `self.training` would silently never shuffle. The paired evaluation
        # scripts call the provider directly and bypass this path.
        phi_sample, phi_inputs = sample, inputs
        if self.config.shuffle_c:
            # The trainer's periodic eval scores a single sample under no_grad;
            # training always runs with grad and a full batch.
            single_ok = not torch.is_grad_enabled()
            if self.config.phi_provider == "future_latent":
                # The provider reads `input_latents` from `inputs`, not the raw
                # batch, so that is where the shuffle has to happen.
                phi_inputs = shuffled_latents(inputs, self._shuffle_generator, zero_if_single=single_ok)
            else:
                phi_sample = shuffled_c(sample, self._shuffle_generator, zero_if_single=single_ok)
        if self.config.shuffle_c and not getattr(self, "_shuffle_announced", False):
            logger.warning("[opsd] shuffle_c CONTROL active: %s is rotated across the batch",
                           "input_latents" if self.config.phi_provider == "future_latent" else "privileged_c")
            self._shuffle_announced = True
        phi = self.phi_provider(phi_sample, phi_inputs)
        self.in_inner_call = True
        try:
            loss, loss_dict = model.training_loss(
                sample,
                tiled=tiled,
                phi=phi,
                action_only=self.config.action_loss_only,
                inputs=inputs,
            )
        finally:
            self.in_inner_call = False

        loss_dict = dict(loss_dict)
        loss_dict["phi_gate"] = self.mean_gate()
        return loss, loss_dict

    def _distill_loss(self, sample, tiled: bool = False):
        model = self._model
        cfg = self.config

        inputs = model.build_inputs(sample, tiled=tiled)
        if cfg.lambda_flow == 0.0:
            # Fine-tuning baseline: exactly the L_bc term of a distillation run, with
            # the same data, trainable parameters, LR and steps -- but no teacher.
            # Skipping the rollout and teacher passes (rather than weighting them by
            # zero) makes it several times cheaper without changing what it trains.
            self.in_inner_call = True
            try:
                loss_bc, _ = model.training_loss(
                    sample, tiled=tiled, phi=None, action_only=True, inputs=inputs
                )
            finally:
                self.in_inner_call = False
            return cfg.lambda_bc * loss_bc, {
                "L_flow": 0.0,
                "L_bc": float(loss_bc.detach().item()),
                "sigma_mass": 0.0,
            }
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]

        ctx = _ActionContext.build(model, inputs)

        if cfg.loss_type in ("energy", "action_mse"):
            return self._energy_loss(sample, inputs, ctx, action, action_is_pad, tiled=tiled)

        # 1. The student rolls out its own sampler, under no_grad. These are the
        #    states inference actually visits -- the whole point of on-policy.
        noise = torch.randn_like(action)
        trajectory = record_student_trajectory(
            velocity_fn=lambda x, t: ctx.velocity(model, x, t, phi=None),
            scheduler=model.infer_action_scheduler,
            initial_noise=noise,
            num_inference_steps=cfg.distill_inference_steps,
        )

        # 2. Keep only the points that carry the discretization error.
        selection = select_sigma_points(
            deltas=trajectory.deltas,
            timesteps=trajectory.timesteps,
            num_points=cfg.num_sigma_points,
            num_train_timesteps=model.infer_action_scheduler.num_train_timesteps,
        )
        states = trajectory.gather(selection.indices)  # [K, B, T, D]

        # 3. Teacher scores those states, with phi. Frozen, no grad.
        with torch.no_grad():
            phi = self.teacher.phi(sample, inputs)
            v_teacher = torch.stack(
                [
                    self.teacher.velocity(
                        states[i], trajectory.timesteps[int(idx)], ctx, phi
                    )
                    for i, idx in enumerate(selection.indices)
                ],
                dim=0,
            )

        # 4. Student re-evaluates the same states, with grad. Gradient flows only
        #    here -- not through the 10 sequential rollout steps, which would be
        #    enormous and is unnecessary: the target is the field at a point.
        batch_size = states.shape[1]
        v_student = torch.stack(
            [
                ctx.velocity(
                    model,
                    states[i],
                    trajectory.timesteps[int(idx)].reshape(1).expand(batch_size),
                    phi=None,
                )
                for i, idx in enumerate(selection.indices)
            ],
            dim=0,
        )

        loss_flow = flow_distillation_loss(
            v_student=v_student,
            v_teacher=v_teacher,
            sigma=selection.sigmas,
            point_weights=selection.weights,
            valid_mask=action_is_pad,
            weighting=cfg.loss_weighting,
        )

        # 5. Behaviour-cloning anchor. Pure distillation lets the student follow
        #    the teacher wherever the teacher is wrong; the ground-truth term
        #    bounds that.
        loss_bc = torch.zeros((), device=loss_flow.device, dtype=loss_flow.dtype)
        if cfg.lambda_bc > 0.0:
            self.in_inner_call = True
            try:
                loss_bc, _ = model.training_loss(
                    sample, tiled=tiled, phi=None, action_only=True, inputs=inputs
                )
            finally:
                self.in_inner_call = False

        total = cfg.lambda_flow * loss_flow + cfg.lambda_bc * loss_bc
        return total, {
            "L_flow": float(loss_flow.detach().item()),
            "L_bc": float(loss_bc.detach().item()),
            "sigma_mass": selection.covered_mass,
        }

    def _energy_loss(self, sample, inputs, ctx, action, action_is_pad, tiled: bool = False):
        """Distributional distillation: match the teacher's action DISTRIBUTION.

        The squared-error path compares one student velocity against one teacher
        velocity, so its optimum is the conditional mean. Where the teacher
        conditions on something the student cannot infer, that mean is an action
        correct under no branch of the teacher's behaviour.

        Here the student draws `energy_samples` action chunks from its own
        sampler and the teacher draws one from its own -- both marginals over
        the same observation, the teacher's conditioned additionally on phi. A
        proper scoring rule then compares the distributions from that single
        teacher sample, which is all a demonstration frame can provide.

        Student samples come from `implied_action` at the selected sigma points
        rather than from differentiating the ten-step rollout: `y_sigma - sigma
        v` is exact under this flow convention, so each sample costs one forward.
        """
        model = self._model
        cfg = self.config
        batch_size = action.shape[0]

        # 1. WHERE the student is supervised, `energy_samples` times, no grad.
        #    state_source="student" (TOPS-WAM): states the student's own sampler visits.
        #    state_source="demo" (off-policy ablation): the demonstration's action,
        #    forward-noised to the same sigma points -- the states SFT trains on.
        #    Everything else (sigma points, teacher sample, loss, anchor) is shared.
        scheduler = model.infer_action_scheduler
        timesteps, deltas = scheduler.build_inference_schedule(
            num_inference_steps=cfg.distill_inference_steps,
            device=action.device,
            dtype=action.dtype,
        )
        selection = select_sigma_points(
            deltas=deltas,
            timesteps=timesteps,
            num_points=cfg.num_sigma_points,
            num_train_timesteps=scheduler.num_train_timesteps,
        )
        state_sets = []  # per sample m: [K, B, T, D]
        with torch.no_grad():
            for _ in range(cfg.energy_samples):
                if cfg.state_source == "demo":
                    state_sets.append(torch.stack([
                        (1.0 - float(sig)) * action + float(sig) * torch.randn_like(action)
                        for sig in selection.sigmas
                    ], dim=0))
                else:
                    traj = record_student_trajectory(
                        velocity_fn=lambda x, t: ctx.velocity(model, x, t, phi=None),
                        scheduler=scheduler,
                        initial_noise=torch.randn_like(action),
                        num_inference_steps=cfg.distill_inference_steps,
                    )
                    state_sets.append(traj.gather(selection.indices))

        # 2. One teacher sample: its own rollout, reading phi. The final state of
        #    that trajectory IS a draw from the teacher's action distribution.
        with torch.no_grad():
            phi = self.teacher.phi(sample, inputs)
            teacher_traj = record_student_trajectory(
                velocity_fn=lambda x, t: self.teacher.velocity(x, t, ctx, phi),
                scheduler=model.infer_action_scheduler,
                initial_noise=torch.randn_like(action),
                num_inference_steps=cfg.distill_inference_steps,
            )
            teacher_action = teacher_traj.final

        # 3. Differentiable student samples at the selected points.
        student_actions = []
        for states in state_sets:  # [K, B, T, D]
            per_point = []
            for i, idx in enumerate(selection.indices):
                timestep = timesteps[int(idx)]
                velocity = ctx.velocity(
                    model, states[i], timestep.reshape(1).expand(batch_size), phi=None
                )
                per_point.append(implied_action(states[i], velocity, selection.sigmas[i]))
            student_actions.append(torch.stack(per_point, dim=0))
        student_actions = torch.stack(student_actions, dim=0)  # [M, K, B, T, D]

        # loss_type="action_mse" (ablation): same samples and target, squared error
        # instead of the energy score -- its minimizer is the teacher's mean action.
        score_fn = action_mse_loss if cfg.loss_type == "action_mse" else energy_score_loss
        loss_flow = score_fn(
            student_actions=student_actions,
            teacher_action=teacher_action.unsqueeze(0).expand_as(student_actions[0]),
            point_weights=selection.weights,
            valid_mask=action_is_pad,
        )

        loss_bc = torch.zeros((), device=loss_flow.device, dtype=loss_flow.dtype)
        if cfg.lambda_bc > 0.0:
            self.in_inner_call = True
            try:
                loss_bc, _ = model.training_loss(
                    sample, tiled=tiled, phi=None, action_only=True, inputs=inputs
                )
            finally:
                self.in_inner_call = False

        total = cfg.lambda_flow * loss_flow + cfg.lambda_bc * loss_bc
        return total, {
            "L_flow": float(loss_flow.detach().item()),
            "L_bc": float(loss_bc.detach().item()),
            "sigma_mass": selection.covered_mass,
        }

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #

    def mean_gate(self) -> float:
        """Mean `|tanh(alpha)|` over installed adapters.

        The single most informative OPSD number. It is exactly 0 at init, and if
        it stays near 0 the teacher found phi useless -- refuting the transfer
        hypothesis directly, rather than by inference from success rates.
        """
        adapters = getattr(self._model.mot, "phi_adapters", None)
        if not adapters:
            return 0.0
        with torch.no_grad():
            gates = [a.gate().abs().float().mean() for a in adapters.values()]
            return float(torch.stack(gates).mean().item())

    def gate_report(self) -> dict:
        """Per-layer `tanh(alpha)`, for seeing WHERE phi gets used."""
        adapters = getattr(self._model.mot, "phi_adapters", None)
        if not adapters:
            return {}
        with torch.no_grad():
            return {key: float(a.gate().item()) for key, a in adapters.items()}


class _ActionContext:
    """Video-side state reused across every action evaluation in one batch.

    The video expert is frozen and its input is fixed for the batch, so its K/V
    are prefilled once and shared by all ~16 action passes a distillation step
    makes. Without this the rollout would re-run the 5 B video expert per ODE
    step and step cost would be dominated by recomputing a constant.
    """

    def __init__(self, video_kv_cache, attention_mask, video_seq_len, context, context_mask):
        self.video_kv_cache = video_kv_cache
        self.attention_mask = attention_mask
        self.video_seq_len = video_seq_len
        self.context = context
        self.context_mask = context_mask

    @classmethod
    def build(cls, model, inputs) -> "_ActionContext":
        first_frame = inputs["first_frame_latents"]
        if first_frame is None:
            first_frame = inputs["input_latents"][:, :, 0:1]
        context = inputs["context"]
        context_mask = inputs["context_mask"]

        timestep_video = torch.zeros(
            (first_frame.shape[0],), dtype=first_frame.dtype, device=first_frame.device
        )
        with torch.no_grad():
            video_pre = model.video_expert.pre_dit(
                x=first_frame,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            attention_mask = model._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=inputs["action"].shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
            video_kv_cache = model.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )
        return cls(video_kv_cache, attention_mask, video_seq_len, context, context_mask)

    def velocity(self, model, latents_action, timestep, phi=None):
        action_pre = model.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep,
            context=self.context,
            context_mask=self.context_mask,
        )
        action_tokens = model.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=self.video_kv_cache,
            attention_mask=self.attention_mask,
            video_seq_len=self.video_seq_len,
            phi=phi,
        )
        return model.action_expert.post_dit(action_tokens, action_pre)


class _FrozenTeacher(nn.Module):
    """A frozen copy of the action expert + adapters + phi provider.

    The teacher CANNOT share weights with the student in Stage B. The student's
    base weights are being trained, so a shared teacher would track the student
    and the distillation target would chase itself -- the classic self-distillation
    collapse. Costs one extra copy of the action expert (~2 GB in bf16).
    """

    def __init__(self, model, config: OPSDConfig):
        super().__init__()
        # Copy ONLY the action expert. The teacher exclusively calls
        # `forward_action_with_video_cache`, which reads the prefilled K/V and
        # never touches the video expert's weights.
        #
        # This has to exclude the video expert BEFORE the deepcopy, not after.
        # Copying the full MoT and deleting afterwards peaks at 12 GB (base) +
        # 12 GB (copy) = 24 GB during construction and OOMs a 24 GB card before
        # training even starts -- and because it happens inside instantiate(),
        # no batch-size or sigma-point setting can rescue it.
        source = model.mot
        saved_mixtures = source.mixtures
        saved_order = list(source.expert_order)
        try:
            source.mixtures = nn.ModuleDict({"action": saved_mixtures["action"]})
            source.expert_order = ["action"]
            self.mot = copy.deepcopy(source)
        finally:
            # Restore the originals by identity, so the live model is untouched.
            source.mixtures = saved_mixtures
            source.expert_order = saved_order

        # In FastWAM `action_expert` IS `mot.mixtures["action"]` -- the same object.
        # Alias rather than deepcopy it a second time: two copies would waste
        # 1.02 B parameters and, worse, let pre_dit/post_dit drift away from the
        # transformer blocks the adapters are attached to.
        self.action_expert = self.mot.mixtures["action"]
        self.phi_provider = build_phi_provider(
            kind=config.phi_provider,
            out_dim=config.phi_dim,
            num_tokens=config.phi_tokens,
            num_heads=config.phi_heads,
            latent_channels=OPSDEngine._infer_latent_channels(model),
            c_dim=config.phi_c_dim,
        )
        self.config = config
        self._load(config.teacher_ckpt)
        self.eval()
        self.requires_grad_(False)

    def _load(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = _teacher_state_from_payload(payload)
        result = self.load_state_dict(state, strict=False)
        if result.missing_keys:
            logger.warning(
                "teacher checkpoint missing %d keys (first: %s)",
                len(result.missing_keys),
                result.missing_keys[0],
            )
        # A gate check cannot detect a checkpoint that failed to load: the gates
        # are at `tanh(alpha_init)` from birth, so an untrained teacher sails
        # through it while its zero-initialized up-projection makes phi a no-op.
        # Stage B then distils the student from a copy of itself. Verify instead
        # that the two things phi actually flows through were populated.
        own = set(self.state_dict())
        matched = {k for k in state if k in own}
        loaded_adapters = sum(1 for k in matched if ".phi_adapters." in k)
        loaded_provider = sum(1 for k in matched if k.startswith("phi_provider."))
        if not loaded_adapters or not loaded_provider:
            raise ValueError(
                f"teacher checkpoint {path} populated {loaded_adapters} adapter and "
                f"{loaded_provider} phi-provider tensors -- phi cannot reach the "
                "teacher's output, so every distillation target would be the "
                "student's own velocity. Check the checkpoint layout (expects the "
                "trainer's `mot`/`opsd` payload) and that Stage A actually trained."
            )
        logger.info(
            "teacher loaded from %s: %d adapter + %d provider tensors",
            path, loaded_adapters, loaded_provider,
        )

    def sync_base(self, source_mot) -> None:
        """Re-copy the base weights from the live model, then re-apply the teacher.

        Ordering problem this solves: the engine (and therefore this teacher) is
        built inside `FastWAM.__init__`, which runs BEFORE the trainer calls
        `load_checkpoint(resume)`. So the deepcopy captures a randomly-initialized
        action expert. Without this the Stage-B teacher would be trained adapters
        bolted onto random weights -- a target the student would be actively
        harmed by matching, and one that fails silently.
        """
        own = dict(self.mot.named_parameters())
        own_buffers = dict(self.mot.named_buffers())
        with torch.no_grad():
            for name, param in source_mot.named_parameters():
                if name in own and own[name].shape == param.shape:
                    own[name].copy_(param)
            for name, buf in source_mot.named_buffers():
                if name in own_buffers and own_buffers[name].shape == buf.shape:
                    own_buffers[name].copy_(buf)
        # The base overwrite just clobbered the trained adapters, so re-apply them.
        self._load(self.config.teacher_ckpt)
        self.eval()
        self.requires_grad_(False)

    def phi(self, sample, inputs=None) -> torch.Tensor:
        return self.phi_provider(sample, inputs)

    def velocity(self, states, timestep, ctx: "_ActionContext", phi):
        # Two callers, two shapes. Scoring a point on the student's trajectory
        # passes the schedule's scalar timestep; the energy objective runs the
        # teacher through its own rollout, where `record_student_trajectory`
        # hands over one timestep per sample. Accept both rather than make the
        # rollout know which model it is driving.
        batch = states.shape[0]
        timestep = timestep.reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(batch)
        elif timestep.numel() != batch:
            raise ValueError(
                f"`timestep` has {timestep.numel()} entries for a batch of {batch}; "
                "expected a scalar or one per sample"
            )
        action_pre = self.action_expert.pre_dit(
            action_tokens=states,
            timestep=timestep,
            context=ctx.context,
            context_mask=ctx.context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=ctx.video_kv_cache,
            attention_mask=ctx.attention_mask,
            video_seq_len=ctx.video_seq_len,
            phi=phi,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)


def _teacher_state_from_payload(payload) -> dict:
    """Teacher-side state dict from whatever layout the checkpoint uses.

    `FastWAM.save_checkpoint` writes `{"mot": {...}, "opsd": {...}, "step": ...}`,
    where the adapters live inside `mot` (as `phi_adapters.*`) and the encoder
    inside `opsd` (as `phi_provider.*`). Those top-level keys are `mot` and
    `opsd` exactly -- not `mot.` -- so feeding the payload to a prefix-stripping
    remap matches nothing and silently loads an untrained teacher. Handle the
    trainer's layout explicitly, and keep the flat form for older files.
    """
    if not isinstance(payload, dict):
        return {}
    if "mot" in payload or "opsd" in payload:
        state = {}
        for key, value in (payload.get("mot") or {}).items():
            state["mot." + key] = value
        for key, value in (payload.get("opsd") or {}).items():
            # The Stage-B teacher is excluded when saving, but a nested teacher
            # in an older file must not be loaded into this one.
            if not key.startswith("teacher."):
                state[key] = value
        return state
    return _remap_teacher_state(payload.get("model", payload))


def _remap_teacher_state(state: dict) -> dict:
    """Strip the model-side prefixes so a flat Stage-A checkpoint loads here."""
    out = {}
    for key, value in state.items():
        if key.startswith("opsd.phi_provider."):
            out["phi_provider." + key[len("opsd.phi_provider."):]] = value
        elif key.startswith("phi_provider."):
            out[key] = value
        elif key.startswith("mot.") or key.startswith("action_expert."):
            out[key] = value
        elif key.startswith("dit."):
            out["mot." + key[len("dit."):]] = value
    return out
