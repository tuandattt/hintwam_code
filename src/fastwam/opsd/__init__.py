"""On-Policy Self-Distillation for World Action Models.

See PROPOSED_METHOD.md. Two training stages:

Stage A (`mode="teacher"`): the base action expert is frozen. Gated adapters and
the phi provider train on the ordinary flow-matching action loss, but with
privileged tokens supplied. Because the gates start at `tanh(0) = 0`, the teacher
begins bit-identical to the student and every gain is attributable to phi.

Stage B (`mode="distill"`): the teacher is frozen. The student integrates its own
action ODE, and is trained to match the teacher's velocity field at the highest-
weight points on that trajectory.
"""

from .config import OPSDConfig
from .losses import flow_distillation_loss, score_scale
from .providers import FutureLatentProvider, PrivilegedFeatureProvider, build_phi_provider
from .rollout import StudentTrajectory, record_student_trajectory
from .schedule import SigmaSelection, effective_sample_size, select_sigma_points

__all__ = [
    "OPSDConfig",
    "FutureLatentProvider",
    "PrivilegedFeatureProvider",
    "build_phi_provider",
    "flow_distillation_loss",
    "score_scale",
    "StudentTrajectory",
    "record_student_trajectory",
    "SigmaSelection",
    "select_sigma_points",
    "effective_sample_size",
]
