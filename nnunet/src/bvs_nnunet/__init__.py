"""Independent nnU-Net v2 integration for the BVS Lingfeng architecture."""

from .networks import KDNetwork, StudentNetwork, TeacherNetwork

__all__ = ["KDNetwork", "StudentNetwork", "TeacherNetwork"]
