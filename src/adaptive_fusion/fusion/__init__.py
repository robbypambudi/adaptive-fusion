"""Metode fusion. Impor modul di sini agar metodenya masuk ``FUSION_REGISTRY``."""

from adaptive_fusion.fusion import baselines  # noqa: F401  (mendaftarkan baseline)
from adaptive_fusion.fusion.base import (
    FUSION_REGISTRY,
    FusionModule,
    build_fusion,
    register_fusion,
)

__all__ = ["FUSION_REGISTRY", "FusionModule", "build_fusion", "register_fusion"]
