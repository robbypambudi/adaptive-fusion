"""GMM-based fusion router (bukan ``FusionModule``): memilih model fusion terlatih per sampel."""

from adaptive_fusion.router.labels import FUSION_METHODS, router_labels, true_class_confidence

__all__ = ["FUSION_METHODS", "router_labels", "true_class_confidence"]
