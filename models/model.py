"""Model aliases for the training pipeline."""

from models.fusion import MultimodalFusionNetwork

# Alias used by engine/ and main.py (mirrors EPGNN's MultimodalGNN naming pattern)
ExtinctionRiskModel = MultimodalFusionNetwork

__all__ = ["ExtinctionRiskModel", "MultimodalFusionNetwork"]
