from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class PredictionData:
    prediction: str
    confidence: float
    heatmap_url: str

    def to_dict(self):
        return {
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "heatmap_url": self.heatmap_url
        }


@dataclass
class ApiResponse:
    success: bool
    prediction: PredictionData

    def to_dict(self):
        return {
            "success": self.success,
            "prediction": self.prediction.to_dict()
        }