from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class PredictionData:
    prediction: str
    confidence: float
    heatmap_url: str
    probabilities: dict
    affected_region: str
    report: str

    def to_dict(self):
        return {
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "probabilities": self.probabilities ,
            "heatmap_url": self.heatmap_url,
            "affected_region": self.affected_region,
            "report": self.report
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