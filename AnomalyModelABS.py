from abc import abstractmethod, ABC


class AnomalyModel(ABC):
    """Base for all anomaly models"""

    @abstractmethod
    def predict_anomaly_proba(self, point) -> float:
        """Predicts anomaly probability for new point"""
        pass
