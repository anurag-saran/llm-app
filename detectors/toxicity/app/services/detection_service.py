from typing import Optional, Dict
import dataclasses
from transformers import pipeline
from schemas import StatusEnum 
from core.config import settings

@dataclasses.dataclass
class Model:

    kwargs: Dict = dataclasses.field(default_factory=dict)
    pipeline_kwargs: Dict = dataclasses.field(default_factory=dict)

    def load_model(self):
        return pipeline("text-classification", model="pykeio/lite-toxic-comment-classification", **self.pipeline_kwargs)


class Toxicity:
    """
    A class used to detect toxicity in the output of a language model.
    """
    def __init__(
        self,
        model: Optional[Model] = None,
        threshold: float = settings.THRESHOLD,
    ):
        """
        Initializes an instance of the Toxicity class.
        """
        self.threshold = threshold
        self.toxicity_scanner = model.load_model() if model else Model().load_model()

    def scan(self, text: str):

        if not self.toxicity_scanner:
            raise Exception("Model not loaded properly.")

        results = self.toxicity_scanner(text)
        score = max((res['score'] for res in results if res['label'] == 'toxic'), default=0.0)
        
        status = StatusEnum.FAILED if any(res['score'] > self.threshold for res in results if res['label'] == 'toxic') else StatusEnum.PASSED

        return score, status
