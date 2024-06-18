import dataclasses
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from schemas import StatusEnum 
from core.config import settings


tokenizer = AutoTokenizer.from_pretrained(settings.TOKENIZER)
model = AutoModel.from_pretrained(settings.MODEL)

feature_extraction_pipeline = pipeline("feature-extraction", model=model, tokenizer=tokenizer)


def _get_embedding(text: str):
    features = feature_extraction_pipeline(text)
    embeddings = torch.tensor(features).mean(dim=1)
    return embeddings


def _cosine_similarity(embedding1, embedding2):
    norm1 = torch.norm(embedding1)
    norm2 = torch.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return (embedding1 @ embedding2.T) / (norm1 * norm2)


@dataclasses.dataclass
class Relevance:
    reference_text: str
    threshold: float = settings.THRESHOLD

    def __post_init__(self):
        self.reference_embedding = _get_embedding(self.reference_text)

    def scan(self, text: str):
        target_embedding = _get_embedding(text)
        similarity = _cosine_similarity(self.reference_embedding, target_embedding)

        status = StatusEnum.PASSED if similarity >= self.threshold else StatusEnum.FAILED
        return similarity.item(), status
