import numpy as np
import torch

from videntra.retrieval.unified_pipeline import UnifiedMultimodalSearchPipeline


class Batch(dict):
    def to(self, device):
        return self


class Tokenizer:
    def __init__(self):
        self.batches = []

    def __call__(self, texts, **kwargs):
        self.batches.append(list(texts))
        return Batch(input_ids=torch.tensor([[sum(map(ord, text))] for text in texts]))


class Model:
    def get_text_features(self, input_ids):
        result = torch.zeros((len(input_ids), 768))
        result[:, 0] = 1
        result[:, 1] = input_ids[:, 0].float()
        return result


def test_deduplicate_nfc_and_bound_cache():
    tokenizer = Tokenizer()
    p = UnifiedMultimodalSearchPipeline(lazy_load=True, tokenizer=tokenizer, text_model=Model(), memory_cache_size=2)
    values = p.encode_texts(["café", "cafe\u0301", "other"])
    assert len(tokenizer.batches[0]) == 2
    np.testing.assert_array_equal(values[0], values[1])
    p.encode_texts(["café"])
    assert len(tokenizer.batches) == 1
    p.encode_texts(["third"])
    p.encode_texts(["other"])
    assert len(tokenizer.batches) == 3
    assert len(p._text_memory_cache) == 2


def test_cache_disabled_encodes_again():
    tokenizer = Tokenizer()
    p = UnifiedMultimodalSearchPipeline(lazy_load=True, tokenizer=tokenizer, text_model=Model(), memory_cache_size=0)
    p.encode_texts(["query"])
    p.encode_texts(["query"])
    assert len(tokenizer.batches) == 2
