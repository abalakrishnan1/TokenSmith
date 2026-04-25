"""
scripts/cache_lru_test.py
"""

import pathlib
import sys
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

class MockEmbedder:
    """random embeddings keyed by text hash"""
    dim = 16

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            r = np.random.default_rng(abs(hash(t)) % (2**32))
            out[i] = r.standard_normal(self.dim).astype("float32")
        return out

import src.semantic_cache as sc

sc._get_embedder = lambda model_name: MockEmbedder()

def make_cache(capacity):
    return sc.SemanticCache(capacity=capacity, pca_min_samples=1)

def chunk_ids_in(cache):
    return {e.chunk_id for e in cache._entries}


def test_basic_lru_eviction():
    """capacity=6, insert 3 batches of 3 chunks, test eviction of oldest"""
    cache = make_cache(capacity=6)

    cache.insert("query A", ["a1", "a2", "a3"], chunk_ids=[0, 1, 2])
    cache.insert("query B", ["b1", "b2", "b3"], chunk_ids=[3, 4, 5])
    assert chunk_ids_in(cache) == {0, 1, 2, 3, 4, 5}, "both batches should fit"

    cache.insert("query C", ["c1", "c2", "c3"], chunk_ids=[6, 7, 8])
    ids = chunk_ids_in(cache)
    # ensure batch A is kicked out
    assert ids == {3, 4, 5, 6, 7, 8}, f"batch A should be evicted, got {ids}"
    print("  basic LRU eviction OK")


def test_promotion_on_hit():
    """test if a touched batch becomes MRU."""
    cache = make_cache(capacity=6)

    cache.insert("query A", ["a1", "a2", "a3"], chunk_ids=[0, 1, 2])
    cache.insert("query B", ["b1", "b2", "b3"], chunk_ids=[3, 4, 5])

    # promote batch A by moving it to the MRU end
    a_batch_id = cache._entries[0].batch_id
    cache._batch_order.move_to_end(a_batch_id)

    cache.insert("query C", ["c1", "c2", "c3"], chunk_ids=[6, 7, 8])
    ids = chunk_ids_in(cache)
    assert ids == {0, 1, 2, 6, 7, 8}, f"batch B should be evicted, got {ids}"
    print("  LRU promotion-on-hit OK")


def main():
    test_basic_lru_eviction()
    test_promotion_on_hit()
    print("\nAll LRU tests passed.")


if __name__ == "__main__":
    main()
