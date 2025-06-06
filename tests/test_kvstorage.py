import sys, os
import tempfile
import importlib.util
import types
sys.modules.setdefault("tiktoken", types.ModuleType("tiktoken"))
from dataclasses import dataclass

import pytest
import numpy as np

# Dynamically load storage module without executing package __init__
pkg_path = os.path.join(os.path.dirname(__file__), '..', 'PathRAG')
pkg = types.ModuleType('PathRAG')
pkg.__path__ = [pkg_path]
sys.modules['PathRAG'] = pkg
path = os.path.join(pkg_path, 'storage.py')
spec = importlib.util.spec_from_file_location('PathRAG.storage', path)
storage = importlib.util.module_from_spec(spec)
sys.modules['PathRAG.storage'] = storage
spec.loader.exec_module(storage)
JsonKVStorage = storage.JsonKVStorage

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable
    async def __call__(self, *args, **kwargs):
        return await self.func(*args, **kwargs)

async def dummy_embedding(strings):
    return np.zeros((len(strings), 3))


def create_storage(tmpdir):
    embed_func = EmbeddingFunc(embedding_dim=3, max_token_size=10, func=dummy_embedding)
    cfg = {"working_dir": tmpdir, "embedding_batch_num": 1}
    return JsonKVStorage(namespace="test", global_config=cfg, embedding_func=embed_func)


@pytest.mark.asyncio
async def test_filter_and_get_by_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = create_storage(tmpdir)
        await storage.upsert({"a": {"content": "1"}, "b": {"content": "2"}})

        missing = await storage.filter_keys(["a", "b", "c"])
        assert missing == {"c"}

        result = await storage.get_by_id("unknown")
        assert result is None
