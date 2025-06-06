import sys, os
import tempfile
import importlib.util
import types
from dataclasses import dataclass

import pytest
import numpy as np

# ensure optional deps don't fail
sys.modules.setdefault("tiktoken", types.ModuleType("tiktoken"))

pkg_path = os.path.join(os.path.dirname(__file__), '..', 'PathRAG')
pkg = types.ModuleType('PathRAG')
pkg.__path__ = [pkg_path]
sys.modules['PathRAG'] = pkg

spec = importlib.util.spec_from_file_location('PathRAG.kg.kuzu_impl', os.path.join(pkg_path, 'kg', 'kuzu_impl.py'))
mod = importlib.util.module_from_spec(spec)
sys.modules['PathRAG.kg.kuzu_impl'] = mod
spec.loader.exec_module(mod)
KuzuGraphStorage = mod.KuzuGraphStorage

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
    embed = EmbeddingFunc(embedding_dim=3, max_token_size=10, func=dummy_embedding)
    cfg = {"working_dir": tmpdir}
    return KuzuGraphStorage(namespace="test", global_config=cfg, embedding_func=embed)

@pytest.mark.asyncio
async def test_kuzu_storage_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = create_storage(tmpdir)
        await storage.upsert_node("A", {"entity_type": "person", "description": "a", "source_id": "1"})
        await storage.upsert_node("B", {"entity_type": "person", "description": "b", "source_id": "1"})
        await storage.upsert_edge(
            "A",
            "B",
            {"weight": 1.0, "description": "knows", "keywords": "knows", "source_id": "1"},
        )
        await storage.index_done_callback()

        res = storage._conn.execute("MATCH (e:Entity) RETURN COUNT(*)")
        assert res.get_next()[0] == 2
        res.close()
        res = storage._conn.execute(
            "MATCH (:Entity {id:$a})-[r:Relation]->(:Entity {id:$b}) RETURN COUNT(*)",
            {"a": "A", "b": "B"},
        )
        assert res.get_next()[0] == 1
        res.close()

        new_storage = create_storage(tmpdir)
        assert await new_storage.has_node("A") is True
        assert await new_storage.has_edge("A", "B") is True
