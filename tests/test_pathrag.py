import sys, os
import tempfile
import types
import importlib.util

import pytest
import numpy as np

# Stub external dependencies used in PathRAG.llm

def make_stub(name, attrs=None):
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod

sys.modules.setdefault('aioboto3', make_stub('aioboto3'))
sys.modules.setdefault('aiohttp', make_stub('aiohttp'))
sys.modules.setdefault('ollama', make_stub('ollama'))
sys.modules.setdefault('torch', make_stub('torch'))
openai_stub = make_stub('openai', {
    'AsyncOpenAI': object,
    'APIConnectionError': Exception,
    'RateLimitError': Exception,
    'Timeout': Exception,
    'AsyncAzureOpenAI': object,
})
sys.modules.setdefault('openai', openai_stub)
sys.modules.setdefault('pydantic', make_stub('pydantic', {'BaseModel': object, 'Field': lambda *a, **k: None}))
sys.modules.setdefault('tenacity', make_stub('tenacity', {
    'retry': lambda *a, **k: (lambda f: f),
    'stop_after_attempt': lambda *a, **k: None,
    'wait_exponential': lambda *a, **k: None,
    'retry_if_exception_type': lambda *a, **k: None,
}))
sys.modules.setdefault('transformers', make_stub('transformers', {'AutoTokenizer': object, 'AutoModelForCausalLM': object}))
sys.modules.setdefault('tiktoken', make_stub('tiktoken', {'encoding_for_model': lambda m: type('Tok', (), {'encode': lambda self, s: [1], 'decode': lambda self, t: ""})()}))

# Dynamically import PathRAG and QueryParam without executing package __init__
pkg_path = os.path.join(os.path.dirname(__file__), '..', 'PathRAG')
pkg = types.ModuleType('PathRAG')
pkg.__path__ = [pkg_path]
sys.modules['PathRAG'] = pkg
spec = importlib.util.spec_from_file_location('PathRAG.PathRAG', os.path.join(pkg_path, 'PathRAG.py'))
mod = importlib.util.module_from_spec(spec)
sys.modules['PathRAG.PathRAG'] = mod
spec.loader.exec_module(mod)
PathRAG = mod.PathRAG
QueryParam = mod.QueryParam
EmbeddingFunc = mod.EmbeddingFunc

async def dummy_embedding(strings):
    return np.zeros((len(strings), 3))

async def dummy_llm(prompt, *args, **kwargs):
    return "ok"

@pytest.mark.asyncio
async def test_query_invalid_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        embed = EmbeddingFunc(embedding_dim=3, max_token_size=10, func=dummy_embedding)
        rag = PathRAG(
            working_dir=tmpdir,
            embedding_func=embed,
            llm_model_func=dummy_llm,
            graph_storage='NetworkXStorage',
        )
        with pytest.raises(ValueError):
            await rag.aquery('hi', QueryParam(mode='bad'))
