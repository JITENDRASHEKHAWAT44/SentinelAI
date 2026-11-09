"""
Compatibility and environment shim for RAGAS and LangChain.
Ensures pyarrow, xxhash, and langchain_community stubs load cleanly on Windows.
"""

import sys
import types
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "backend" / ".env")

# 1. xxhash AppLocker/Windows C-Extension fallback
if "xxhash" not in sys.modules:
    try:
        import xxhash  # noqa: F401
    except ImportError:
        class _FallbackXXHash:
            @staticmethod
            def xxh3_128(data: bytes):
                class _Digest:
                    def digest(self):
                        return hashlib.md5(data).digest()
                return _Digest()
        sys.modules["xxhash"] = _FallbackXXHash()

# 2. PyArrow compatibility for datasets 2.14.x with pyarrow >= 20.x
try:
    import pyarrow as pa
    if not hasattr(pa, "PyExtensionType") and hasattr(pa, "ExtensionType"):
        pa.PyExtensionType = pa.ExtensionType
except Exception:
    pass

# 3. LangChain community stub for legacy VertexAI imports in ragas.llms.base
if "langchain_community.chat_models.vertexai" not in sys.modules:
    m1 = types.ModuleType("langchain_community.chat_models.vertexai")
    m1.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = m1

if "langchain_community.llms.vertexai" not in sys.modules:
    m2 = types.ModuleType("langchain_community.llms.vertexai")
    m2.VertexAI = type("VertexAI", (), {})
    sys.modules["langchain_community.llms.vertexai"] = m2

# 4. nest_asyncio compatibility with Python 3.12+ uvicorn (loop_factory parameter)
try:
    import asyncio
    import nest_asyncio

    _orig_nest_apply = nest_asyncio.apply

    def _safe_nest_apply(*args, **kwargs):
        _orig_nest_apply(*args, **kwargs)
        _inner_run = asyncio.run

        def _compat_run(main, *, debug=False, **extra):
            try:
                return _inner_run(main, debug=debug)
            except (KeyboardInterrupt, asyncio.CancelledError):
                return None

        asyncio.run = _compat_run

    nest_asyncio.apply = _safe_nest_apply
    # Also patch current asyncio.run if nest_asyncio was already applied
    _cur_run = asyncio.run

    def _compat_cur_run(main, *, debug=False, **extra):
        try:
            return _cur_run(main, debug=debug, **extra)
        except (KeyboardInterrupt, asyncio.CancelledError):
            return None
        except TypeError:
            try:
                return _cur_run(main, debug=debug)
            except (KeyboardInterrupt, asyncio.CancelledError):
                return None

    asyncio.run = _compat_cur_run
except Exception:
    pass

