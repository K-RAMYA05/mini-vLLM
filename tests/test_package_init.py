import importlib
import sys
import builtins


def test_sampling_import_does_not_require_torch(monkeypatch):
    for name in list(sys.modules):
        if name == "mini_vllm" or name.startswith("mini_vllm."):
            sys.modules.pop(name)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mini_vllm.engine":
            raise AssertionError("mini_vllm.engine should not be imported eagerly")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    sampling = importlib.import_module("mini_vllm.sampling")
    assert sampling.SamplingParams(max_tokens=4).max_tokens == 4
