"""Run mini_vllm on Modal.

Usage:
    pip install modal
    modal run examples/modal_app.py --gpu a100 --mode generate --prompt "Explain paged attention"
    modal run examples/modal_app.py --gpu h100 --mode throughput
    modal deploy examples/modal_app.py

After deploy, the FastAPI app is exposed as web endpoints:
  - mini-vllm-a100
  - mini-vllm-h100

This file also exposes remote functions for the rest of the project:
  - throughput benchmark
  - speculative benchmark
  - teacher-data generation
  - distillation training
  - draft acceptance eval
  - HumanEval quality eval
  - generic repo CLI/module execution
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal


APP_NAME = "mini-vllm"
MODEL_NAME = "meta-llama/Llama-3.1-8B"
REMOTE_ROOT = "/root/mini_vllm"
REMOTE_RESULTS = "/cache/project"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("mini-vllm-hf-cache", create_if_missing=True)
triton_cache = modal.Volume.from_name("mini-vllm-triton-cache", create_if_missing=True)
project_cache = modal.Volume.from_name("mini-vllm-project-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .add_local_dir(
        ".",
        remote_path=REMOTE_ROOT,
        copy=True,
        ignore=[
            ".git",
            ".pytest_cache",
            ".venv",
            "__pycache__",
            "*.pyc",
            "results",
        ],
    )
    .workdir(REMOTE_ROOT)
    .run_commands(
        "pip install --upgrade pip",
        "pip install --index-url https://download.pytorch.org/whl/cu124 torch",
        "pip install -e '.[serve,bench,test]'",
        "pip install human-eval",
        "pip install packaging ninja wheel psutil",
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "TRANSFORMERS_CACHE": "/cache/hf",
            "TRITON_CACHE_DIR": "/cache/triton",
            "PYTHONUNBUFFERED": "1",
        }
    )
)


def _modal_kwargs(gpu: str, timeout_s: int) -> dict:
    return dict(
        image=image,
        gpu=gpu,
        timeout=timeout_s,
        scaledown_window=60 * 5,
        secrets=[hf_secret],
        volumes={
            "/cache/hf": hf_cache,
            "/cache/triton": triton_cache,
            "/cache/project": project_cache,
        },
    )


def _engine_config_kwargs(prefill_backend: str = "flash") -> dict:
    return dict(
        model_name_or_path=MODEL_NAME,
        dtype="bfloat16",
        device="cuda",
        num_gpu_blocks=4096,
        max_num_seqs=16,
        enable_prefix_cache=True,
        prefill_attention_backend=prefill_backend,
    )


def _run_subprocess(argv: list[str]) -> dict:
    proc = subprocess.run(
        argv,
        cwd=REMOTE_ROOT,
        text=True,
        capture_output=True,
    )
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _output_path(name: str) -> str:
    path = Path(REMOTE_RESULTS) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _bench_throughput_impl(
    batch_size: int,
    num_prompts: int,
    prompt_len: int,
    max_tokens: int,
    prefill_backend: str,
    skip_hf: bool,
) -> dict:
    argv = [
        "python",
        "-m",
        "benchmarks.bench_throughput",
        "--model",
        MODEL_NAME,
        "--dtype",
        "bfloat16",
        "--batch-size",
        str(batch_size),
        "--num-prompts",
        str(num_prompts),
        "--prompt-len",
        str(prompt_len),
        "--max-tokens",
        str(max_tokens),
        "--prefill-backend",
        prefill_backend,
    ]
    if skip_hf:
        argv.append("--skip-hf")
    return _run_subprocess(argv)


def _bench_speculative_impl(draft_path: str, gamma: int, max_tokens: int) -> dict:
    argv = [
        "python",
        "-m",
        "benchmarks.bench_speculative",
        "--target",
        MODEL_NAME,
        "--draft",
        draft_path,
        "--gamma",
        str(gamma),
        "--max-tokens",
        str(max_tokens),
        "--dtype",
        "bfloat16",
    ]
    return _run_subprocess(argv)


def _distill_generate_teacher_impl(
    num_sequences: int,
    seq_len: int,
    topk: int,
    batch_size: int,
    output_subdir: str,
) -> dict:
    output_dir = _output_path(output_subdir)
    argv = [
        "python",
        "-m",
        "mini_vllm.distill.generate_teacher_data",
        "--teacher",
        MODEL_NAME,
        "--num-sequences",
        str(num_sequences),
        "--seq-len",
        str(seq_len),
        "--output-dir",
        output_dir,
        "--topk",
        str(topk),
        "--batch-size",
        str(batch_size),
    ]
    result = _run_subprocess(argv)
    result["output_dir"] = output_dir
    return result


def _distill_train_impl(
    data_subdir: str,
    output_subdir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    use_bf16: bool,
    init_from_pretrained: str = "meta-llama/Llama-3.2-1B",
) -> dict:
    data_dir = _output_path(data_subdir)
    output_dir = _output_path(output_subdir)
    argv = [
        "python",
        "-m",
        "mini_vllm.distill.train_distill",
        "--teacher",
        MODEL_NAME,
        "--data-dir",
        data_dir,
        "--init-from-pretrained",
        init_from_pretrained,
        "--output-dir",
        output_dir,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
    ]
    if use_bf16:
        argv.append("--use-bf16")
    result = _run_subprocess(argv)
    result["data_dir"] = data_dir
    result["output_dir"] = output_dir
    return result


def _eval_acceptance_impl(
    draft_subdir: str,
    gamma: int,
    num_sequences: int,
    max_new_tokens: int,
    json_subpath: str,
) -> dict:
    draft_dir = _output_path(draft_subdir)
    json_out = _output_path(json_subpath)
    argv = [
        "python",
        "-m",
        "mini_vllm.distill.eval_acceptance",
        "--target",
        MODEL_NAME,
        "--draft",
        draft_dir,
        "--gamma",
        str(gamma),
        "--num-sequences",
        str(num_sequences),
        "--max-new-tokens",
        str(max_new_tokens),
        "--dtype",
        "bfloat16",
        "--json-out",
        json_out,
    ]
    result = _run_subprocess(argv)
    result["draft_dir"] = draft_dir
    result["json_out"] = json_out
    return result


def _eval_humaneval_impl(
    configs: list[str] | None,
    draft_subdir: str,
    num_problems: int,
    max_new_tokens: int,
    json_subpath: str,
    markdown_subpath: str,
) -> dict:
    if configs is None:
        configs = [
            "mini-vllm-baseline",
            "mini-vllm-int8kv",
            "mini-vllm-gptq8",
        ]
    json_out = _output_path(json_subpath)
    markdown_out = _output_path(markdown_subpath)
    argv = [
        "python",
        "-m",
        "benchmarks.eval_humaneval",
        "--model",
        MODEL_NAME,
        "--num-problems",
        str(num_problems),
        "--max-new-tokens",
        str(max_new_tokens),
        "--json-out",
        json_out,
        "--markdown-out",
        markdown_out,
        "--configs",
        *configs,
    ]
    if "mini-vllm-spec" in configs:
        argv.extend(["--draft", _output_path(draft_subdir)])
    result = _run_subprocess(argv)
    result["json_out"] = json_out
    result["markdown_out"] = markdown_out
    return result


SMALL_DRAFT_INIT = "meta-llama/Llama-3.2-1B"


def _small_draft_smoke_impl(prefix: str = "small-draft-smoke") -> dict:
    teacher = _distill_generate_teacher_impl(
        num_sequences=64,
        seq_len=128,
        topk=20,
        batch_size=4,
        output_subdir=f"{prefix}/distill_data",
    )
    if teacher["returncode"] != 0:
        return {"stage": "teacher-data", "teacher": teacher}

    train = _distill_train_impl(
        data_subdir=f"{prefix}/distill_data",
        output_subdir=f"{prefix}/draft-llama32-1b",
        epochs=1,
        batch_size=4,
        lr=1e-4,
        use_bf16=True,
        init_from_pretrained=SMALL_DRAFT_INIT,
    )
    if train["returncode"] != 0:
        return {"stage": "train-distill", "teacher": teacher, "train": train}

    acceptance = _eval_acceptance_impl(
        draft_subdir=f"{prefix}/draft-llama32-1b/final",
        gamma=4,
        num_sequences=16,
        max_new_tokens=32,
        json_subpath=f"{prefix}/eval_acceptance.json",
    )
    if acceptance["returncode"] != 0:
        return {
            "stage": "eval-acceptance",
            "teacher": teacher,
            "train": train,
            "acceptance": acceptance,
        }

    speculative = _bench_speculative_impl(
        draft_path=_output_path(f"{prefix}/draft-llama32-1b/final"),
        gamma=4,
        max_tokens=32,
    )
    return {
        "stage": "done",
        "teacher": teacher,
        "train": train,
        "acceptance": acceptance,
        "speculative": speculative,
        "artifacts_root": _output_path(prefix),
    }


def _small_draft_impl(prefix: str = "small-draft") -> dict:
    teacher = _distill_generate_teacher_impl(
        num_sequences=5000,
        seq_len=384,
        topk=50,
        batch_size=8,
        output_subdir=f"{prefix}/distill_data",
    )
    if teacher["returncode"] != 0:
        return {"stage": "teacher-data", "teacher": teacher}

    train = _distill_train_impl(
        data_subdir=f"{prefix}/distill_data",
        output_subdir=f"{prefix}/draft-llama32-1b",
        epochs=2,
        batch_size=16,
        lr=1e-4,
        use_bf16=True,
        init_from_pretrained=SMALL_DRAFT_INIT,
    )
    if train["returncode"] != 0:
        return {"stage": "train-distill", "teacher": teacher, "train": train}

    acceptance = _eval_acceptance_impl(
        draft_subdir=f"{prefix}/draft-llama32-1b/final",
        gamma=4,
        num_sequences=128,
        max_new_tokens=64,
        json_subpath=f"{prefix}/eval_acceptance.json",
    )
    if acceptance["returncode"] != 0:
        return {
            "stage": "eval-acceptance",
            "teacher": teacher,
            "train": train,
            "acceptance": acceptance,
        }

    speculative = _bench_speculative_impl(
        draft_path=_output_path(f"{prefix}/draft-llama32-1b/final"),
        gamma=4,
        max_tokens=96,
    )
    humaneval = _eval_humaneval_impl(
        configs=["mini-vllm-baseline", "mini-vllm-spec"],
        draft_subdir=f"{prefix}/draft-llama32-1b/final",
        num_problems=32,
        max_new_tokens=256,
        json_subpath=f"{prefix}/quality_eval.json",
        markdown_subpath=f"{prefix}/quality_eval.md",
    )
    return {
        "stage": "done",
        "teacher": teacher,
        "train": train,
        "acceptance": acceptance,
        "speculative": speculative,
        "humaneval": humaneval,
        "artifacts_root": _output_path(prefix),
    }


def _generate_impl(prompt: str, max_tokens: int = 128) -> str:
    from mini_vllm import EngineConfig, LLMEngine, SamplingParams

    engine = LLMEngine(EngineConfig(**_engine_config_kwargs()))
    engine.add_request(prompt, SamplingParams(max_tokens=max_tokens, temperature=0.0))
    return engine.run_until_done()[0].output_text


@app.function(**_modal_kwargs("A100-80GB", 60 * 30))
def generate_a100(prompt: str, max_tokens: int = 128) -> str:
    return _generate_impl(prompt, max_tokens=max_tokens)


@app.function(**_modal_kwargs("H100", 60 * 30))
def generate_h100(prompt: str, max_tokens: int = 128) -> str:
    return _generate_impl(prompt, max_tokens=max_tokens)


def _serve_impl():
    from mini_vllm import EngineConfig, LLMEngine
    from mini_vllm.entrypoints.openai_server import create_app

    engine = LLMEngine(EngineConfig(**_engine_config_kwargs()))
    return create_app(engine)


@app.function(**_modal_kwargs("A100-80GB", 60 * 60))
@modal.asgi_app(label="mini-vllm-a100")
def serve_a100():
    return _serve_impl()


@app.function(**_modal_kwargs("H100", 60 * 60))
@modal.asgi_app(label="mini-vllm-h100")
def serve_h100():
    return _serve_impl()


@app.function(**_modal_kwargs("A100-80GB", 60 * 30))
def bench_throughput_a100(
    batch_size: int = 32,
    num_prompts: int = 128,
    prompt_len: int = 512,
    max_tokens: int = 128,
    prefill_backend: str = "flash",
    skip_hf: bool = False,
) -> dict:
    return _bench_throughput_impl(
        batch_size=batch_size,
        num_prompts=num_prompts,
        prompt_len=prompt_len,
        max_tokens=max_tokens,
        prefill_backend=prefill_backend,
        skip_hf=skip_hf,
    )


@app.function(**_modal_kwargs("H100", 60 * 30))
def bench_throughput_h100(
    batch_size: int = 32,
    num_prompts: int = 128,
    prompt_len: int = 512,
    max_tokens: int = 128,
    prefill_backend: str = "flash",
    skip_hf: bool = False,
) -> dict:
    return _bench_throughput_impl(
        batch_size=batch_size,
        num_prompts=num_prompts,
        prompt_len=prompt_len,
        max_tokens=max_tokens,
        prefill_backend=prefill_backend,
        skip_hf=skip_hf,
    )


@app.function(**_modal_kwargs("A100-80GB", 60 * 30))
def bench_speculative_a100(
    draft_path: str,
    gamma: int = 4,
    max_tokens: int = 128,
) -> dict:
    return _bench_speculative_impl(draft_path=draft_path, gamma=gamma, max_tokens=max_tokens)


@app.function(**_modal_kwargs("H100", 60 * 30))
def bench_speculative_h100(
    draft_path: str,
    gamma: int = 4,
    max_tokens: int = 128,
) -> dict:
    return _bench_speculative_impl(draft_path=draft_path, gamma=gamma, max_tokens=max_tokens)


@app.function(**_modal_kwargs("A100-80GB", 60 * 60 * 6))
def distill_generate_teacher_a100(
    num_sequences: int = 5000,
    seq_len: int = 512,
    topk: int = 50,
    batch_size: int = 8,
    output_subdir: str = "distill_data",
) -> dict:
    return _distill_generate_teacher_impl(
        num_sequences=num_sequences,
        seq_len=seq_len,
        topk=topk,
        batch_size=batch_size,
        output_subdir=output_subdir,
    )


@app.function(**_modal_kwargs("H100", 60 * 60 * 6))
def distill_generate_teacher_h100(
    num_sequences: int = 5000,
    seq_len: int = 512,
    topk: int = 50,
    batch_size: int = 8,
    output_subdir: str = "distill_data",
) -> dict:
    return _distill_generate_teacher_impl(
        num_sequences=num_sequences,
        seq_len=seq_len,
        topk=topk,
        batch_size=batch_size,
        output_subdir=output_subdir,
    )


@app.function(**_modal_kwargs("A100-80GB", 60 * 60 * 12))
def distill_train_a100(
    data_subdir: str = "distill_data",
    output_subdir: str = "outputs/llama-3.1-8b-draft-llama32-1b",
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 1e-4,
    use_bf16: bool = True,
    init_from_pretrained: str = "meta-llama/Llama-3.2-1B",
) -> dict:
    return _distill_train_impl(
        data_subdir=data_subdir,
        output_subdir=output_subdir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_bf16=use_bf16,
        init_from_pretrained=init_from_pretrained,
    )


@app.function(**_modal_kwargs("H100", 60 * 60 * 12))
def distill_train_h100(
    data_subdir: str = "distill_data",
    output_subdir: str = "outputs/llama-3.1-8b-draft-llama32-1b",
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 1e-4,
    use_bf16: bool = True,
    init_from_pretrained: str = "meta-llama/Llama-3.2-1B",
) -> dict:
    return _distill_train_impl(
        data_subdir=data_subdir,
        output_subdir=output_subdir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_bf16=use_bf16,
        init_from_pretrained=init_from_pretrained,
    )


@app.function(**_modal_kwargs("A100-80GB", 60 * 60 * 3))
def eval_acceptance_a100(
    draft_subdir: str = "outputs/llama-3.1-8b-draft-llama32-1b/final",
    gamma: int = 4,
    num_sequences: int = 100,
    max_new_tokens: int = 128,
    json_subpath: str = "results/eval_acceptance.json",
) -> dict:
    return _eval_acceptance_impl(
        draft_subdir=draft_subdir,
        gamma=gamma,
        num_sequences=num_sequences,
        max_new_tokens=max_new_tokens,
        json_subpath=json_subpath,
    )


@app.function(**_modal_kwargs("H100", 60 * 60 * 3))
def eval_acceptance_h100(
    draft_subdir: str = "outputs/llama-3.1-8b-draft-llama32-1b/final",
    gamma: int = 4,
    num_sequences: int = 100,
    max_new_tokens: int = 128,
    json_subpath: str = "results/eval_acceptance.json",
) -> dict:
    return _eval_acceptance_impl(
        draft_subdir=draft_subdir,
        gamma=gamma,
        num_sequences=num_sequences,
        max_new_tokens=max_new_tokens,
        json_subpath=json_subpath,
    )


@app.function(**_modal_kwargs("A100-80GB", 60 * 60 * 4))
def eval_humaneval_a100(
    configs: list[str] | None = None,
    draft_subdir: str = "outputs/llama-3.1-8b-draft-llama32-1b/final",
    num_problems: int = 164,
    max_new_tokens: int = 512,
    json_subpath: str = "results/quality_eval.json",
    markdown_subpath: str = "results/quality_eval.md",
) -> dict:
    return _eval_humaneval_impl(
        configs=configs,
        draft_subdir=draft_subdir,
        num_problems=num_problems,
        max_new_tokens=max_new_tokens,
        json_subpath=json_subpath,
        markdown_subpath=markdown_subpath,
    )


@app.function(**_modal_kwargs("H100", 60 * 60 * 4))
def eval_humaneval_h100(
    configs: list[str] | None = None,
    draft_subdir: str = "outputs/llama-3.1-8b-draft-llama32-1b/final",
    num_problems: int = 164,
    max_new_tokens: int = 512,
    json_subpath: str = "results/quality_eval.json",
    markdown_subpath: str = "results/quality_eval.md",
) -> dict:
    return _eval_humaneval_impl(
        configs=configs,
        draft_subdir=draft_subdir,
        num_problems=num_problems,
        max_new_tokens=max_new_tokens,
        json_subpath=json_subpath,
        markdown_subpath=markdown_subpath,
    )


@app.function(**_modal_kwargs("A100-80GB", 60 * 60 * 2))
def small_draft_smoke_a100(prefix: str = "small-draft-smoke") -> dict:
    return _small_draft_smoke_impl(prefix=prefix)


@app.function(**_modal_kwargs("H100", 60 * 60 * 2))
def small_draft_smoke_h100(prefix: str = "small-draft-smoke") -> dict:
    return _small_draft_smoke_impl(prefix=prefix)


@app.function(**_modal_kwargs("A100-80GB", 60 * 60 * 6))
def small_draft_a100(prefix: str = "small-draft") -> dict:
    return _small_draft_impl(prefix=prefix)


@app.function(**_modal_kwargs("H100", 60 * 60 * 6))
def small_draft_h100(prefix: str = "small-draft") -> dict:
    return _small_draft_impl(prefix=prefix)


@app.function(image=image, timeout=60 * 10, secrets=[hf_secret], volumes={"/cache/project": project_cache})
def analyze_sweep(
    sweep_subdir: str,
    gamma: int = 4,
    report_subpath: str = "results/sweep_report.md",
) -> dict:
    argv = [
        "python",
        "-m",
        "mini_vllm.distill.analyze_sweep",
        "--sweep-root",
        _output_path(sweep_subdir),
        "--gamma",
        str(gamma),
        "--report",
        _output_path(report_subpath),
    ]
    result = _run_subprocess(argv)
    result["report"] = _output_path(report_subpath)
    return result


@app.function(**_modal_kwargs("A100-80GB", 60 * 60 * 3))
def run_cli_a100(argv: list[str]) -> dict:
    return _run_subprocess(["python", *argv])


@app.function(**_modal_kwargs("H100", 60 * 60 * 3))
def run_cli_h100(argv: list[str]) -> dict:
    return _run_subprocess(["python", *argv])


@app.local_entrypoint()
def main(
    mode: str = "generate",
    gpu: str = "a100",
    prompt: str = "Explain paged attention.",
    max_tokens: int = 128,
    draft_subdir: str = "outputs/llama-3.1-8b-draft-llama32-1b/final",
):
    gpu = gpu.lower()
    if gpu not in {"a100", "h100"}:
        raise ValueError("gpu must be 'a100' or 'h100'")

    if mode == "generate":
        fn = generate_a100 if gpu == "a100" else generate_h100
        print(fn.remote(prompt, max_tokens=max_tokens))
        return

    if mode == "throughput":
        fn = bench_throughput_a100 if gpu == "a100" else bench_throughput_h100
        print(json.dumps(fn.remote(), indent=2))
        return

    if mode == "teacher-data":
        fn = distill_generate_teacher_a100 if gpu == "a100" else distill_generate_teacher_h100
        print(json.dumps(fn.remote(), indent=2))
        return

    if mode == "train-distill":
        fn = distill_train_a100 if gpu == "a100" else distill_train_h100
        print(json.dumps(fn.remote(), indent=2))
        return

    if mode == "eval-acceptance":
        fn = eval_acceptance_a100 if gpu == "a100" else eval_acceptance_h100
        print(json.dumps(fn.remote(draft_subdir=draft_subdir), indent=2))
        return

    if mode == "bench-spec":
        fn = bench_speculative_a100 if gpu == "a100" else bench_speculative_h100
        print(json.dumps(fn.remote(draft_path=_output_path(draft_subdir)), indent=2))
        return

    if mode == "humaneval":
        fn = eval_humaneval_a100 if gpu == "a100" else eval_humaneval_h100
        print(json.dumps(fn.remote(draft_subdir=draft_subdir), indent=2))
        return

    if mode == "small-draft-smoke":
        fn = small_draft_smoke_a100 if gpu == "a100" else small_draft_smoke_h100
        print(json.dumps(fn.remote(), indent=2))
        return

    if mode == "small-draft":
        fn = small_draft_a100 if gpu == "a100" else small_draft_h100
        print(json.dumps(fn.remote(), indent=2))
        return

    raise ValueError(
        "mode must be one of: generate, throughput, teacher-data, "
        "train-distill, eval-acceptance, bench-spec, humaneval, "
        "small-draft-smoke, small-draft"
    )
