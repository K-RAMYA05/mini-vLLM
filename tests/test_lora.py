import torch
import torch.nn as nn

from mini_vllm.lora import apply_lora_adapters


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.linear(x)


def test_apply_lora_adapters_loads_and_switches_adapter(tmp_path):
    model = _TinyModel()
    adapter_path = tmp_path / "adapter_model.pt"
    torch.save(
        {
            "base_model.model.linear.lora_A.weight": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "base_model.model.linear.lora_B.weight": torch.tensor([[1.0], [0.0], [0.0]]),
        },
        adapter_path,
    )
    (tmp_path / "adapter_config.json").write_text('{"r": 1, "lora_alpha": 1}')

    manager = apply_lora_adapters(model, [f"demo={adapter_path}"])
    x = torch.tensor([[2.0, 3.0, 4.0, 5.0]])

    base = model(x)
    manager.set_active_adapter("demo")
    adapted = model(x)
    manager.set_active_adapter(None)
    reset = model(x)

    assert "demo" in manager.available_adapters
    torch.testing.assert_close(base, reset)
    assert adapted[0, 0].item() != base[0, 0].item()
