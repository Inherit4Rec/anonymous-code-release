"""Small, data-free checks for the retained D2D conversion and simple training loop."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.utils.data import DataLoader

from growth import get_growth_parameters, grow_uniformer_in_place
from model import PerTokenSwiGLU
from trainer import evaluate, save_weights, train_epoch


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.per_token_ffn = PerTokenSwiGLU(2, 4, 6, 0.1)


class TinyFIM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fim_layers = nn.ModuleList([Layer()])


class TinyRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, batch):
        return self.linear(batch["x"])

    def get_dense_params(self):
        return list(self.parameters())


def main() -> None:
    torch.manual_seed(7)
    fim = TinyFIM()
    x = torch.randn(3, 2, 4)
    optimizer = torch.optim.AdamW(fim.parameters(), lr=1e-4)
    old = fim.fim_layers[0].per_token_ffn
    old(x).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    old_ids = {id(p) for p in old.parameters()}
    old_moments = {p: optimizer.state[p]["exp_avg"].clone() for p in old.parameters()}
    torch.manual_seed(12)
    expected = old(x).detach()
    grow_uniformer_in_place(fim, 18, seed=2026)
    torch.manual_seed(12)
    actual = fim.fim_layers[0].per_token_ffn(x).detach()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert old_ids.issubset({id(p) for p in fim.parameters()})
    assert len(get_growth_parameters(fim)) == 2
    assert all(torch.equal(optimizer.state[p]["exp_avg"], moment)
               for p, moment in old_moments.items())

    examples = [
        {"x": torch.tensor([float(i % 2), 1.0]),
         "labels": torch.tensor([float(i % 2)]),
         "user_id": torch.tensor(i // 2)}
        for i in range(8)
    ]
    loader = DataLoader(examples, batch_size=4)
    ranker = TinyRanker()
    dense = torch.optim.AdamW(ranker.parameters(), lr=1e-3)
    result = train_epoch(ranker, loader, device=torch.device("cpu"),
                         dense_optimizer=dense, sparse_optimizer=None, use_bf16=False)
    assert result["updates"] == 2
    assert "gauc" in evaluate(ranker, loader, device=torch.device("cpu"),
                              task_names=["click"], use_bf16=False)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "weights.pt"
        save_weights(ranker, path)
        assert set(torch.load(path, weights_only=True)) == set(ranker.state_dict())
    print("D2D smoke test passed")


if __name__ == "__main__":
    main()
