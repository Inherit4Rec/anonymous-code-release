"""Small, data-free checks for the retained D2S partition and sparse forward."""

import torch

from model import PerTokenSparseMoE, PerTokenSwiGLU
from moe import CoActivationStatistics, build_coactivation_partitions


def main() -> None:
    torch.manual_seed(7)
    collector = CoActivationStatistics(num_tokens=2, hidden_dim=6, top_k=2)
    hidden = torch.randn(8, 2, 6)
    down = torch.randn(2, 6, 4)
    collector.update(hidden, down)
    partitions, report = build_coactivation_partitions(
        [collector], num_routed_experts=2, num_starts=4,
        refine_rounds=2, seed=7, coactivation_tau=10,
    )
    assert partitions[0].shape == (2, 3, 2)
    assert report["layers"][0]["samples"] == 8
    assert all(sorted(row.flatten().tolist()) == list(range(6)) for row in partitions[0])

    dense = PerTokenSwiGLU(2, 4, 6, 0.0)
    optimizer = torch.optim.AdamW(dense.parameters(), lr=1e-3)
    x = torch.randn(3, 2, 4)
    dense(x).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected = dense(x).detach()
    input_moment = optimizer.state[dense.in_ffn]["exp_avg"].clone()
    output_moment = optimizer.state[dense.out_ffn]["exp_avg"].clone()
    old_ids = {id(p) for p in dense.parameters()}
    moe = PerTokenSparseMoE.from_dense(
        dense, source_channel_indices=partitions[0],
        num_routed_experts=2, optimizer=optimizer,
    )
    assert old_ids.issubset({id(p) for p in moe.parameters()})
    for token in range(2):
        order = partitions[0][token].flatten()
        expected_input = torch.cat((
            input_moment[token, :, :6].index_select(1, order),
            input_moment[token, :, 6:].index_select(1, order)), dim=1)
        torch.testing.assert_close(optimizer.state[moe.in_ffn]["exp_avg"][token],
                                   expected_input)
        torch.testing.assert_close(optimizer.state[moe.out_ffn]["exp_avg"][token],
                                   output_moment[token].index_select(0, order))
    torch.testing.assert_close(moe.dense_equivalent_forward(x), expected)
    output = moe(x)
    assert output.shape == expected.shape
    assert moe.auxiliary_loss() is not None
    (output.square().mean() + 0.01 * moe.auxiliary_loss()).backward()
    assert moe.router.grad is not None and torch.isfinite(moe.router.grad).all()
    print("D2S smoke test passed")


if __name__ == "__main__":
    main()
