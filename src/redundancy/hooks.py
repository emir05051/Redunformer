import torch


class HeadPruningHook:
    def __init__(self, head_indices, head_dim):
        self.head_indices = tuple(sorted(set(head_indices)))
        self.head_dim = head_dim

    def __call__(self, module, args):
        hidden_states = args[0]
        feature_mask = torch.ones(
            hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype
        )
        for head in self.head_indices:
            start = head * self.head_dim
            feature_mask[start : start + self.head_dim] = 0
        return (hidden_states * feature_mask, *args[1:])


class RandomHeadPruningHook(HeadPruningHook):
    pass


class HeadImportanceMaskHook:
    def __init__(self, mask: torch.Tensor, head_dim: int):
        self.mask = mask
        self.head_dim = head_dim

    def __call__(self, module, args):
        hidden_states = args[0]
        expanded_mask = self.mask.repeat_interleave(self.head_dim).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        expanded_mask = expanded_mask.view(*([1] * (hidden_states.dim() - 1)), -1)
        return (hidden_states * expanded_mask, *args[1:])
