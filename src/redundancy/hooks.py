class HeadPruningHook:

    def __init__(self, head_indices, head_dim):
        self.head_indices = head_indices
        self.head_dim = head_dim

    def __call__(self, module, args):
        hidden_states = args[0].clone()

        # Mask with zeros for the specified heads
        for head in self.head_indices:
            start = head * self.head_dim
            end = (head + 1) * self.head_dim
            hidden_states[..., start:end] = 0

        return (hidden_states,)


class RandomHeadPruningHook(HeadPruningHook):
    def __init__(self, head_indices, head_dim):
        super(RandomHeadPruningHook, self).__init__(head_indices, head_dim)
