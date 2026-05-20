"""Feature datasets for layer-wise compensation training."""

from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """Dataset that yields (index, feature) pairs from a hidden states tensor."""

    def __init__(self, inps, device):
        self.inps = inps
        self.device = device

    def __len__(self):
        return self.inps.size(0)

    def __getitem__(self, idx):
        return idx, self.inps[idx].to(self.device)


class DoubleFeatureDataset(Dataset):
    """Dataset yielding two aligned feature streams (e.g. pruned vs. dense hidden states)."""

    def __init__(self, err_stream_state, std_stream_state, device):
        self.err_stream_state = err_stream_state
        self.std_stream_state = std_stream_state
        self.device = device

    def __len__(self):
        return self.err_stream_state.size(0)

    def __getitem__(self, idx):
        return (
            idx,
            self.err_stream_state[idx].to(self.device),
            self.std_stream_state[idx].to(self.device),
        )


class ThreeFeatureDataset(Dataset):
    """Dataset yielding error stream + standard stream + MLP/Attention split streams."""

    def __init__(self, err_stream_state, std_stream_state, std_stream_state_mlp,
                 std_stream_state_attn, device):
        self.err_stream_state = err_stream_state
        self.std_stream_state = std_stream_state
        self.std_stream_state_mlp = std_stream_state_mlp
        self.std_stream_state_attn = std_stream_state_attn
        self.device = device

    def __len__(self):
        return self.err_stream_state.size(0)

    def __getitem__(self, idx):
        return (
            idx,
            self.err_stream_state[idx].to(self.device),
            self.std_stream_state[idx].to(self.device),
            self.std_stream_state_mlp[idx].to(self.device),
            self.std_stream_state_attn[idx].to(self.device),
        )
