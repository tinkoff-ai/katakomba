import torch
import numpy as np

# Code adapted from huggingface implementation:
# https://github.com/huggingface/transformers/blob/61f79b2986005dba96f1257aaff74693c7fbdbfd/src/transformers/models/perceiver/modeling_perceiver.py#L2795


def _build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
    def _linspace(n_xels_per_dim):
        return torch.linspace(start=output_range[0], end=output_range[1], steps=n_xels_per_dim, dtype=torch.float32)

    dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    array_index_grid = torch.meshgrid(*dim_ranges, indexing="ij")

    return torch.stack(array_index_grid, dim=-1)


def _build_spatial_positions(index_dims):
    pos = _build_linear_positions(index_dims)
    pos = pos[None].expand((1,) + pos.shape)
    pos = torch.reshape(pos, [1, np.prod(index_dims), -1])

    return pos


def _generate_fourier_features(pos, num_bands, max_resolution=(224, 224)):
    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.stack(
        [torch.linspace(start=min_freq, end=res / 2, steps=num_bands) for res in max_resolution], dim=0
    )
    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos[0, :, :][:, :, None] * freq_bands[None, :, :]
    per_pos_features = torch.reshape(per_pos_features, [-1, np.prod(per_pos_features.shape[1:])])

    # Output is size [n, 2 * d * num_bands]
    per_pos_features = torch.cat(
        [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
    )
    # Concatenate the raw input positions.
    per_pos_features = torch.cat([pos, per_pos_features.expand(1, -1, -1)], dim=-1)

    return per_pos_features


def generate_fourier_pos_encoding(index_dims, num_bands):
    pos = _build_spatial_positions(index_dims)
    fourier_pos_enc = _generate_fourier_features(
        pos,
        num_bands=num_bands,
        max_resolution=index_dims,
    )
    return fourier_pos_enc