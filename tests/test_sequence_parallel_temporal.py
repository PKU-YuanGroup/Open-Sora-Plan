import sys
import types
from types import SimpleNamespace

import pytest
import torch

from opensora.utils.temporal_utils import (
    find_closest_y,
    latent_frame_count,
    pad_temporal_frames,
    temporal_parallel_multiple,
)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (29, 29),
        (32, 29),
        (61, 61),
        (65, 61),
        (93, 93),
    ],
)
def test_dataset_bucket_length_is_sequence_parallel_safe(requested, expected):
    selected = find_closest_y(requested, vae_stride_t=4, model_ds_t=4)
    assert selected == expected
    assert latent_frame_count(selected, 4) % 4 == 0


def test_dataset_bucket_accounts_for_temporal_patch_size():
    # A latent length divisible by SP alone can still leave a partial temporal
    # patch.  The shared LCM rule avoids that second source of padding.
    multiple = temporal_parallel_multiple(3, 2)
    selected = find_closest_y(65, vae_stride_t=4, model_ds_t=multiple)
    assert selected == 45
    assert latent_frame_count(selected, 4) % multiple == 0


def test_padding_rounds_raw_frames_to_a_vae_aligned_sp_multiple():
    padded = pad_temporal_frames(66, ae_stride_t=4, patch_size_t=2, sp_size=4)
    assert padded == 77
    assert padded >= 66
    assert latent_frame_count(padded, 4) == 20
    assert latent_frame_count(padded, 4) % temporal_parallel_multiple(4, 2) == 0


def test_image_only_padding_keeps_single_frame_for_sp_disabled_path():
    assert (
        pad_temporal_frames(
            1, ae_stride_t=4, patch_size_t=1, sp_size=8, preserve_single_frame=True
        )
        == 1
    )


def test_collate_pads_latent_time_for_sequence_parallelism(monkeypatch):
    # dataset_utils imports decord at module import time, but collation itself
    # is pure CPU logic.  A tiny module stub keeps this regression test light.
    monkeypatch.setitem(sys.modules, "decord", types.ModuleType("decord"))
    from opensora.utils.dataset_utils import Collate

    args = SimpleNamespace(
        train_batch_size=2,
        group_data=False,
        force_resolution=False,
        max_height=8,
        max_width=8,
        ae_stride=1,
        ae_stride_t=4,
        patch_size=1,
        patch_size_t=1,
        num_frames=65,
        sp_size=4,
    )
    collate = Collate(args)
    batch = [
        {
            "pixel_values": torch.ones(3, 65, 8, 8),
            "input_ids_1": torch.ones(1, 2, dtype=torch.long),
            "cond_mask_1": torch.ones(1, 2, dtype=torch.long),
            "input_ids_2": None,
            "cond_mask_2": None,
        },
        {
            "pixel_values": torch.ones(3, 61, 8, 8),
            "input_ids_1": torch.ones(1, 2, dtype=torch.long),
            "cond_mask_1": torch.ones(1, 2, dtype=torch.long),
            "input_ids_2": None,
            "cond_mask_2": None,
        },
    ]

    padded, attention_mask, *_ = collate(batch)
    assert padded.shape == (2, 3, 77, 8, 8)
    assert attention_mask.shape == (2, 20, 8, 8)
    assert torch.all(attention_mask[0, :17])
    assert not torch.any(attention_mask[0, 17:])
    assert torch.all(attention_mask[1, :16])
    assert not torch.any(attention_mask[1, 16:])


def test_collate_keeps_image_batch_one_frame(monkeypatch):
    monkeypatch.setitem(sys.modules, "decord", types.ModuleType("decord"))
    from opensora.utils.dataset_utils import Collate

    args = SimpleNamespace(
        train_batch_size=1,
        group_data=True,
        force_resolution=False,
        max_height=8,
        max_width=8,
        ae_stride=1,
        ae_stride_t=4,
        patch_size=1,
        patch_size_t=1,
        num_frames=1,
        sp_size=8,
    )
    batch = [
        {
            "pixel_values": torch.ones(3, 1, 8, 8),
            "input_ids_1": torch.ones(1, 2, dtype=torch.long),
            "cond_mask_1": torch.ones(1, 2, dtype=torch.long),
            "input_ids_2": None,
            "cond_mask_2": None,
        }
    ]

    padded, attention_mask, *_ = Collate(args)(batch)
    assert padded.shape[2] == 1
    assert attention_mask.shape[1] == 1
    assert torch.all(attention_mask)
