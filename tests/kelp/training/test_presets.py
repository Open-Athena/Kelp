# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for model/resource presets."""

import pytest

from kelp.training.presets import PRESETS, ModelPreset, get_preset


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_registered_preset_resolves_and_self_names(name):
    """get_preset(name) returns a ModelPreset whose name matches the registry
    key -- catches a function/registry mismatch or a typo'd registration."""
    preset = get_preset(name)
    assert isinstance(preset, ModelPreset)
    assert preset.name == name
    assert preset.batch_size > 0


def test_tpu_vet_targets_cheap_slice_at_mid_scale():
    """The data-scaling vet preset sits between the toy and 1B presets: a
    ~300M model on the cheap v6e-4 slice."""
    preset = get_preset("tpu_vet")
    assert preset.resource.device.variant == "v6e-4"
    assert preset.config.hidden_dim == 768
