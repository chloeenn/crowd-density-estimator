"""Architecture guards: a wrong VGG slice index or dilation shows up here immediately."""

import pytest
import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16

from crowd.models.csrnet import CSRNet

# Derived by hand from the architecture (not read off the model):
#   frontend: 3->64->64->128->128->256x3->512x3 convs (3x3)            = 7,635,264
#   backend : 512->512 x3, 512->256, 256->128, 128->64 (3x3 dilated)    = 8,628,160
#   head    : 64->1 (1x1)                                               =        65
EXPECTED_PARAMS = 7_635_264 + 8_628_160 + 65  # 16,263,489


@pytest.fixture(scope="module")
def model():
    return CSRNet(pretrained=False)


def test_parameter_count(model):
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMS


def test_frontend_is_first_ten_convs_at_stride_8(model):
    convs = [m for m in model.frontend if isinstance(m, nn.Conv2d)]
    pools = [m for m in model.frontend if isinstance(m, nn.MaxPool2d)]
    assert len(convs) == 10 and len(pools) == 3  # 2^3 = stride 8
    assert convs[-1].out_channels == 512


def test_backend_is_six_dilated_convs(model):
    convs = [m for m in model.backend if isinstance(m, nn.Conv2d)]
    assert [c.out_channels for c in convs] == [512, 512, 512, 256, 128, 64]
    assert all(c.dilation == (2, 2) for c in convs)


@pytest.mark.parametrize(("h", "w"), [(64, 96), (256, 256), (72, 40)])
def test_output_is_one_eighth_resolution(model, h, w):
    with torch.no_grad():
        out = model(torch.zeros(1, 3, h, w))
    assert out.shape == (1, 1, h // 8, w // 8)


def test_init_of_new_layers(model):
    w = model.backend[0].weight
    assert abs(w.std().item() - 0.01) < 0.002
    assert torch.count_nonzero(model.output.bias) == 0


def test_freezing_frontend_only_freezes_frontend():
    m = CSRNet(pretrained=False)
    m.set_frontend_trainable(False)
    assert not any(p.requires_grad for p in m.frontend.parameters())
    assert all(p.requires_grad for p in m.backend.parameters())
    assert all(p.requires_grad for p in m.output.parameters())
    out = m(torch.randn(1, 3, 32, 32))
    out.sum().backward()
    assert m.frontend[0].weight.grad is None and m.output.weight.grad is not None


def test_pretrained_frontend_matches_torchvision_vgg16():
    """Downloads the ~528 MB VGG-16 weights on first run, then uses the torch cache."""
    ref = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23]
    m = CSRNet(pretrained=True)
    for a, b in zip(m.frontend.state_dict().values(), ref.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    # ...but the backend is NOT pretrained
    assert abs(m.backend[0].weight.std().item() - 0.01) < 0.002
