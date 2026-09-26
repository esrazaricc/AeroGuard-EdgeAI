import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from main import DualInputFusionEncoder
from decode import FusionDecoder


def main():
    encoder = DualInputFusionEncoder(latent_dim=128).eval()
    decoder = FusionDecoder(latent_dim=128).eval()
    x = torch.randn(1, 4, 640, 640)
    with torch.no_grad():
        z = encoder(x)
        y = decoder(z)
    assert z.shape == (1, 128), z.shape
    assert y.shape == (1, 4, 640, 640), y.shape
    print("SMOKE TEST PASS", tuple(z.shape), tuple(y.shape))


if __name__ == "__main__":
    main()
