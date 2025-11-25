import torch
import torch.nn as nn
import random
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitVsGlowTestCase

class TestModel(JitVsGlowTestCase):
    @classmethod
    def setUpClass(cls):
        super(TestModel, cls).setUpClass()
        cls.enable_glow_fusion_passes()

    def test_model(self):
        M = 1024
        N = 8192
        K = 8192
        model = Model(N, K)
        model.train()
        inputs = torch.rand(M, N)
        self.run_glow_fuser_test(inputs, model, fusion_patterns="glow")

if __name__ == "__main__":
    run_tests()