
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from infer_bridge_states import capture_bridge_states, collect_images, parse_args, save_png
from models.sb_model import SBModel
from options.test_options import TestOptions


class RecordingGenerator(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.inputs = []
        self.outputs = []

    def forward(self, x, time_idx, z, direction=0):
        if self.fail:
            raise RuntimeError("test generator failure")
        self.inputs.append(x.detach().clone())
        y = 0.7 * x + 0.03 * z.mean(dim=1).reshape(-1, 1, 1, 1)
        self.outputs.append(y.detach().clone())
        return y


def make_model(fail=False):
    model = SBModel.__new__(SBModel)
    model.opt = SimpleNamespace(num_timesteps=5, tau=0.01, ngf=2)
    model.real_A = torch.linspace(-1, 1, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    model.netG = RecordingGenerator(fail=fail)
    return model


class BridgeStateExportTests(unittest.TestCase):
    def test_actual_inputs_and_same_final_prediction(self):
        plain = make_model()
        observed = make_model()
        original = observed.real_A.clone()
        torch.manual_seed(1234)
        plain.inference()
        torch.manual_seed(1234)
        states, final = capture_bridge_states(observed)

        self.assertEqual(sorted(states), [0, 1, 2, 3, 4])
        self.assertTrue(torch.equal(final, plain.fake_B))
        self.assertTrue(torch.equal(states[0], original))
        self.assertTrue(torch.equal(observed.real_A, original))
        for step, state in states.items():
            self.assertTrue(torch.equal(state, observed.netG.inputs[step]))
        self.assertFalse(torch.equal(states[1], observed.netG.outputs[0]))
        self.assertFalse(torch.equal(states[4], final))
        self.assertEqual(len(observed.netG._forward_pre_hooks), 0)
        states[0].zero_()
        self.assertTrue(torch.equal(observed.real_A, original))

    def test_hook_is_removed_after_failure(self):
        model = make_model(fail=True)
        with self.assertRaisesRegex(RuntimeError, "test generator failure"):
            capture_bridge_states(model)
        self.assertEqual(len(model.netG._forward_pre_hooks), 0)

    def test_png_display_clipping_does_not_change_bridge_tensor(self):
        tensor = torch.tensor([-2.0, 0.0, 2.0]).reshape(1, 3, 1, 1)
        original = tensor.clone()
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "x1" / "same_name.png"
            save_png(tensor, 0, out)
            with Image.open(out) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.getpixel((0, 0)), (0, 127, 255))
        self.assertTrue(torch.equal(tensor, original))

    def test_duplicate_png_names_are_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.png").touch()
            (root / "sample.jpg").touch()
            with self.assertRaisesRegex(RuntimeError, "Duplicate output name"):
                collect_images(root, False, 0)

    def test_model_option_is_not_batch_size_abbreviation(self):
        argv = ["infer_bridge_states.py", "--input_dir", "inputs",
                "--output_dir", "outputs", "--model_batch_size", "8",
                "--model", "sb", "--gpu_ids", "0"]
        with patch.object(sys, "argv", argv), patch.object(
            TestOptions, "parse", return_value=SimpleNamespace()
        ):
            dopt, _ = parse_args()
            self.assertEqual(dopt.model_batch_size, 8)
            self.assertEqual(sys.argv[1:], ["--model", "sb", "--gpu_ids", "0"])


if __name__ == "__main__":
    unittest.main()
