"""Applying saved merge rules must rebuild the original grey image."""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLBACKEND", "Agg")

import torch

from elsner_apply_rules import apply_one_rule, apply_rules, initial_state, paint_original, sequence_length


def _notebook_pass(unique, classes, class_a, class_b, offset_x, offset_y, new_id):
    """Direct reading of the notebook's replacement loops."""
    used = set()
    height, width = unique.shape
    for x in range(height):
        for y in range(width):
            if int(unique[x, y]) in used:
                continue
            for n_x, n_y in ((x + 1, y), (x, y + 1)):
                if n_x >= height or n_y >= width:
                    continue
                if int(unique[n_x, n_y]) in used:
                    continue
                if int(unique[x, y]) == int(unique[n_x, n_y]):
                    continue
                if int(classes[x, y]) != class_a or int(classes[n_x, n_y]) != class_b:
                    continue
                a_pixels = torch.nonzero(unique == unique[x, y])
                b_pixels = torch.nonzero(unique == unique[n_x, n_y])
                anchor_a = a_pixels[0]
                anchor_b = b_pixels[0]
                if int(anchor_a[0]) - int(anchor_b[0]) != offset_x:
                    continue
                if int(anchor_a[1]) - int(anchor_b[1]) != offset_y:
                    continue
                a_unique = unique[x, y].clone()
                for index in range(b_pixels.shape[0]):
                    unique[int(b_pixels[index, 0]), int(b_pixels[index, 1])] = a_unique
                    classes[int(b_pixels[index, 0]), int(b_pixels[index, 1])] = new_id
                a_pixels = torch.nonzero(unique == a_unique)
                for index in range(a_pixels.shape[0]):
                    classes[int(a_pixels[index, 0]), int(a_pixels[index, 1])] = new_id
                used.add(int(unique[x, y]))


class ApplyRulesTest(unittest.TestCase):
    def test_one_merge_paints_the_original_image(self):
        image = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        # Right-hand neighbour: anchor offset is (0, -1).
        rules = [[1, 2, 0, -1]]
        unique, classes = apply_rules(image, rules, first_new_id=256)
        painted = paint_original(image, unique)
        self.assertTrue(torch.equal(painted, image))
        self.assertEqual(sequence_length(unique), 3)
        self.assertEqual(int(classes[0, 0]), 256)
        self.assertEqual(int(classes[0, 1]), 256)
        self.assertEqual(int(unique[0, 0]), int(unique[0, 1]))

    def test_unmatched_rule_leaves_every_pixel(self):
        image = torch.tensor([[8, 9], [8, 9]], dtype=torch.long)
        rules = [[1, 2, 0, -1]]
        unique, _classes = apply_rules(image, rules)
        painted = paint_original(image, unique)
        self.assertTrue(torch.equal(painted, image))
        self.assertEqual(sequence_length(unique), 4)

    def test_second_rule_uses_the_new_class_id(self):
        image = torch.tensor([[1, 2, 5]], dtype=torch.long)
        # The second rule joins the new token to the pixel on its right.
        # Anchors are (0, 0) and (0, 2), so the offset is (0, -2).
        rules = [[1, 2, 0, -1], [256, 5, 0, -2]]
        unique, classes = apply_rules(image, rules, first_new_id=256)
        painted = paint_original(image, unique)
        self.assertTrue(torch.equal(painted, image))
        self.assertEqual(sequence_length(unique), 1)
        self.assertEqual(int(classes[0, 0]), 257)
        self.assertEqual(int(classes[0, 2]), 257)

    def test_fast_pass_matches_the_notebook_loops(self):
        torch.manual_seed(0)
        image = torch.randint(0, 6, (5, 5))
        rules = [[1, 2, 0, -1], [2, 3, -1, 0], [256, 4, 0, -1]]
        left_unique, left_classes = initial_state(image)
        right_unique, right_classes = initial_state(image)
        for index, rule in enumerate(rules):
            class_a, class_b, offset_x, offset_y = rule
            apply_one_rule(left_unique, left_classes, class_a, class_b, offset_x, offset_y, 256 + index)
            _notebook_pass(right_unique, right_classes, class_a, class_b, offset_x, offset_y, 256 + index)
        self.assertTrue(torch.equal(left_unique, right_unique))
        self.assertTrue(torch.equal(left_classes, right_classes))
        self.assertTrue(torch.equal(paint_original(image, left_unique), image))


if __name__ == "__main__":
    unittest.main()
