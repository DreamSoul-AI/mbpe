from io import BytesIO
import unittest

import grid_bpe_file as archive
from grid_bpe_study import reference_bytes


class GridBPEFileTests(unittest.TestCase):
    def check_round_trip(self, spec, directions, packets, samples, codec, width=0):
        lengths = archive.code_lengths(spec, directions, packets) if codec == "huffman" else None
        handle = BytesIO()
        written = archive.write_archive(handle, spec, directions, packets, codec, lengths, width)
        self.assertEqual(written["total_bytes"], len(handle.getvalue()))
        self.assertEqual(archive.read_archive(BytesIO(handle.getvalue())),
                         (spec, directions, packets, codec, lengths, width, written))
        output = BytesIO()
        self.assertEqual(archive.read_archive(BytesIO(handle.getvalue()), output)[-1], written)
        self.assertEqual(output.getvalue(), reference_bytes(samples))
        return handle.getvalue(), written

    def test_mixed_rectangular_scans_and_fixed_width(self):
        spec = archive.Spec("adaptive", 2, (0, 1), (((0, 1),), ((0, 0),)))
        packets = [archive.Packet((2, 3), None, (), tokens) for tokens in ((2, 0, 2, 1), (2, 1, 1, 0, 1))]
        values = [0, 1, 0, 0, 1, 1]
        samples = [((2, 3), dict(zip(archive.grid_points((2, 3)), values)))] * 2
        for codec, width in (("zlib", 9), ("huffman", 0)):
            with self.subTest(codec=codec):
                _, written = self.check_round_trip(spec, [0, 1], packets, samples, codec, width)
                self.assertEqual(written["direction_bytes"], 1)
                self.assertEqual(written["widths"], [9, 9] if codec == "zlib" else [])

    def test_sparse_geometry_with_multiple_unknown_integers(self):
        spec = archive.Spec("nd", 3, (0,), (((0, 0, 0, 1, -1),),))
        domain = ((0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 1))
        large = 1 << 90
        packet = archive.Packet((2, 2, 2), domain, (-large, large), (1, 3, 2))
        samples = [(packet.shape, dict(zip(domain, (0, 0, large, -large))))]
        for codec in archive.CODECS:
            with self.subTest(codec=codec):
                blob, cost = self.check_round_trip(spec, [0], [packet], samples, codec)
                self.assertEqual(cost["domain_bytes"], 1)
                self.assertEqual(cost["geometry_bytes"], 3)
                for malformed in (blob[:-1], blob + b"\x00"):
                    with self.assertRaises(ValueError):
                        archive.read_archive(BytesIO(malformed), BytesIO())
