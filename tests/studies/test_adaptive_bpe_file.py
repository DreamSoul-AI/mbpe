from io import BytesIO
import unittest

from adaptive_bpe_file import read_archive, read_token_archive, write_archive
from bpe_block_file import BLOCK, HEADER, FileSpec
from canonical_huffman import lengths_from_validation


class AdaptiveBPEFileTests(unittest.TestCase):
    specs = (FileSpec("row", 2, 4, ((0, 1),)), FileSpec("column", 2, 4, ((0, 2),)))
    directions = [0, 1, 0]
    sequences = [[4, 2, 3], [4, 1, 3], [4, 3, 2]]

    def test_mixed_scans_restore_grids_and_count_all_file_parts(self):
        tables = tuple(lengths_from_validation([seq], spec.symbols)
                       for seq, spec in zip(self.sequences, self.specs))
        for lengths in (None, tables):
            with self.subTest(lengths=lengths):
                archive = BytesIO()
                cost = write_archive(archive, self.specs, self.directions, self.sequences, lengths)
                blob = archive.getvalue()
                self.assertEqual(read_token_archive(BytesIO(blob)),
                                 (self.specs, lengths, self.directions, self.sequences, cost))
                decoded = BytesIO()
                self.assertEqual(read_archive(BytesIO(blob), decoded), cost)
                self.assertEqual(decoded.getvalue(), bytes([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 3, 2]))
                table_bytes = 2 * self.specs[0].symbols if lengths else 0
                self.assertEqual(cost["code_table_bytes"], table_bytes)
                self.assertEqual(cost["rule_symbol_bytes"], 2)
                self.assertEqual(cost["direction_bytes"], 1)
                self.assertEqual(cost["actual_rules"], 2)
                self.assertEqual(blob[HEADER.size + 2 + table_bytes], 0b01000000)
                self.assertEqual(cost["total_bytes"], len(blob))
                self.assertEqual(len(blob), HEADER.size + 2 + table_bytes + 1 + 3 * BLOCK.size + cost["payload_bytes"])

    def test_rejects_bad_direction_padding_truncation_and_trailing_bytes(self):
        archive = BytesIO()
        write_archive(archive, self.specs, self.directions, self.sequences)
        blob = archive.getvalue()
        bad_padding = bytearray(blob)
        bad_padding[HEADER.size + 2] |= 1
        for malformed in (blob[:-1], blob + b"\x00", bad_padding):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                read_archive(BytesIO(malformed), BytesIO())
