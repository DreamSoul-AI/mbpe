from io import BytesIO
import unittest
import zlib

from bpe_block_file import (
    BLOCK, HEADER, FileSpec, ORDERS, pack_symbols, read_archive, read_token_archive,
    write_archive, write_huffman_archive,
)
from canonical_huffman import CanonicalHuffman, lengths_from_validation


class BPEBlockFileTests(unittest.TestCase):
    def round_trip(self, spec, sequences, expected):
        archive = BytesIO()
        written = write_archive(archive, spec, sequences)
        self.assertEqual(written["total_bytes"], len(archive.getvalue()))
        archive.seek(0)
        decoded = BytesIO()
        self.assertEqual(read_archive(archive, decoded), written)
        self.assertEqual(decoded.getvalue(), expected)
        return archive.getvalue(), written

    def test_raw_matches_zlib_on_packed_bits(self):
        blob, cost = self.round_trip(FileSpec("raw", 2, 4), [[0, 1, 2, 3]], bytes([0, 1, 2, 3]))
        payload = zlib.compress(b"\x1b", level=9)
        self.assertEqual(blob[HEADER.size + BLOCK.size:], payload)
        self.assertEqual(cost["payload_sizes"], [len(payload)])

    def test_one_dimensional_scans_restore_positions(self):
        expected = bytes([0, 1, 2, 3])
        for mode, sequence in zip(ORDERS, ([0, 1, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2], [0, 2, 3, 1])):
            with self.subTest(mode=mode):
                spec = FileSpec(mode, 2, 4, (tuple(sequence[:2]),))
                self.round_trip(spec, [[4, *sequence[2:]]], expected)

    def test_two_dimensional_rules_keep_both_signed_offsets(self):
        rules = ((0, 1, 1, -1), (2, 3, 1, 1))
        # The first anchor is (0,0). A fill using the negative-x rule starts at (0,1).
        expected = bytes([2, 0, 1, 3])
        _, cost = self.round_trip(FileSpec("2d", 2, 4, rules), [[5, 4]], expected)
        self.assertEqual(cost["geometry_bytes"], 4)

    def test_nine_bit_tokens_and_multiple_blocks(self):
        spec = FileSpec("row", 2, 4, ((0, 1),) * 256)
        self.round_trip(spec, [[259, 2, 3], [3, 2, 259]], bytes([0, 1, 2, 3, 3, 2, 0, 1]))

    def test_malformed_archive_fails(self):
        spec = FileSpec("row", 2, 2, ((0, 1),))
        archive = BytesIO()
        write_archive(archive, spec, [[2, 2]])
        valid = archive.getvalue()
        payload = zlib.compress(pack_symbols([2], spec.width, spec.symbols))
        too_short = valid[:HEADER.size + 1] + BLOCK.pack(1, len(payload)) + payload
        for blob in (valid[:-1], valid + b"\x00", too_short):
            with self.subTest(blob=blob), self.assertRaises(ValueError):
                read_archive(BytesIO(blob), BytesIO())

    def test_huffman_has_canonical_codes_and_covers_unseen_ids(self):
        code = CanonicalHuffman(bytes([1, 2, 3, 3]))
        payload, bits = code.encode([0, 1, 2, 3])
        self.assertEqual((payload, bits), (b"\x5b\x80", 9))
        self.assertEqual(code.decode(payload, bits, 4), [0, 1, 2, 3])
        lengths = lengths_from_validation([[1, 1], [0]], 4)
        self.assertEqual(lengths, lengths_from_validation([[0, 1, 1]], 4))
        code = CanonicalHuffman(lengths)
        payload, bits = code.encode([2, 3])
        self.assertEqual(code.decode(payload, bits, 2), [2, 3])

    def test_huffman_archive_preserves_grammar_tokens_and_file_cost(self):
        spec = FileSpec("2d", 2, 4, ((0, 1, 1, -1), (2, 3, 1, 1)))
        sequences = [[5, 4], [0, 0, 0, 0]]
        lengths = lengths_from_validation(sequences[:1], spec.symbols)
        archive = BytesIO()
        cost = write_huffman_archive(archive, spec, sequences, lengths)
        self.assertEqual(cost["code_table_bytes"], spec.symbols)
        self.assertEqual(cost["total_bytes"], len(archive.getvalue()))
        self.assertEqual(8 * cost["payload_bytes"], cost["payload_bits"] + cost["padding_bits"])
        archive.seek(0)
        self.assertEqual(read_token_archive(archive), (spec, sequences, cost))
        archive.seek(0)
        output = BytesIO()
        self.assertEqual(read_archive(archive, output), cost)
        self.assertEqual(output.getvalue(), bytes([2, 0, 1, 3, 0, 0, 0, 0]))

    def test_huffman_rejects_malformed_codes_and_nonzero_padding(self):
        for lengths in ([1, 1, 1], [1, 3, 3, 3], [0, 1]):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                CanonicalHuffman(lengths)
        code = CanonicalHuffman(bytes([1, 2, 3, 3]))
        with self.assertRaises(ValueError):
            code.decode(b"\x01", 1, 1)
        with self.assertRaises(ValueError):
            code.decode(b"\xc0", 2, 1)
