from unittest import TestCase

from make_dataset import parse_rank_specification


class TestParse_rank_specification(TestCase):
    def test_parse_rank_specification(self):
        self.assertEqual(parse_rank_specification('1'), set([1]))
        self.assertEqual(parse_rank_specification('1,2,3'), set([1, 2, 3]))
        self.assertEqual(parse_rank_specification('1..3'), set([1, 2, 3]))
        self.assertEqual(parse_rank_specification('-1..3'), set([0, 1, 2, 3, -1]))
        self.assertEqual(parse_rank_specification('-1..3,5..6'), set([0, 1, 2, 3, 5, 6, -1]))
        self.assertEqual(parse_rank_specification('5..5,1..4'), set([1, 2, 3, 4, 5]))
        self.assertEqual(parse_rank_specification(''), set([None]))
        self.assertEqual(parse_rank_specification('1..3,'), set([1, 2, 3, None]))
        self.assertEqual(parse_rank_specification(','), set([None]))


if __name__ == '__main__':
    import unittest
    unittest.main()
