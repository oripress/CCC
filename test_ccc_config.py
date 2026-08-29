import unittest

from ccc_config import baseline_name, dataset_name, stream_config


class CCCConfigTest(unittest.TestCase):
    def test_integer_baseline_has_no_decimal_suffix(self):
        self.assertEqual(baseline_name(20.0), "20")
        self.assertEqual(
            dataset_name(20.0, 1000, 44),
            "baseline_20_transition+speed_1000_seed_44",
        )

    def test_stream_config_matches_generation_order(self):
        self.assertEqual(stream_config(0), (1000, 43))
        self.assertEqual(stream_config(3), (1000, 44))
        self.assertEqual(stream_config(8), (5000, 45))


if __name__ == "__main__":
    unittest.main()
