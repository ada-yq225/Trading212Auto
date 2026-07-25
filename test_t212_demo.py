import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import t212_demo


class ClientTests(unittest.TestCase):
    def test_client_is_locked_to_demo(self):
        self.assertEqual(t212_demo.DEMO_BASE_URL, "https://demo.trading212.com/api/v0")

    @patch("urllib.request.urlopen")
    def test_market_sell_uses_negative_quantity(self, urlopen):
        response = MagicMock()
        response.read.return_value = b'{"id": 42}'
        response.__enter__.return_value = response
        urlopen.return_value = response
        client = t212_demo.Trading212DemoClient("key", "secret")
        client.market_order("AAPL_US_EQ", -0.5, False)
        request = urlopen.call_args.args[0]
        self.assertTrue(request.full_url.startswith(t212_demo.DEMO_BASE_URL))
        self.assertEqual(json.loads(request.data), {
            "ticker": "AAPL_US_EQ",
            "quantity": -0.5,
            "extendedHours": False,
        })

    def test_dotenv_does_not_override_existing_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("T212_API_KEY=file-key\nT212_API_SECRET=file-secret\n", encoding="utf-8")
            with patch.dict(os.environ, {"T212_API_KEY": "existing"}, clear=True):
                t212_demo.load_dotenv(path)
                self.assertEqual(os.environ["T212_API_KEY"], "existing")
                self.assertEqual(os.environ["T212_API_SECRET"], "file-secret")

    def test_positive_number_rejects_invalid_values(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.assertRaises(Exception):
                t212_demo.positive_number(value)


if __name__ == "__main__":
    unittest.main()
