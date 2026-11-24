import unittest
from unittest.mock import patch
from src.data import load_catalog

class CatalogTests(unittest.TestCase):
    def test_empty_catalog_directory(self):
        with patch('src.data.DESCRIPTIONS_FILE') as jf, patch('src.data.IMAGE_DIR') as idir:
            jf.exists.return_value=True; jf.read_text.return_value='{}'; idir.glob.return_value=[]
            self.assertEqual(load_catalog(), [])

if __name__ == '__main__': unittest.main()
