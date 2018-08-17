import unittest
import pablo
import numpy as np

class CanvasTests(unittest.TestCase):

    def test_init_generated_image(self):
        c = pablo.Canvas()
        c.load_target_image_from_file('pablo.jpg')
        c.init_generated_image()
        self.assertIsNotNone(c.generated_image)

    def test_load_target_image_from_file(self):
        c = pablo.Canvas()
        c.load_target_image_from_file('pablo.jpg')
        self.assertIsNotNone(c.target_image)

    def test_calculate_error_identity(self):
        c = pablo.Canvas()
        c.target_image = np.zeros([300, 300, 0], dtype=int)
        c.generated_image = np.zeros([300, 300, 0], dtype=int)

        self.assertEqual(c.calculate_error(), 0)

    def test_calculate_error_nonnegative(self):
        c = pablo.Canvas()
        c.target_image = np.empty([300, 300, 0], dtype=int)
        c.generated_image = np.empty([300, 300, 0], dtype=int)

        self.assertEqual(c.calculate_error(), 0)

if __name__ == '__main__':
    unittest.main()
