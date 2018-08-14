import unittest
import pablo
import numpy as np

class PainterTests(unittest.TestCase):

    def test_init_painter(self):
        p = pablo.Painter()
        self.assertIsNotNone(p.num_iters)
        self.assertIsNotNone(p.num_emojis)

    def test_init_generated_image(self):
        p = pablo.Painter()
        p.load_target_image_from_file('pablo.jpg')
        p.init_generated_image()
        self.assertIsNotNone(p.generated_image)

    def test_load_target_image_from_file(self):
        p = pablo.Painter()
        p.load_target_image_from_file('pablo.jpg')
        self.assertIsNotNone(p.target_image)

    def test_calculate_error_identity(self):
        p = pablo.Painter()
        p.target_image = np.zeros([300, 300, 0], dtype=int)
        p.generated_image = np.zeros([300, 300, 0], dtype=int)

        self.assertEqual(p.calculate_error(), 0)

    def test_calculate_error_nonnegative(self):
        p = pablo.Painter()
        p.target_image = np.empty([300, 300, 0], dtype=int)
        p.generated_image = np.empty([300, 300, 0], dtype=int)

        self.assertGreaterEqual(p.calculate_error(), 0)

if __name__ == '__main__':
    unittest.main()
