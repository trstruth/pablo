import unittest
import pablo
import numpy as np

class CanvasTests(unittest.TestCase):

    def test_init(self):
        c = pablo.Canvas('pablo.png')
        self.assertIsNotNone(c.target_image)
        self.assertIsNotNone(c.generated_image)

    def test_calculate_error_identity(self):
        c = pablo.Canvas('pablo.png')
        c.target_image = np.zeros([300, 300, 0], dtype=int)
        c.generated_image = np.zeros([300, 300, 0], dtype=int)

        self.assertEqual(c.calculate_error(), 0)

    def test_calculate_error_nonnegative(self):
        c = pablo.Canvas('pablo.png')
        c.target_image = np.empty([300, 300, 0], dtype=int)
        c.generated_image = np.empty([300, 300, 0], dtype=int)

        self.assertEqual(c.calculate_error(), 0)

    def test_gym_reset(self):
        c = pablo.Canvas('pablo.png')
        initial_observation = c.reset()

        self.assertIsNotNone(c.generated_image)
        self.assertEqual(c.generated_image.shape, c.target_image.shape)
        self.assertIsNotNone(initial_observation)

    def test_gym_step(self):
        pass

    def test_gym_render(self):
        pass

    def test_gym_close(self):
        pass

if __name__ == '__main__':
    unittest.main()
