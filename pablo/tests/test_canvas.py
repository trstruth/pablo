import unittest
import pablo
import numpy as np

class CanvasTests(unittest.TestCase):

    def setUp(self):
        self.c = pablo.Canvas('pablo.png')

    def test_init(self):
        self.assertIsNotNone(self.c.target_image)
        self.assertIsNotNone(self.c.generated_image)

    def test_calculate_error_identity(self):
        self.c.target_image = np.zeros([300, 300, 0], dtype=int)
        self.c.generated_image = np.zeros([300, 300, 0], dtype=int)

        self.assertEqual(self.c._calculate_error(), 0)

    def test_calculate_error_nonnegative(self):
        self.c.target_image = np.empty([300, 300, 0], dtype=int)
        self.c.generated_image = np.empty([300, 300, 0], dtype=int)

        self.assertEqual(self.c._calculate_error(), 0)

    def test_gym_reset(self):
        initial_observation = self.c.reset()

        self.assertIsNotNone(self.c.generated_image)
        self.assertEqual(self.c.generated_image.shape, self.c.target_image.shape)
        self.assertIsNotNone(initial_observation)

    def test_gym_step(self):
        pass

    def test_gym_render(self):
        pass

    def test_gym_close(self):
        pass

if __name__ == '__main__':
    unittest.main()
