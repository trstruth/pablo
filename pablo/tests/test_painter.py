import unittest
import pablo

class MyFirstTests(unittest.TestCase):

    def test_init_painter(self):
        p = pablo.Painter()
        self.assertIsNotNone(p.num_iters)
        self.assertIsNotNone(p.num_emojis)

    def test_init_generated_image(self):
        p = pablo.Painter()
        p.init_generated_image()
        self.assertIsNotNone(p.generated_image)

    def test_load_target_image(self):
        p = pablo.Painter()
        p.load_target_image('sample_filename.jpg')
        self.assertIsNotNone(p.target_image)

    def test_calculate_error_identity(self):
        p = pablo.Painter()
        image1 = 1
        image2 = 1

        p.target_image = image1
        p.generated_image = image2

        self.assertEqual(p.calculate_error(), 0)

if __name__ == '__main__':
    unittest.main()
