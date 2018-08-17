import unittest
import pablo

class PainterTests(unittest.TestCase):

    def test_init_painter(self):
        p = pablo.Painter()
        self.assertIsNotNone(p.num_iters)
        self.assertIsNotNone(p.num_emojis)

if __name__ == '__main__':
    unittest.main()
