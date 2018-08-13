import unittest
import pablo

class MyFirstTests(unittest.TestCase):

    def test_hello(self):
        p = pablo.Painter()
        self.assertEqual(p.hello_world(), 'hello world')

if __name__ == '__main__':
    unittest.main()
