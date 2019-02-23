from pablo.envs import Canvas

c = Canvas('pablo.png')

class Agent():
    
    def __init__(self, target_image_filename='pablo.png'):
        self.canvas = Canvas(target_image_filename=target_image_filename)


    def setup(self):
        raise NotImplementedError

    
    def run(self):
        raise NotImplementedError
