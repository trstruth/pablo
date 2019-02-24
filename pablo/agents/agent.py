from pablo.envs import Canvas


class Agent():
    
    def __init__(self, target_image_filename):
        self.canvas = Canvas(target_image_filename)


    def setup(self):
        raise NotImplementedError

    
    def run(self):
        raise NotImplementedError
