from pablo.agents import Agent
from tqdm import tqdm
import random
import numpy as np

class NaiveAgent(Agent):

    def __init__(self,
                 target_image_filename,
                 iters=100000,
                 emoji_size=25):

        super().__init__(target_image_filename)

        #setup vars
        self.iters = iters
        self.emoji_size = emoji_size

    def run(self):
        
        for i in tqdm(range(self.iters)):
            location = self._sampler(self.canvas.target_image)
            pixel = self.canvas.target_image.getpixel(location)
            x, y = location
            r, g, b = pixel
            
            selected_action = {
                'x': x,
                'y': y,
                'r': r,
                'g': g,
                'b': b,
                'scale': self.emoji_size,
                'rotation': 0
            }

            self.canvas.step(selected_action)
    
    def _sampler(self, image):
        """Picks out points on image."""
        len_x, len_y = (image.size[0], image.size[1])
        x, y = random.randint(0, len_x-1), random.randint(0, len_y-1)
        return x, y
