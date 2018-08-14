import os
import numpy as np
from numpy import linalg as LA
import imageio

class Painter(object):

    def __init__(self,
                 num_iters=10000,
                 num_emojis=1000,
                 images_directory=os.path.abspath(os.path.join(__file__,"../../","images/"))):

        self.target_image = None
        self.generated_image = None
        self.num_iters = num_iters
        self.num_emojis = num_emojis
        self.images_directory = images_directory
        self.error = float("inf")

    def load_target_image_from_file(self, filename):
        target_image_filepath = os.path.join(self.images_directory, filename)
        self.target_image = imageio.imread(target_image_filepath, format='JPEG-FI')

    def write_generated_image_to_file(self, filename):
        generated_image_filepath = os.path.join(self.images_directory, filename)
        self.target_image = imageio.imwrite(generated_image_filepath, self.generated_image, format='JPEG-FI')

    def init_generated_image(self):
        assert self.target_image is not None
        self.generated_image = np.empty_like(self.target_image)
        # TODO: add randomly selected emojis

    def calculate_error(self):
        assert self.target_image.shape == self.generated_image.shape
        return LA.norm(self.target_image - self.generated_image)

    def choose_action(self):
        # TODO: implement logic for selecting an action or "iterating" on the image
        # could consist of replacing an emoji, or repositioning an existing one
        # ultimately, the painter is aiming to make the value of the error lower
        raise NotImplementedError

    def take_action(self, a):
        # TODO: apply action a to self.generated_image
        raise NotImplementedError

    def step(self):
        # TODO: implement one iteration: choose action, take action, recalculate error
        chosen_action = self.choose_action()
        self.take_action(chosen_action)
        self.error = self.calculate_error()

    def create_image(self, filename):
        # TODO: step self.num_iters times and generate image
        self.load_target_image_from_file(filename)
        self.init_generated_image()

        '''
        for _ in range(self.num_iters):
            self.step()
        '''

        output_filename = filename.split('.')
        output_filename[0] = output_filename[0] + '_out'
        output_filename = '.'.join(output_filename)
        self.write_generated_image_to_file(output_filename)
