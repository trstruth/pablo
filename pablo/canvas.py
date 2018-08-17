import os
import numpy as np
from numpy import linalg as LA
import imageio

class Canvas(object):

    def __init__(self,
                 images_directory=os.path.abspath(os.path.join(__file__,"../../","images/"))):

        self.target_image = None
        self.generated_image = None
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
