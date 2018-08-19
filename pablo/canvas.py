"""A module for the Canvas where the image where be created

analogous to the environment in a reinforcement learning problem - accepts actions,
performs transformations to the images stored as members, thus updating the environment
and returning a reward scalar
"""

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
        """Load the image file stored at <filename>
        filename will be appended to self.images_directory to create the path where the image is stored

        Args:
            filename (str): The name of the file 

        Returns:
            None
        """
        target_image_filepath = os.path.join(self.images_directory, filename)
        self.target_image = imageio.imread(target_image_filepath, format='PNG-FI')

    def write_generated_image_to_file(self, filename):
        """Write self.generated_image to <filename>
        
        Args:
            filename (str): The name of the file 

        Returns:
            None
        """
        generated_image_filepath = os.path.join(self.images_directory, filename)
        self.target_image = imageio.imwrite(generated_image_filepath, self.generated_image, format='PNG-FI')

    def init_generated_image(self):
        """Initialize the generated image in a random manner

        Returns:
            None
        """
        assert self.target_image is not None
        self.generated_image = np.empty_like(self.target_image)
        # TODO: add randomly selected emojis

    def calculate_error(self):
        """ Calculate the error between self.target_image and self.generated_image

        Returns:
            Int: the error scalar
        """
        assert self.target_image.shape == self.generated_image.shape
        return LA.norm(self.target_image - self.generated_image)
