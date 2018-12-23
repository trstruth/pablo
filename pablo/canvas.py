# coding: utf-8

"""A module for the Canvas where the image where be created

analogous to the environment in a reinforcement learning problem - accepts actions,
performs transformations to the images stored as members, thus updating the environment
and returning a reward scalar
"""

import numpy as np

import imageio
import os
import gym

from gym import spaces, logger
from numpy import linalg as LA

class Canvas(gym.Env):

    def __init__(self,
                 target_image_filename,
                 images_directory=os.path.abspath(os.path.join(__file__,"../../","images/"))):

        self.target_image_filename = target_image_filename
        self.generated_image = None
        self.images_directory = images_directory
        self.emoji_directory = os.path.join(self.images_directory, 'emojis')
        self.num_emojis = len(os.listdir(self.emoji_directory))
        self.error = float("inf")

        self._load_target_image_from_file(self.target_image_filename)
        self._init_generated_image()
        self.target_image_h, self.target_image_w, _ = self.target_image.shape

        self.action_space = spaces.Dict({
            "y": spaces.Discrete(self.target_image_h),
            "x": spaces.Discrete(self.target_image_w),
            "emoji_selection": spaces.Discrete(self.num_emojis)
        })

    def reset(self):
        """Resets the state of the canvas by:
          - reinitializing the generated image
          - returning the inital observation

        Returns: observation (object): the initialized
            generated image, a numpy array who's shape matches
            that of self.target_image
        """
        self._init_generated_image()
        return self.generated_image

    def step(self):
        """
        step returns four values. These are:

            * observation (object): an environment-specific object representing your observation of the environment. For 
              example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board
              game.
            * reward (float): amount of reward achieved by the previous action. The scale varies between environments, but
              the goal is always to increase your total reward.
            * done (boolean): whether it’s time to reset the environment again. Most (but not all) tasks are divided up into 
              well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole
              tipped too far, or you lost your last life.)
            * info (dict): diagnostic information useful for debugging. It can sometimes be useful for learning (for example,
              it might contain the raw probabilities behind the environment’s last state change). However, official evaluations
              of your agent are not allowed to use this for learning.
        """
        pass


    def _load_target_image_from_file(self, filename):
        """Load the image file stored at <filename>
        filename will be appended to self.images_directory to create the path where the image is stored

        Args:
            filename (str): The name of the file 

        Returns:
            None
        """
        target_image_filepath = os.path.join(self.images_directory, filename)
        try:
            self.target_image = imageio.imread(target_image_filepath, format='PNG-FI')
        except IOError:
            print('There was an error opening the file {}'.format(target_image_filepath))

    def _write_generated_image_to_file(self, filename):
        """Write self.generated_image to <filename>
        
        Args:
            filename (str): The name of the file 

        Returns:
            None
        """
        generated_image_filepath = os.path.join(self.images_directory, filename)
        self.target_image = imageio.imwrite(generated_image_filepath, self.generated_image, format='PNG-FI')

    def _init_generated_image(self):
        """Initialize the generated image in a random manner

        Returns:
            None
        """
        assert self.target_image is not None
        self.generated_image = np.full(self.target_image.shape, 255, dtype=np.uint8)

    def _calculate_error(self):
        """ Calculate the error between self.target_image and self.generated_image

        Returns:
            Int: the error scalar
        """
        assert self.target_image.shape == self.generated_image.shape
        return LA.norm(self.target_image - self.generated_image)
