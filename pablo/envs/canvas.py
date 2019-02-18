# coding: utf-8

"""A module for the Canvas where the image will be created

analogous to the environment in a reinforcement learning problem - accepts actions,
performs transformations to the images stored as members, thus updating the environment
and returning a reward scalar
"""

import numpy as np

import os
import gym

from PIL import Image
from gym import spaces, logger
from numpy import linalg as LA
from skimage.measure import compare_ssim

class Canvas(gym.Env):
        
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 50
    }

    def __init__(self,
                 target_image_filename='pablo.png',
                 images_directory=os.path.abspath(os.path.join(__file__,'../../../','images/'))):

        self.target_image_filename = target_image_filename
        self.generated_image = None
        self.images_directory = images_directory
        self.emoji_directory = os.path.join(self.images_directory, 'emojis')
        self.num_available_emojis = len(os.listdir(self.emoji_directory)) - 1
        self.emoji_count = 0
        self.max_emojis = 300
        self.similarity = None
        self.similarity_threshold = 0.1

        self.viewer = None
        self._load_target_image_from_file(self.target_image_filename)
        self.reset()
        self.target_image_w, self.target_image_h = self.target_image.size

        self.action_space = spaces.Dict({
            'y': spaces.Discrete(self.target_image_h),
            'x': spaces.Discrete(self.target_image_w),
            'emoji': spaces.Discrete(self.num_available_emojis),
            'scale': spaces.Box(low=1, high=5, shape=(1,), dtype=np.float32),
            'rotation': spaces.Discrete(360)
        })

        self.observation_space = spaces.Dict({
            'target': spaces.Box(0, 255, shape=(self.target_image_h, self.target_image_w, 3), dtype=np.uint8),
            'generated': spaces.Box(0, 255, shape=(self.target_image_h, self.target_image_w, 3), dtype=np.uint8)
        })

        self.action_space.n = len(self.action_space.spaces.items())


    def reset(self):
        """Resets the state of the canvas by:
          - reinitializing the generated image
          - returning the inital observation

        Returns: observation (object): The observation object that is described by
        self.observation_space
        """
        assert self.target_image is not None
        self.emoji_count = 0
        self.generated_image = Image.new('RGBA', self.target_image.size, (255, 255, 255))
        self.similarity = self._calculate_mssim()
        return {
            'target': np.array(self.target_image.convert('RGB')),
            'generated': np.array(self.generated_image.convert('RGB'))
        }


    def step(self, action):
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
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # Place emoji as described by the action
        selected_emoji = Image.open('{}/{}.png'.format(self.emoji_directory, action['emoji']))
        coordinate = (action['x'], action['y'])
        scale = action['scale']
        cur_size = selected_emoji.size
        scaled_size = (cur_size[0] / scale, cur_size[1] / scale)
        selected_emoji = selected_emoji.resize(scaled_size)
        selected_emoji = selected_emoji.rotate(action['rotation'], expand=1)

        self.generated_image.paste(selected_emoji, coordinate, selected_emoji)

        # construct the observation object
        observation = {
            'target': np.array(self.target_image.convert('RGB')),
            'generated': np.array(self.generated_image.convert('RGB'))
        }

        # TODO: calculate reward
        new_similarity = self._calculate_mssim()
        reward = new_similarity - self.similarity
        self.similarity = new_similarity

        # increment emoji count and set done flag
        self.emoji_count += 1
        done = (self.emoji_count >= self.max_emojis) or (self.similarity < self.similarity_threshold)

        # construct diagnositic info dict
        info = {
            'selected_emoji': action['emoji'],
            'position': (action['x'], action['y']),
            'emoji_count': self.emoji_count,
            'mssim': self.similarity,
        }
        
        return observation, reward, done, info
    

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(np.array(self.generated_image.convert('RGB')))
        return self.viewer.isopen


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
            self.target_image = Image.open(target_image_filepath)
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
        self.generated_image.save(generated_image_filepath)


    def _calculate_error(self):
        """ Calculate the error between self.target_image and self.generated_image

        Returns:
            Int: the error scalar
        """
        assert self.target_image.size == self.generated_image.size
        return LA.norm(self.target_image - self.generated_image)

    def _calculate_mssim(self):
        """ Calculate the mean structural similarity between self.target_image and self.generated_image

        Returns:
            Float: The mean structural similarity over the image.
        """
        assert self.target_image.size == self.generated_image.size
        thumb_size = (128, 128)
        target_thumb = self.target_image.copy().convert('RGB') 
        generated_thumb = self.generated_image.copy().convert('RGB') 
        target_thumb.thumbnail(thumb_size, resample=Image.ANTIALIAS)
        generated_thumb.thumbnail(thumb_size, resample=Image.ANTIALIAS)

        return compare_ssim(np.array(target_thumb), np.array(generated_thumb), multichannel=True)
        


