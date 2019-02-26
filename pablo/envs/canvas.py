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
from scipy.spatial import KDTree
from skimage.measure import compare_ssim

class Canvas(gym.Env):
        
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 50
    }

    def __init__(self,
                 target_image_filename,
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

        # These values control the quality of the output image
        self.min_dim = 4000
        self.image_res_ratio = 1

        self.emoji_KDT = self._construct_emoji_KDTree()
        self.emoji_cache = self._construct_emoji_cache()

        self.viewer = None
        self._load_target_image_from_file(self.target_image_filename)
        self.reset()
        self.target_image_w, self.target_image_h = self.target_image.size

        self.action_space = spaces.Dict({
            'y': spaces.Discrete(self.target_image_h),
            'x': spaces.Discrete(self.target_image_w),
            'r': spaces.Discrete(256),
            'g': spaces.Discrete(256),
            'b': spaces.Discrete(256),
            'scale': spaces.Discrete(1000),
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
        self._init_generated_image()
        self.similarity = 0
        return {
            'target': self.target_image,
            'generated': self.generated_image
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

        index = self._find_nearest_emoji_index(action['r'], action['g'], action['b'])
        selected_emoji = self._get_emoji(index)

        # Place emoji as described by the action
        coordinate = (
            int(action['x']*self.image_res_ratio),
            int(action['y']*self.image_res_ratio)
        )
        scale = action['scale']
        cur_size = selected_emoji.size
        scaled_size = (int(cur_size[0] / scale), int(cur_size[1] / scale))
        selected_emoji = selected_emoji.resize(scaled_size, Image.ANTIALIAS)
        selected_emoji = selected_emoji.rotate(action['rotation'], expand=1)

        self.generated_image.paste(selected_emoji, coordinate, selected_emoji)

        # construct the observation object
        observation = {
            'target': self.target_image,
            'generated': self.generated_image
        }

        # TODO: calculate reward
        # new_similarity = self._calculate_mssim()
        new_similarity = 0
        reward = new_similarity - self.similarity
        self.similarity = new_similarity

        # increment emoji count and set done flag
        self.emoji_count += 1
        done = (self.emoji_count >= self.max_emojis) # or (self.similarity < self.similarity_threshold)

        # construct diagnositic info dict
        info = {
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
            self.target_image = Image.open(target_image_filepath).convert('RGB')
        except IOError:
            print('There was an error opening the file {}'.format(target_image_filepath))
            self.target_image = None


    def _init_generated_image(self):
        """Initialize the generated image
        in order to avoid unnecessary downsampling of the emojis, we create a high resolution canvas
        to paste them on.  The aspect ratio of of the generated image should be the same as that of
        the target image. For the generated image, the min(width, height) should equal some parameter
        set for the canvas.

        Returns:
            None
        """
        assert self.target_image is not None

        # divide 5000 by each dim, store ratios in scale_ratios
        scale_ratios = [self.min_dim/dim for dim in self.target_image.size]
        # the min(width, height) will create the larger res ratio
        self.image_res_ratio = max(scale_ratios)

        scaled_size = tuple([int(self.image_res_ratio*dim) for dim in self.target_image.size])
        self.generated_image = Image.new('RGBA', scaled_size, (255, 255, 255))


    def _write_generated_image_to_file(self, filename):
        """Write self.generated_image to <filename>
        
        Args:
            filename (str): The name of the file 

        Returns:
            None
        """
        generated_image_filepath = os.path.join(self.images_directory, filename)
        self.generated_image.save(generated_image_filepath, quality=95)


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
            Float: The mean structural similarity over the image
        """
        assert self.target_image.size == self.generated_image.size
        thumb_size = (128, 128)
        target_thumb = self.target_image.copy().convert('RGB') 
        generated_thumb = self.generated_image.copy().convert('RGB') 
        target_thumb.thumbnail(thumb_size, resample=Image.ANTIALIAS)
        generated_thumb.thumbnail(thumb_size, resample=Image.ANTIALIAS)

        return compare_ssim(np.array(target_thumb), np.array(generated_thumb), multichannel=True)
        

    def _construct_emoji_KDTree(self):
        """ The rgb color space can be visualized in 3 dimensions, with
        the intensity of each color along each dim.  We can achieve
        logarithmic lookup performance by indexing each emoji in a KDTree.
        This method constructs such a tree.  We iterate through each of
        the emojis, construct a list of those values, then use them to 
        construct a KDTree.

        Returns:
            KDTree: The KDTree indexing each of the emojis
        """
        avg_rgb_list = np.zeros((self.num_available_emojis, 3))

        for i in range(self.num_available_emojis):
            emoji = Image.open('{}/{}.png'.format(self.emoji_directory, i))
            average_rgb = self._get_average_rgb(emoji)
            avg_rgb_list[i, :] = average_rgb   

        return KDTree(avg_rgb_list)
    

    def _construct_emoji_cache(self):
        """ Instead of loading each emoji from disk each time, on first load
        we cache the emoji in a dictionary.  On each lookup, we first check
        to see if the emoji is loaded in the cache, and if so we can retrieve
        it from memory as opposed to disk
        """
        return {i: None for i in range(self.num_available_emojis)}


    def _get_average_rgb(self, image):
        """ Calculate the average rgb values given an image.  Ignore pixels
        that have a 0 alpha value

        Args:
            image (Image): the input image

        Returns:
            np.array: Array with [r, g, b] values
        """
        im = np.array(image)
        h, w, d = im.shape
        im_vector = im.reshape((h*w, d))

        im_mask = im_vector[:,3] != 0
        average_rgb = np.average(im_vector, axis=0, weights=im_mask)[:3]

        return average_rgb


    def _find_nearest_emoji_index(self, r, g, b):
        """Find and return the emoji with average rgb values closest to supplied rgb
        
        Args:
            r (int): The red component
            g (int): The green component
            b (int): The blue component

        Returns:
            int: the index of the nearest emoji
        """
        dist, emoji_index = self.emoji_KDT.query([r, g, b])
        return emoji_index

    def _get_emoji(self, index):
        """ Return the emoji that corresponds with the given index.
        Check cache for the emoji first.  If it doesnt exist, make a record of it
        otherwise, return the emoji in the cache.

        Args:
            index (int): The index of the desired emoji

        Returns:
            Image: the Image object that represents the emoji
        """
        if self.emoji_cache[index] is None:
            image = Image.open('{}/{}.png'.format(self.emoji_directory, index)) 
            self.emoji_cache[index] = image
            return image
        else:
            return self.emoji_cache[index]

