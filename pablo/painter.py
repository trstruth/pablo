"""A module for the Painter who selects actions and creates the image

analogous to the agent in a reinforcement learning problem - selects
actions, and submits them to the environment in return for a
reward scalar and updated environment
"""

import pablo

class Painter(object):

    def __init__(self, num_iters=10000, num_emojis=1000):
        self.num_iters = num_iters
        self.num_emojis = num_emojis
        self.canvas = pablo.Canvas()

    def choose_action(self):
        """Use policy to select action

        Returns:
            None
        """

        # TODO: implement logic for selecting an action or "iterating" on the image
        # could consist of replacing an emoji, or repositioning an existing one
        # ultimately, the painter is aiming to make the value of the error lower
        # Consider implementing an action class that represents the action chosen
        raise NotImplementedError

    def take_action(self, action):
        """takes the action <action>

        Args:
            action (str): The action object to take 

        Returns:
            None
        """

        # TODO: apply action to self.canvas
        raise NotImplementedError

    def step(self):
        """Load the image file stored at <filename>
        filename will be appended to self.images_directory to create the path where the image is stored

        Args:
            filename (str): The name of the file 

        Returns:
            None
        """

        # TODO: implement one iteration: choose action, take action, recalculate error
        chosen_action = self.choose_action()
        self.take_action(chosen_action)
        self.error = self.calculate_error()
