class Painter(object):

    def __init__(self, num_iters=10000, num_emojis=1000):
        self.num_iters = num_iters
        self.num_emojis = num_emojis

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
