class Painter(object):

    def __init__(self, num_iters=10000, num_emojis=1000):
        self.target_image = None
        self.generated_image = None
        self.num_iters = num_iters
        self.num_emojis = num_emojis
        self.error = float("inf")

    def load_target_image(self, filepath):
        # TODO: method to load the target image from a given filepath
        raise NotImplementedError

    def init_generated_image(self):
        # TODO: initialize generated image with randomly placed emojis
        raise NotImplementedError

    def calculate_error(self):
        # TODO: calculate error between self.target_image and self.generated_image.  Could be simple pixel diff or more sophisticated method
        raise NotImplementedError 

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
        self.calculate_error()

    def create_image(self, filepath):
        # TODO: step self.num_iters times and generate image
        self.load_target_image(filepath)
        self.init_generated_image()

        for _ in range(self.num_iters):
            self.step()
