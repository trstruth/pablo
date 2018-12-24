# Pablo 
An artistic deep reinforcement learning experiment that tries to teach an AI to draw.

The Canvas class inherits from the generic OpenAI Gym.Env class and implements the methods needed to train a deep rl agent.  The agent selects an emoji before positioning, scaling, and orienting and pasting it on a generated image.  This is done iteratively, and the agent receives rewards that indicate whether it is approaching a better approximation of the image.

### TODO
* Determine a better ordering for the emojis.  How should they be structured such that it is more intuitive for the agent to find the color/shape of the emoji it will use next?
* Determine a good reward metric.  It must be a notion of distance between target image and generated image.  Pixelwise comparison is too sensitive to minor changes.  Perhaps downsample and calculate difference beyond a threshold?  Use a neural net's certainty estimation as the reward?
