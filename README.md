# Pablo

Target | Generated
--- | ---
![basq_before](images/basquiat_skull.jpg) | ![basq_after](images/basquiat_skull-100k@15.png)

An artistic deep reinforcement learning experiment that tries to teach an AI to "draw".

### Overview

The Canvas class inherits from the generic OpenAI Gym.Env class and implements the methods needed to train a deep rl agent.  The agent selects an emoji before positioning, scaling, and orienting and pasting it on a generated image.  This is done iteratively, and the agent receives rewards that indicate whether it is approaching a better approximation of the image.

### 

### TODO
* Determine a good reward/loss metric.  It must be a notion of distance between target image and generated image.  Pixelwise comparison is too sensitive to minor changes.  Perhaps downsample and calculate difference beyond a threshold?  Use a neural net's certainty estimation as the reward?  Adrien proposed 3 part loss:
  1. distance between new image and target image
  2. penalty for the size of each emoji placed (the smaller the higher, so if k is size then 1/k^2. Otherwise it could just cheat and make everything tiny to minimize error
  3. penalty for the amount of emojis the agent has used so far, to emphasize minimalism and not rote copying.
* Emoji's are still blurry in generated image.  Maybe fix the generated image to some large resolution, then wrap the x,y placement code in something that deals in ratios to fix this?
* Find a way to strip out the background and only have the face be emojified
* Faster iterations - probably a lot of clever tricks we could to to pull out more performance from canvas
