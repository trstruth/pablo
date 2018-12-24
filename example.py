import numpy as np
import pablo 
import time

c = pablo.Canvas('pablo.png')
c.reset()

for _ in range(100):
    c.step(c.action_space.sample())
    c.render()

time.sleep(3)
c._write_generated_image_to_file('pablo-out.png')
