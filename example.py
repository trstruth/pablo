import cProfile
from pablo.agents import NaiveAgent

n = NaiveAgent('pablo.png', iters=300000)

n.run()
# cProfile.run('n.run()', sort='cumtime')

n.canvas._write_generated_image_to_file('pablo-out.png')
