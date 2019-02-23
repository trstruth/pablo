import cProfile

from pablo.agents.naive.naive_agent import Naive

n = Naive()

cProfile.run('n.run()', sort='cumtime')
# n.run()

