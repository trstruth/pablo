import cProfile
from pablo.agents import NaiveAgent

n = NaiveAgent()

n.run()
# cProfile.run('n.run()', sort='cumtime')
