import pablo
env = pablo.Canvas('pablo.png')
for i_episode in range(20):
    observation = env.reset()
    info = {}
    for t in range(500):
        print info
        env.render()
        action = env.action_space.sample()
        try:
            observation, reward, done, info = env.step(action)
        except ValueError:
            print action
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
