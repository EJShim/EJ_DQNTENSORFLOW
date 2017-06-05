
from E_PolicyGradientBrain import *

# Main FUnction
env = gym.make('CartPole-v0')
brain = E_PolicyGradientBrain(4, 2)

num_epiosdes = 500
max_step = 999


#Run Training
for i in range(num_epiosdes):

    state = env.reset()

    brain.Reset()
    transitions = []
    epsilon = 1.0

    done = False

    rewardsAll = 0.0

    for j in range(max_step):

        action = brain.Forward(state, epsilon)
        state1, reward, done, info = env.step(action)
        transitions.append((state, action, reward))

        rewardsAll += reward

        if done : break


        state = state1

    # #Backward
    brain.Backward(transitions)


    print(rewardsAll)


for i in range(num_epiosdes):

    state = env.reset()
    brain.Reset()
    # transitions = []

    done = False

    rewardsAll = 0.0

    for j in range(max_step):
        action = brain.Forward(state)
        state1, reward, done, info = env.step(action)
        env.render()
        # transitions.append((state, action, reward))

        rewardsAll += reward

        if done : break


        state = state1

    # # #Backward
    # brain.Backward(transitions)


    print(rewardsAll)
