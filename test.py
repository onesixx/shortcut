from collections import deque

# Create a deque with a maximum length of 6
my_deque = deque(maxlen=6)
my_deque


# Add elements to the deque
my_deque.append(1)
my_deque.append(2)
my_deque.append(3)
# Current state of the deque: deque([1, 2, 3], maxlen=6)

# Add more elements, exceeding the maximum length
my_deque.append(4)
my_deque.append(5)
my_deque.append(6)
my_deque.append(7)
# Current state of the deque: deque([3, 4, 5, 6, 7], maxlen=6)




import gymnasium
# Create the LunarLander environment
env = gymnasium.make('LunarLander-v2', render_mode='human')
# Reset the environment
env.reset()
# Render the environment (a separate window should appear)
env.render()
# Optionally, interact with the environment, take actions, etc.
# Close the environment once done
env.close()
