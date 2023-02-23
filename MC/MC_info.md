# Mountain Car (Continuous)

## Arguments
    import gymnasium as gym
    env = gym.make(
        "MountainCarContinuous-v0"
    )

## Observation space
`numpy.ndarray` of shape `(8,)`

    [x, y, v_x, v_y, a, w, contact_left, contact_right]

## Action space
`numpy.ndarray` of shape `(1,)`
- The action is clipped in the range [-1, 1] and multiplied by a power of 0.0015

## Reward
A negative reward of $-0.1*action^2$ is received at each timestep to penalise for taking actions of large magnitude. If the mountain car reaches the goal then a positive reward of +100 is added to the negative reward for that timestep.

## Episode terminates when
The position of the car is greater than or equal to 0.45 (the goal position)

## Episode truncates when
The length of the episode is 999