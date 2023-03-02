# Bipedal Walker

## Arguments
    import gymnasium as gym
    env = gym.make(
        "BipedalWalker-v3",
        hardcore=False
    )

## Observation space
`numpy.ndarray` of shape `(24,)`

- hull angle speed
- angular velocity
- horizontal speed
- vertical speed
- position of joints
- joints angular speed
- legs contact with ground
- 10 lider ranglfinder measurements

## Action space
`numpy.ndarray` of shape `(4,)`

motor speed values in the `[-1, 1]` range for each of the 4 joints at both hips and knees

## Reward
Reward is given for moving forward, totaling 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque costs a small amount of points. A more optimal agent will get a better score.

## Episode terminates when
- the hull gets in contact with the ground
- the walker exceeds the right end of the terrain length