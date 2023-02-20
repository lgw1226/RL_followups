# Lunar Lander
**Considered success if the reward exceeds over 200**

## Arguments
    import gymnasium as gym
    env = gym.make(
        "LunarLander-v2",
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    )

## Action space
Integer from 0 to 3
- 0: Do nothing
- 1: fire left orientation engine
- 2: fire main engine
- 3: fire right orientation engine

## Observation space
`numpy.ndarray` of shape `(8,)`

    [x, y, v_x, v_y, a, w, contact_left, contact_right]

## Episode terminates when
1. the lander crashes
2. the lander gets outside of the viewport
3. the lander is not awake (doesn't move and doesn't collide)