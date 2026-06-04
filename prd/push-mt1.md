**Task:** push-mt1 (`push-v3`)

Push the puck to a goal. Puck and goal positions are randomized each episode.

**Optimization hints (non-exhaustive):**

- Strengthen distance-based shaping between object and goal in `reward.py`
- Consider larger hidden layers `[256, 256]` if training is stable
- Increase `total_timesteps` only within the wall-clock budget
- You can redesign the reward function to improve the success rate

**Human baseline:** 0.54
