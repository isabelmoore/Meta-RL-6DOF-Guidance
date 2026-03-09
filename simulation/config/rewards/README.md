# Reward Configs

Reward shaping weights and curriculum. Set per-scenario via `reward_config`.

## Files

| File | Hit Radius | Curriculum | Use |
|------|-----------|------------|-----|
| gaudet.yaml | 50 m | 500 -> 50 m | Short range scenarios |
| gaudet_tight.yaml | 10 m | 3000 -> 10 m | Extended range / wide angle |

Both: hit_bonus=500, violation_penalty=25, alpha=0.1, sigma=0.05.

To add a new reward, copy an existing file and update your scenario:

    reward_config: "simulation/config/rewards/my_reward.yaml"
