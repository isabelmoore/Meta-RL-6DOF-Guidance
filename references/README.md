# References

## Primary Paper

**Integrated and Adaptive Guidance and Control for Endoatmospheric Missiles via Reinforcement Meta-Learning**
Brian Gaudet & Roberto Furfaro (2023)
- Paper: https://arxiv.org/pdf/2109.03880
- Journal: AIAA Journal of Guidance, Control, and Dynamics

This paper defines the core methodology: RL² (meta-RL) for 6-DOF missile guidance using
RecurrentPPO with LSTM policies trained across multiple engagement scenarios.

Key contributions used in this codebase:
- 23-dim observation space (LOS geometry, attitude, rates, fins)
- Reward function (Eq. 30a-30f): LOS-rate shaping, closing reward, hit bonus
- Proportional Navigation as reference guidance
- Engagement initial conditions and path constraints (Tables 1-2)

## Guidance Law References

- **Proportional Navigation (PN)**: Zarchan, P. "Tactical and Strategic Missile Guidance" (6th ed.), AIAA, 2012.
- **Augmented PN (APN)**: Zarchan Ch. 8 — adds target acceleration compensation to PN.
- **Zero Effort Miss (ZEM)**: Zarchan Ch. 9 — optimal guidance minimizing predicted miss distance.
- **Pure Pursuit**: Classical guidance — always steer toward target's current position.

## Meta-RL / RL²

- Duan et al., "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning", 2016. https://arxiv.org/abs/1611.02779
- Wang et al., "Learning to reinforcement learn", 2016. https://arxiv.org/abs/1611.05763
