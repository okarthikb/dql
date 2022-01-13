# dql

`dql.py` is an implementation of DeepMind's DQL algorithm given in this [paper](https://arxiv.org/abs/1312.5602). 

![](dql.png)

![](CartPole-v0_rewards.png)

![](qnet_after.gif)

- above implementation is for the CartPole-v0 env in gym
- to train for another env, change `env_name`, the network architecture, and `frames` in `net.py`
- change hyperparameters in `train.py` and run to train
- load model in `play.py` and run to render
