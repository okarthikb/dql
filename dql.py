import gym
import time
import tqdm
import torch
import random
import collections
import numpy as np


def decor(agent, step, render):
  def wrap(*args, **kwargs):
    out = step(*args, **kwargs)
    agent.env.render()
    return out
  return wrap if render else step
    

class Agent:
  def __init__(self, env, net, frames, phi, loc, capacity=10000):
    self.env = env
    self.env_name = env.unwrapped.spec.id
    self.net = net
    self.frames = frames  # number of consecutive frames to process to pass into phi
    self.phi = phi  # state/sequence pre-process function
    self.memory = collections.deque(maxlen=capacity)  # replay memory
    self.loc = loc  # directory for checkpoints

  def play(self, render=True, fps=25):
    state = self.env.reset()
    done = False
    reward = 0
    
    if not render:
      fps = float("inf")

    # initial sequence
    seq = collections.deque(
      [np.zeros_like(state) for _ in range(self.frames - 1)] + [state], 
      maxlen=self.frames
    )

    def step():
      state = self.phi(seq)  # pre-process state/sequence
      # predict Q values
      Qs = torch.reshape(self.net(state), (self.env.action_space.n,))
      action = torch.argmax(Qs).item()  # pick action w/ max Q val
      state, reward_, done, _ = self.env.step(action)  # take action
      seq.append(state)  # update frame sequence
      time.sleep(1 / fps)  # time delay (for rendering properly)
      return reward + reward_, done  # return total reward and done flag

    # step renders game if render == True
    step = decor(self, step, render)

    with torch.no_grad():
      while not done:
        reward, done = step()

    self.env.close()
    return reward

  def train(self, episodes=150, eps=0.9, decay=0.99, gamma=0.999, batch_size=256):
    rewards = []
    # store reward from each episode in a text file
    f = open(self.loc + "/" + self.env_name + "_rewards.txt", "w")

    for i in tqdm.tqdm(range(episodes)):
      done = False
      # reset state every episode and create initial frame sequence
      state = self.env.reset()
      seq = collections.deque(
        [np.zeros_like(state) for _ in range(self.frames - 1)] + [state], 
        maxlen=self.frames
      )
      state = self.phi(seq)

      while not done:
        if random.random() < eps:
          action = self.env.action_space.sample()
        else:
          with torch.no_grad():
            action = torch.argmax(self.net(state)).item()
        # eps-greedy approach - w/ probability eps, pick random action
        # get next_state, reward @ current step, and done flag
        # update sequence
        next_state, reward, done, _ = self.env.step(action)
        seq.append(next_state)
        next_state = self.phi(seq)
        # add transition to replay memory
        self.memory.append((state, action, reward, next_state, done))
        # update state
        state = next_state
        # sample batch from replay memory
        if len(self.memory) <= batch_size:
          batch = self.memory
        else:
          batch = random.sample(self.memory, batch_size)
        # training loop - go through each transition and do grad descent
        # where L = (reward + gamma * max(Q(s')) - max(Q(s)))^2
        for transition in batch:
          state_, action, reward, next_state_, done_ = transition
          self.net.opt.zero_grad()
          with torch.no_grad():
            if done_:
              target = torch.tensor(reward, dtype=torch.float32)  
            else:
              target = reward + gamma * torch.max(self.net(next_state_))
          Qs = torch.reshape(self.net(state_), (self.env.action_space.n,))
          loss = (target - Qs[action]) ** 2
          loss.backward()
          self.net.opt.step()
      # after episode ends, save current NN state
      state_dict = self.net.state_dict()
      torch.save(state_dict, self.loc + "/" + self.env_name + f"_ep{i}.pt")
      # store episode reward in the text file
      rewards.append(self.play(render=False))
      f.write(f"{rewards[-1]}\n")
      eps *= decay  # decay epsilon

    self.env.close()
    return rewards
