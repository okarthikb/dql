from net import *
import matplotlib.pyplot as plt


# adjust hyperparameters
rewards = agent.train(
  episodes=100,
  eps=0.9,
  decay=0.99,
  gamma=0.99999,
  batch_size=256
)

# save model
state_dict = agent.net.state_dict()
torch.save(state_dict, agent.loc + "/" + agent.env_name + ".pt")

# do something with rewards (usually plotting)
fig = plt.figure(figsize=(8, 8))
plt.xlabel("episode")
plt.ylabel("reward")
plt.plot(np.arange(1, len(rewards) + 1), rewards)
plt.savefig(agent.loc + "/" + agent.env_name + "_rewards.png")
