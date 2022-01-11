from dql import *


# init environment from gym
env = gym.make("CartPole-v0")


# define the Q network
class DQN(torch.nn.Module):
  def __init__(self):
    super(DQN, self).__init__()
    # define layers
    self.fc = torch.nn.Linear(4, 256)
    self.out = torch.nn.Linear(256, env.action_space.n)
    # define optimizer
    self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

  # forward pass
  def forward(self, x):
    x = torch.nn.functional.relu(self.fc(x))
    x = torch.nn.functional.relu(self.out(x))
    return x


# define state/sequence pre-processing function
def phi(seq):
  return torch.tensor(seq[0], dtype=torch.float)


# init Q net
net = DQN().to(torch.device("cpu"))

# number of consecutive frames to process
frames = 1

# directory to save model
loc = "checkpoints"

# init agent to train
agent = Agent(
  env=env,
  net=net,
  frames=frames,
  phi=phi,
  loc=loc
)
