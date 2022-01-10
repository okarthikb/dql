from net import *


# final Q net named agent.env_name + ".pt"
# Q net @ episode N named agent.env_name + f"_ep{N}.pt"

try:
  loc = agent.loc + "/" + agent.env_name + ".pt"
  print("using model @", loc)
  state_dict = torch.load(loc)
  agent.net.load_state_dict(state_dict)
except:
  print("using default instead")

print(agent.play())
