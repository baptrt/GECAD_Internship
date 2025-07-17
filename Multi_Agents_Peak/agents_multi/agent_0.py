from peak import Agent
from Agent0Behaviour import Agent0Behaviour
import numpy as np

class agent_0(Agent):
    async def setup(self):
        print(f"[{self.name}] O Global Agent(ID: 0)")
        self.params = {
            "id": 0,
            "role": "global",
        }
        behaviour = Agent0Behaviour(
            n_agents=4,
            max_error=1e-1,
        )
        self.add_behaviour(behaviour)