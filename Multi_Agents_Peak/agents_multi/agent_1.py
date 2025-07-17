from peak import Agent
from ADMMBehaviour_Test import ADMMBehaviour_Test
import numpy as np

class agent_1(Agent):
    async def setup(self):
        print(f"[{self.name}] O Agent started (ID: 1)")
        self.params = {
            "id": 1,
            "role": "producer",
        }
        
        self.last_T = None
        self.last_prices = None
        
        behaviour = ADMMBehaviour_Test(
            a=1.0,
            b=-10.0,
            n_agents=4,
            tmin=0.0,
            tmax=15.0,
            pmin=0.0,
            pmax=15.0,
            neighbors=[2, 3],
            rho=10.0,
            rhol=1.0,
            max_iters=100,
            max_error=1e-1
        )
        self.add_behaviour(behaviour)