from peak import Agent
from ADMMBehaviour_Test import ADMMBehaviour_Test
import numpy as np

class agent_2(Agent):
    async def setup(self):
        print(f"[{self.name}] O Agent started (ID: 2)")
        self.params = {
            "id": 2,
            "role": "consumer",
        }
        
        self.last_T = None
        self.last_prices = None
        
        behaviour = ADMMBehaviour_Test(
            a=1.0,
            b=20.0,
            n_agents=4,
            tmin=-15.0,
            tmax=0,
            pmin=-15.0,
            pmax=0,
            neighbors=[1, 3],
            rho=10.0,
            rhol=1.0,
            max_iters=100,
            max_error=1e-1
        )
        self.add_behaviour(behaviour)