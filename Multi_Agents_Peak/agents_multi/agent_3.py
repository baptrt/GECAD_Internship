from peak import Agent
from ADMMBehaviour_Grid import ADMMBehaviour_Test
import numpy as np

class agent_3(Agent):
    async def setup(self):
        print(f"[{self.name}] O Agent started (ID: 3)")
        self.params = {
            "id": 3,
            "role": "grid",
        }
        
        self.last_T = None
        self.last_prices = None
        
        behaviour = ADMMBehaviour_Test(
            a=1.0,
            b=10.0,
            n_agents=4,
            tmin=-np.inf,
            tmax=np.inf,
            pmin=-np.inf,
            pmax=np.inf,
            neighbors=[1, 2],
            rho=10.0,
            rhol=1.0,
            max_iters=100,
            max_error=1e-1
        )
        self.add_behaviour(behaviour)