from spade.behaviour import OneShotBehaviour
from spade.message import Message
import numpy as np
import json

class Agent0Behaviour(OneShotBehaviour):
    def __init__(self, n_agents, max_error):
        super().__init__()
        self.n_agents = n_agents
        self.exchanges = {}
        self.residual = [1.0]
        self.max_error = max_error

    async def receive_power_requests(self):
        
        expected_msgs = (self.n_agents - 1)*2        
        received_msgs = 0
        
        if hasattr(self, "buffered_message"):
            first_msg = self.buffered_message
            del self.buffered_message  
            msgs = [first_msg]
        else:
            msgs = []

        expected_msgs = (self.n_agents - 1)*2 - len(msgs)
        received_msgs = len(msgs)


        while received_msgs < expected_msgs:
            msg = await self.receive(timeout=10)

            if msg and msg.get_metadata("type") == "exchange_request":
                try:
                    data = json.loads(msg.body)
                    i = int(data["from"])
                    j = int(data["to"])
                    Tij = float(data["T_ij"])

                    # on mémorise l'échange
                    self.exchanges[(i, j)] = Tij
                    received_msgs += 1
                    print(f"[{self.agent.name}] <-- Received {received_msgs}/{expected_msgs} "
                        f"({i}->{j}) : {Tij}")

                except Exception as e:
                    print(f"[{self.agent.name}] ()()() Parsing error : {e}")

            else:
                print(f"[{self.agent.name}] !!! Timeout/Msg invalid - I'm still waiting...")

        print(f"[{self.agent.name}] ()()() All messages received "
            f"({received_msgs}/{expected_msgs}).")
        return True


    def compute_mean_exchanges(self):
        T_final = np.zeros((self.n_agents, self.n_agents))
        pairs = set((min(i, j), max(i, j)) for i, j in self.exchanges)
        list_residuals = []

        if not hasattr(self, "prev_exchanges"):
            self.prev_exchanges = {}

        for i, j in pairs:
            Tij = self.exchanges.get((i, j), 0)
            Tji = self.exchanges.get((j, i), 0)

            avg = 0.5 * (Tij - Tji)
            T_final[i, j] = avg
            T_final[j, i] = -avg

            # Comparaison avec l'itération précédente
            prev_Tij = self.prev_exchanges.get((i, j), None)
            prev_Tji = self.prev_exchanges.get((j, i), None)

            print(
                f"[{self.agent.name}] --- Exchange ({i}->{j}): "
                f"Tij={Tij:.6f} (prev: {prev_Tij}), "
                f"Tji={Tji:.6f} (prev: {prev_Tji}), "
                f"ΔTij={'' if prev_Tij is None else f'{Tij - prev_Tij:+.6f}'}"
            )

            # Seulement si valeurs précédentes disponibles
            if prev_Tij is not None:
                list_residuals.append(abs(prev_Tij - Tij))
            if prev_Tji is not None:
                list_residuals.append(abs(prev_Tji - Tji))

            # Mise à jour pour la prochaine itération
            self.prev_exchanges[(i, j)] = Tij
            self.prev_exchanges[(j, i)] = Tji

        print(f"[{self.agent.name}] --- Matrix T_final calculated :\n{T_final*self.n_agents}")
        
        residual = max(list_residuals) if list_residuals else float("inf")
        return T_final, residual


    async def send_final_exchanges(self, T_final):
        for i in range(1, self.n_agents):  
            msg = Message(to=f"agent_{i}@xmpp.gecad.isep.ipp.pt")
            msg.set_metadata("performative", "inform")
            msg.set_metadata("type", "final_exchange")
            msg.body = json.dumps({"T_final": T_final.tolist()})
            await self.send(msg)
            print(f"[{self.agent.name}] --- Sent T_final to agent_{i}")

    async def run(self):
        self.residual = []
        self.stopped_agents = set()

        while True:
            # Écoute prioritaire des messages d'arrêt
            msg = await self.receive(timeout=0.5)
            if msg:
                print(f"[{self.agent.name}] Message received : {msg}")
                msg_type = msg.get_metadata("type")
                if msg_type == "stopping":
                    try:
                        agent_id = int(msg.body)
                        self.stopped_agents.add(agent_id)
                        print(f"[{self.agent.name}] !!! Agent {agent_id} has indicated his stop.")
                    except ValueError:
                        print(f"[{self.agent.name}] Body parsing error : {msg.body}")
                    continue

                elif msg_type == "exchange_request":
                    self.buffered_message = msg
                else: 
                    print(f"[{self.agent.name}] --> Message ignored : {msg_type}")
                
                if len(self.stopped_agents) == self.n_agents - 1:
                    print(f"[{self.agent.name}] --- All other agents have stopped. End of behaviour.")
                    break


            # Traitement normal si au moins un agent est encore actif
            self.residual = []
            success = await self.receive_power_requests()
            if not success:
                print(f"[{self.agent.name}] !!! Failure to receive exchanges.")
                return

            T_final, residual = self.compute_mean_exchanges()
            await self.send_final_exchanges(T_final)
            self.residual.append(residual)
            print(f"[{self.agent.name}] --> Behaviour terminated.")
        
