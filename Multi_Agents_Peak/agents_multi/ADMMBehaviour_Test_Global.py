from spade.behaviour import OneShotBehaviour
from spade.message import Message
import numpy as np
import json
import asyncio

class ADMMBehaviour_Test_Global(OneShotBehaviour):
    def __init__(self, a, b, n_agents, tmin, tmax, pmin, pmax, neighbors, rho, rhol, max_iters=500, max_error=1e-3):
        super().__init__()
        self.a = a
        self.b = b
        self.n_agents = n_agents
        self.tmin = tmin
        self.tmax = tmax
        self.pmin = pmin
        self.pmax = pmax
        self.neighbors = neighbors
        self.rho = rho
        self.rhol = rhol
        self.max_iters = max_iters
        self.max_error = max_error

        self.converged = False
        self.global_converged = False
        self.neighbor_status = {nid: False for nid in neighbors}
        self.neighbor_global_status = {nid: False for nid in neighbors}
        self.neighbor_residuals = {nid: (1.0, 1.0) for nid in neighbors}
        self.ready_global_update_received = {f"agent_{nid}": False for nid in self.neighbors}
        self.residual_r = 1.0
        self.residual_s = 1.0

        self.T = np.zeros((n_agents, n_agents))
        self.T_new = np.zeros_like(self.T)
        self.lambdas = np.zeros_like(self.T)
        self.errors = []

        self.mu = 0.0
        self.p = 0.0
        self.t_mean = 0.0
        self.iter = 0
        self.local_iter = 0

        self.stable_counter = 0
        self.required_stable_iters = 3

    async def send_state_to_neighbors(self, i):
        self.ack_received = {nid: False for nid in self.neighbors}
        for j in self.neighbors:
            msg = Message(to=f"agent_{j}@xmpp.gecad.isep.ipp.pt")
            msg.set_metadata("performative", "inform")
            msg.set_metadata("type", "state")  
            msg.body = json.dumps({
                "id": i,
                "T_ij": self.T[i, j],
                "converged": self.converged,
                "global_converged": self.global_converged,
                "residual_r": getattr(self, "residual_r", 1.0),
                "residual_s": getattr(self, "residual_s", 1.0)
            })
            await self.send(msg)
    
    async def await_ack_from_neighbors(self):
        while not all(self.ack_received.values()):
            msg = await self.receive(timeout=10)
            if msg and msg.get_metadata("type") == "ack":
                sender_id = int(msg.body)
                if sender_id in self.ack_received:
                    self.ack_received[sender_id] = True
            else:
                print(f"[{self.agent.name}] !!! No ACK from any of the neighbours.")
                return False
        return True
    
    async def receive_states_from_neighbors(self, i):
        received_ids = set()
        
        # Initialise the buffer once
        if not hasattr(self.agent, "_custom_buffer"):
            self.agent._custom_buffer = []

        timeout_counter = 0
        max_timeouts = 3  # Tolerates a few minor timeouts

        while len(received_ids) < len(self.neighbors):
            # Process messages already buffered first 
            if self.agent._custom_buffer:
                msg = self.agent._custom_buffer.pop(0)
            else:
                msg = await self.receive(timeout=3)

            if msg:
                msg_type = msg.get_metadata("type")
                if msg_type == "state":
                    try:
                        data = json.loads(msg.body)
                        j = int(data["id"])

                        if j in self.neighbors and j not in received_ids:
                            T_ij = data["T_ij"]
                            if isinstance(T_ij, list):
                                raise ValueError(f"T_ij reçu est une liste : {T_ij}")
                            self.T[j, i] = - float(T_ij)

                            self.neighbor_status[j] = data.get("converged", False)
                            self.neighbor_global_status[j] = data.get("global_converged", False)
                            self.neighbor_residuals[j] = (
                                data.get("residual_r", 1.0),
                                data.get("residual_s", 1.0),
                            )
                            received_ids.add(j)

                            # Sending the ACK
                            ack = Message(to=f"agent_{j}@xmpp.gecad.isep.ipp.pt")
                            ack.set_metadata("performative", "inform")
                            ack.set_metadata("type", "ack")
                            ack.body = str(i)
                            await self.send(ack)

                        else:
                            # Message state useless now, but useful later
                            self.agent._custom_buffer.append(msg)
                    except Exception as e:
                        print(f"[{self.agent.name}] !!! Erreur traitement message state : {e}")
                        return False

                elif msg_type == "ack":
                    pass

                else:
                    print(f"[{self.agent.name}] !!! Type inconnu, mis en buffer : {msg}")
                    self.agent._custom_buffer.append(msg)

            else:
                timeout_counter += 1
                if timeout_counter >= max_timeouts:
                    print(f"[{self.agent.name}]  Réception interrompue après {max_timeouts} timeouts.")
                    return False
                
        return True

    def update_local_variables(self, i, bt1):
        s = 0.0
        list_err = []
        
        print(f"[{self.agent.name}] Neighbors of agent {i}: {self.neighbors}")
        
        for j in self.neighbors:
            bt2 = self.T[i, j] - self.t_mean + self.p - self.mu
            self.T_new[i, j] = (self.rho * bt1[i, j] + bt2 * self.rhol) / (self.rho + self.rhol)
            self.T_new[i, j] = np.clip(self.T_new[i, j], self.tmin, self.tmax)
            list_err.append(abs(self.T_new[i, j] - self.T[i, j]))
            s += self.T_new[i, j]
            
            print(f"[{self.agent.name}] mu: {self.mu:.3f}, p: {self.p:.3f}, t_mean: {self.t_mean:.3f}")
        
        denom = len(self.neighbors) * self.rhol + (len(self.neighbors) ** 2) * self.a
        self.t_mean = s / len(self.neighbors)
        self.p = (len(self.neighbors) * self.rhol * (self.mu + self.t_mean) - len(self.neighbors) * self.b) / denom
        self.p = np.clip(self.p, self.pmin / len(self.neighbors), self.pmax / len(self.neighbors))
        self.mu = self.mu + self.t_mean - self.p
            
        return max(list_err)        
    
    def enforce_antisymmetry(self):
        """
        Imposes T_ij = -T_ji for all pairs (i, j), without the grid.
        """
        n_agents = self.T_new.shape[0]  
        for i in range(1, n_agents):
            for j in range(i + 1, n_agents):
                avg = 0.5 * (self.T_new[i, j] - self.T_new[j, i])
                self.T_new[i, j] = avg
                self.T_new[j, i] = -avg
                
    def enforce_lambda_symmetry(self):
        """Imposes λ_ij = λ_ji for all agents, including the grid."""
        n_agents = self.lambdas.shape[0]

        for i in range(1, n_agents):
            for j in range(i + 1, n_agents):
                avg = 0.5 * (self.lambdas[i, j] + self.lambdas[j, i])
                self.lambdas[i, j] = avg
                self.lambdas[j, i] = avg

    def enforce_zero_on_agent_0(self):
        n_agents = self.lambdas.shape[0]
        for i in range(n_agents):
            for j in range(n_agents):
                if i == 0 or j == 0:
                    self.T_new[i, j] = 0.0
                    self.lambdas[i, j] = 0.0
                    
    def check_antisymmetry(self):
        error = 0
        for i in range(1, self.T_new.shape[0]):
            for j in range(1, self.T_new.shape[1]):
                if i != j:
                    error += abs(self.T_new[i, j] + self.T_new[j, i])
        print(f"Antisymmetry error (excluding agent 0): {error:.6f}")

    
    async def global_update_phase(self, i: int):
        n_agents = self.T.shape[0] 

        bt1 = np.zeros((n_agents, n_agents))
        residual_r = 0.0
        residual_s = 0.0

        for j in range(n_agents):
            if j == i:
                continue
            self.lambdas[i, j] += (self.rho / 2) * (self.T_new[i, j] + self.T_new[j, i])
            bt1[i, j] = 0.5 * (self.T_new[i, j] - self.T_new[j, i]) - self.lambdas[i, j] / self.rho
            residual_r += abs(self.T_new[i, j] + self.T_new[j, i])
            residual_s += abs(self.T_new[i, j] - self.T[i, j])

        # Antisymmetry for all
        # self.enforce_zero_on_agent_0()
        self.enforce_lambda_symmetry()
        self.enforce_antisymmetry()
        self.check_antisymmetry()
                        
        return bt1, residual_r, residual_s

    async def synchronize_global_update(self, i):
        for j in self.neighbors:
            msg = Message(to=f"agent_{j}@xmpp.gecad.isep.ipp.pt")
            msg.set_metadata("performative", "inform")
            msg.set_metadata("type", "ready_global_update")
            agent_id = int(self.agent.name.split("@")[0].split("_")[1])
            msg.body = str(agent_id)
            await self.send(msg)
            print(f"[{self.agent.name}] Sent ready_global_update to {j}")

        received_ready = set()
        deadline = asyncio.get_event_loop().time() + 10  

        while len(received_ready) < len(self.neighbors):
            timeout = deadline - asyncio.get_event_loop().time()
            if timeout <= 0:
                print(f"[{self.agent.name}] !!! Timeout global_sync: missing {set(self.neighbors) - received_ready}")
                return False

            msg = await self.receive(timeout=timeout)
            if msg:
                msg_type = msg.get_metadata("type")
                if msg_type == "ready_global_update":
                    sender_id = int(msg.body)
                    if sender_id in self.neighbors:
                        received_ready.add(sender_id)
                        print(f"[{self.agent.name}] Received ready_global_update from {sender_id}")
                else:
                    print(f"[{self.agent.name}] (info) Message ignored (type={msg_type}) during global sync.")
                    continue
            else:
                continue

        print(f"[{self.agent.name}] All ready_global_update received.")
        return True

    async def notify_local_convergence(self, i):
        for j in self.neighbors:
            msg = Message(to=f"agent_{j}@xmpp.gecad.isep.ipp.pt")
            msg.set_metadata("performative", "inform")
            msg.set_metadata("type", "local_converged")
            msg.body = str(i)
            await self.send(msg)
            
    async def send_frozen_state(self, i):
        for neighbor in self.neighbors:
            jid = f"agent_{neighbor}@xmpp.gecad.isep.ipp.pt"
            
            # Retrieve the T_ij value specific to this neighbor
            T_ij = float(self.T[i, neighbor])

            # Status message construction
            state = {
                "id": i,
                "T_ij": T_ij,
                "converged": True,
                "global_converged": False,
                "residual_r": float(self.residual_r),
                "residual_s": float(self.residual_s),
            }

            msg = Message(to=jid)
            msg.body = json.dumps(state)
            msg.set_metadata("performative", "inform")
            msg.set_metadata("type", "state")
            await self.send(msg)

    async def wait_for_all_local_convergence(self):
        converged_neighbors = set()
        message_buffer = []  # Manual buffer for ignored messages

        while len(converged_neighbors) < len(self.neighbors):
            msg = await self.receive(timeout=10)
            if msg:
                msg_type = msg.get_metadata("type")
                if msg_type == "local_converged":
                    j = int(msg.body)
                    if j in self.neighbors:
                        converged_neighbors.add(j)
                else:
                    print(f"[{self.agent.name}] [DEBUG] Message ignored : {msg_type}")
                    message_buffer.append(msg)  # Stores the message for later
            else:
                print(f"[{self.agent.name}] !!! Timeout (no message received for 10s).")
                return False

        self.agent._custom_buffer = getattr(self.agent, "_custom_buffer", []) + message_buffer

        return True
    
    async def wait_for_all_ready_global_updates(self):
        self.ready_global_update_received = {f"agent_{nid}": False for nid in self.neighbors}

        # Sends ‘ready_global_update’ message to all neighbours
        for j in self.neighbors:
            msg = Message(
                to=f"agent_{j}@xmpp.gecad.isep.ipp.pt",
                body="ready_global_update",
            )
            msg.set_metadata("performative", "inform")
            msg.set_metadata("type", "ready_global_update")
            await self.send(msg)
            print(f"[{self.agent.name}] Sent ready_global_update to {j}")

        while not all(self.ready_global_update_received.values()):
            try:
                msg = await self.receive(timeout=1.0)
                if msg and msg.metadata["performative"] == "inform" and msg.metadata["type"] == "ready_global_update":
                    sender = str(msg.sender).split("@")[0]
                    self.ready_global_update_received[sender] = True
                    print(f"[{self.agent.name}] Received ready_global_update from {sender}")
                    print(f"[{self.agent.name}] All neighbors sent {self.ready_global_update_received}.")
            except asyncio.TimeoutError:
                pass

        print(f"[{self.agent.name}] All neighboirs sent ready_global_update.")
        return True

    async def run(self):
        await asyncio.sleep(1)
        i = self.agent.params["id"]
        self.iter += 1
        global_iter = 0
        error = 2 * self.max_error
        bt1 = np.zeros((self.n_agents, self.n_agents))

        while global_iter < 10 * self.max_iters and error > self.max_error:
            global_iter += 1

            self.local_iter = 0
            self.converged = False
            self.stable_counter = 0
            self.neighbor_status = {nid: False for nid in self.neighbors}

            while not self.converged or not all(self.neighbor_status.values()):
                self.local_iter += 1
                print(f"[{self.agent.name}] State of convergence of neighbours : {self.neighbor_status}")
                LocalError = self.update_local_variables(i, bt1)
                self.T = self.T_new.copy()

                local_error = max(LocalError, abs(self.t_mean - self.p))
                self.errors.append(local_error)

                if local_error <= self.max_error:
                    self.stable_counter += 1
                else:
                    self.stable_counter = 0

                self.converged = self.stable_counter >= self.required_stable_iters

                print(f"[{self.agent.name}] Iter {self.local_iter} | Local error: {local_error:.6f} | Stable: {self.stable_counter}")

                # Stable state saved if local convergence reached
                if self.converged:
                    self.last_state = {
                        "id": i,
                        "T_ij": self.T[i].tolist(),
                        "converged": True,
                        "global_converged": False,
                        "residual_r": self.residual_r,
                        "residual_s": self.residual_s,
                    }

                # Send current state (or frozen if converged)
                if self.converged and all(self.neighbor_status.values()):
                    await self.send_frozen_state(i)
                else:
                    await self.send_state_to_neighbors(i)

                await asyncio.sleep(1.0)
                ok1 = await self.receive_states_from_neighbors(i)
                ok2 = await self.await_ack_from_neighbors()
                await asyncio.sleep(1.0)

                if not (ok1 and ok2):
                    return

            print(f"[{self.agent.name}] --- Local convergence achieved. Notification sent.")
            await self.notify_local_convergence(i)

            ok_sync_local = await self.wait_for_all_local_convergence()
            if not ok_sync_local:
                print(f"[{self.agent.name}] !!! Local synchronisation failed.")
                return

            print(f"[{self.agent.name}] --- All the neighbours confirmed the local convergence.")
            
            bt1, r, s = await self.global_update_phase(i)
            
            self.residual_r = r
            self.residual_s = s

            all_residuals = [(r, s)] + list(self.neighbor_residuals.values())
            error = max(max(r_ for r_, _ in all_residuals), max(s_ for _, s_ in all_residuals))

            print(f"[{self.agent.name}] Global Iter {global_iter} | Residual Error: {error:.6f}")
            print(f"[{self.agent.name}] T[i] : {self.T_new[i]}")
            print(f"[{self.agent.name}] Lambda[i] : {self.lambdas[i]}")
            print(f"[{self.agent.name}] Local synchronisation successful, go to global update.")
            
            # ok_sync = await self.wait_for_all_ready_global_updates()

            # if not ok_sync:
            #     print(f"[{self.agent.name}] !!! Échec de la synchronisation globale.")
            #     return
            
            self.T = self.T_new.copy()

        print(f"[{self.agent.name}] ||| Total convergence achieved.")

        # Once the global convergence loop is complete, check that everyone has converged
        global_wait_iter = 0
        while not all(self.neighbor_global_status.values()):
            ok = await self.receive_states_from_neighbors(i)
            if not ok:
                print(f"[{self.agent.name}] !!! Timeout or reception failure after convergence. Forced restart.")
                self.global_converged = False  # Allows the loop to start again 
                break
            await asyncio.sleep(0.5)
            global_wait_iter += 1
            if global_wait_iter > 20: 
                print(f"[{self.agent.name}] !!! No complete global convergence. Loop again.")
                self.global_converged = False
                break


        # Final results recorded and agent's own stop
        self.agent.results = {
            "id": i,
            "role": self.agent.params["role"],
            "p": self.p,
            "mu": self.mu,
            "T_i": self.T[i].tolist(),
            "lambda_i": self.lambdas[i].tolist(),
            "neighbors": self.neighbors
        }

        print(f"[{self.agent.name}] Final power: p = {self.p:.3f}")
        print(f"[{self.agent.name}] xxx All the neighbours have converged globally. Stop.")
        await self.agent.stop()