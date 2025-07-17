from spade.behaviour import OneShotBehaviour
from spade.message import Message
import numpy as np
import json
import asyncio

class ADMMBehaviour_Test(OneShotBehaviour):
    def __init__(self, a, b, n_agents, tmin, tmax, pmin, pmax, neighbors, rho, rhol, max_iters=500, max_error=1e-1):
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
        self.gamma = np.zeros_like(self.T)
        self.errors = []

        self.mu = 0.0
        self.p = 0.0
        self.t_mean = 0.0
        self.iter = 0
        self.local_iter = 0

        self.stable_counter = 0
        self.required_stable_iters = 3
    
    async def await_ack_from_neighbors(self):
        print(f"[{self.agent.name}] Waiting for the neighbours' ACKs...")
        if not hasattr(self, "msg_buffer"):
            self.msg_buffer = []

        max_attempts = 3
        attempts = 0

        while not all(self.ack_received.values()) and attempts < max_attempts:
            # Verify buffer
            for buffered_msg in list(self.msg_buffer):  
                if buffered_msg.get_metadata("type") == "ack":
                    sender_id = int(buffered_msg.body)
                    if sender_id in self.ack_received:
                        self.ack_received[sender_id] = True
                        print(f"[{self.agent.name}] ACK received from agent {sender_id} (via buffer)")
                        self.msg_buffer.remove(buffered_msg)
            
            if all(self.ack_received.values()):
                break

            # Wait a message 
            msg = await self.receive(timeout=10)

            if msg:
                msg_type = msg.get_metadata("type")
                if msg_type == "ack":
                    sender_id = int(msg.body)
                    if sender_id in self.ack_received:
                        self.ack_received[sender_id] = True
                        print(f"[{self.agent.name}] ACK received from agent {sender_id}")
                else:
                    print(f"[{self.agent.name}] Message stored (type: {msg_type})")
                    self.msg_buffer.append(msg)
            else:
                print(f"[{self.agent.name}] Timeout #{attempts + 1} without ACK")
                attempts += 1
                if attempts > 2: 
                    return True

        if not all(self.ack_received.values()):
            print(f"[{self.agent.name}] !!! Incomplete ACK reception : {self.ack_received}")
            return False

        print(f"[{self.agent.name}] --- All ACKs received : {self.ack_received}")
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
                                raise ValueError(f"T_ij received is a list : {T_ij}")
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
                            print(f"[{self.agent.name}] --> Sending ACK to agent {j} (received state from {j})")

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
            self.T_new[i, j] = (self.rho * bt1[i, j] + bt2 * self.rhol - self.gamma[i, j]) / (self.rho + self.rhol)
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
    
    async def global_update_phase(self, i: int, T_final):
        n_agents = self.T.shape[0] 
        
        bt1 = np.zeros((n_agents, n_agents))
        residual_r = 0.0
        residual_s = 0.0
        
        for j in range(n_agents):
            if j == i:
                continue
            residual_r += abs(self.T_new[i, j] + self.T_new[j, i])
            residual_s += abs(self.T_new[i, j] - self.T[i, j])
            self.lambdas[i, j] += (self.rho / 2) * (self.T_new[i, j] + self.T_new[j, i])
            print(f"Tji =", self.T_new[j, i])
            print(f"Tij =", self.T_new[i, j])
            bt1[i, j] = 0.5 * (self.T_new[i, j] - self.T_new[j, i]) - self.lambdas[i, j] / self.rho
        
        for j in range(n_agents):
            if j == i:
                continue
            self.T_new[i, j] = np.array(T_final)[i, j]
            self.T_new[j, i] = np.array(T_final)[j, i]
            
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
                "iter": self.iter
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
                        print(f"[{self.agent.name}] Received local_converged from agent {j}")
                else:
                    print(f"[{self.agent.name}] [DEBUG] Message ignored during local convergence: {msg_type}")
                    message_buffer.append(msg)  # Stores the message for later
            else:
                print(f"[{self.agent.name}] !!! Timeout (no message received for 10s).")
                return False

        print(f"[{self.agent.name}] All neighbors confirmed local convergence.")

        # Store remaining messages in the shared buffer for later usage
        if not hasattr(self.agent, "_custom_buffer"):
            self.agent._custom_buffer = []
        self.agent._custom_buffer.extend(message_buffer)

        return True

    
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
                "residual_s": getattr(self, "residual_s", 1.0), 
                "iter": self.iter
            })
            await self.send(msg)
            print(f"[{self.agent.name}] --> Envoie état à agent_{j}, T_ij = {self.T[i, j]:.3f}")
    
    async def send_exchange_to_agent_0(self, i):
        if i == 0:
            print(f"[{self.agent.name}] Agent 0 does not send exchanges to itself.")
            return
        for j in self.neighbors:
            msg = Message(to="agent_0@xmpp.gecad.isep.ipp.pt")
            msg.set_metadata("performative", "inform")
            msg.set_metadata("type", "exchange_request")
            msg.body = json.dumps({
                    "from": i,
                    "to": j,
                    "T_ij": self.T[i, j]
            })
            await self.send(msg)
            print(f"[{self.agent.name}] --> Sent T[{i},{j}] to agent_0 : {self.T[i,j]}")

    async def receive_power_requests(self, expected_count):
        exchanges = {}
        received = 0
        buffer = []

        while received < expected_count:
            if buffer:
                msg = buffer.pop(0)
            else:
                msg = await self.receive(timeout=3)
                if msg is None:
                    continue 
            if msg:
                try:
                    if msg.get_metadata("type") == "exchange_request":
                        data = json.loads(msg.body)
                        i, j = int(data["from"]), int(data["to"])
                        if i == 0 and j == 0:
                            print(f"[{self.agent.name}] Ignoring exchange request from agent 0 to agent 0.")
                            continue
                        Tij = float(data["T_ij"])
                        exchanges[(i, j)] = Tij
                        received += 1
                        print(f"[{self.agent.name}] <-- Received message : from {i} to {j} | T_ij = {Tij}")
                    else:
                        buffer.append(msg)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"[{self.agent.name}] Parsing message error: {e} | msg.body: {msg.body}")
                    continue
            else:
                print(f"[{self.agent.name}] Timeout for receiving exchanges.")
                return None 
            if received < expected_count:
                print(f"[{self.agent.name}] !!! Missing messages : {expected_count - received}")
                return exchanges if exchanges else None
        return exchanges

    async def wait_for_T_final(self):
        print(f"[{self.agent.name}] Waiting for T_final from agent_0...")
        buffer = []

        while True:
            msg = await self.receive(timeout=10)

            if msg:
                msg_type = msg.get_metadata("type")

                if msg_type == "final_exchange":
                    try:
                        data = json.loads(msg.body)
                        T_final = np.array(data["T_final"])
                        print(f"[{self.agent.name}] T_final received from agent_0.")
                        return T_final
                    except Exception as e:
                        print(f"[{self.agent.name}] T_final parsing error : {e}")
                        return None
                else:
                    print(f"[{self.agent.name}] Message stored (type: {msg_type})")
                    buffer.append(msg)  
            else:
                print(f"[{self.agent.name}] Still waiting for T_final...")
            
    async def wait_for_gamma_update(self):
        received_from = set()

        while True:
            msg = await self.receive(timeout=10)

            if msg:
                msg_type = msg.get_metadata("type")

                if msg_type == "gamma_update":
                    try:
                        data = json.loads(msg.body)
                        from_id = int(data["from"])
                        to_id = int(data["to"])
                        gamma_dict = data["gamma_dict"]

                        if from_id in received_from:
                            continue 

                        for key, gamma_val in gamma_dict.items():
                            try:
                                j_str, n_str = key.split(",")
                                j = int(j_str)
                                n = int(n_str)
                                gamma_val = float(gamma_val)

                                self.gamma[j, n] = gamma_val
                                self.gamma[n, j] = gamma_val

                                print(f"[Agent {to_id}] Received gamma[{j},{n}] = {gamma_val:.4f} from agent {from_id}")
            
                            except ValueError as ve:
                                print(f"[{self.agent.name}] Invalid gamma_dict key '{key}': {ve}")
                                continue

                        received_from.add(from_id)
                        return

                    except (ValueError, KeyError, json.JSONDecodeError) as e:
                        print(f"[{self.agent.name}] gamma_update decoding error : {e}")
                        continue
                else:
                    continue
            else:
                print(f"[{self.agent.name}] Warning: Timeout while waiting for gamma_update messages")
                break
   
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
            self.ack_received = {n: False for n in self.neighbors}
            
            await asyncio.sleep(3.0)
            
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
                        "iter": self.iter
                    }

                # Send current state (or frozen if converged)
                if self.converged and all(self.neighbor_status.values()):
                    await self.send_frozen_state(i)
                else:
                    await self.send_state_to_neighbors(i)
                
                await asyncio.sleep(1.0)
                ok1 = await self.receive_states_from_neighbors(i)
                await asyncio.sleep(1.0)
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
            
            await self.send_exchange_to_agent_0(i)
            
            T_final = await self.wait_for_T_final()
            if T_final is not None:
                bt1, r, s = await self.global_update_phase(i, T_final)
            else: 
                print(f"[{self.agent.name}] No update, T_final missing.")
            
            self.residual_r = r
            self.residual_s = s

            all_residuals = [(r, s)] + list(self.neighbor_residuals.values())
            error = max(max(r_ for r_, _ in all_residuals), max(s_ for _, s_ in all_residuals))

            self.T = self.T_new.copy()
            
            await asyncio.sleep(2.0)
            
            print(f"[{self.agent.name}] Global Iter {global_iter} | Residual Error: {error:.6f}")
            print(f"[{self.agent.name}] T[i] : {self.T[i]}")
            print(f"[{self.agent.name}] Lambda[i] : {self.lambdas[i]}")
            print(f"[{self.agent.name}] Local synchronisation successful, go to global update.")
            
            await asyncio.sleep(1.0)
            await self.send_state_to_neighbors(i)
            await asyncio.sleep(3.0)
            await self.wait_for_gamma_update()

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
            "T_i": np.array(self.T[i]),
            "lambda_i": self.lambdas[i],
            "neighbors": self.neighbors
        }
        
        print(f"[{self.agent.name}] Final power: p = {self.p:.3f}")
        print(f"[{self.agent.name}] xxx All the neighbours have converged globally. Stop.")
        
        msg = Message(to="agent_0@xmpp.gecad.isep.ipp.pt")
        msg.set_metadata("type", "stopping")
        msg.body = str(self.agent.name)
        self.agent.last_T = self.T 
        self.agent.last_prices = self.local_prices
        await self.send(msg)