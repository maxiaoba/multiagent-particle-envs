import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
# Xiaobai: test env, two good agents try to push one adversary up
# Not working for now

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 2
        num_adversaries = 1
        num_agents = num_adversaries + num_good_agents

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.5 if agent.adversary else 0.2
            agent.accel = 4.0 if agent.adversary else 3.0
            agent.max_speed = 1.0

        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks

        world.agents[0].state.p_pos = np.array([0.,0.5])
        world.agents[0].state.p_vel = np.zeros(world.dim_p)
        world.agents[0].state.c = np.zeros(world.dim_c)

        world.agents[1].state.p_pos = np.array([0.3,-0.5])
        world.agents[1].state.p_vel = np.zeros(world.dim_p)
        world.agents[1].state.c = np.zeros(world.dim_c)

        world.agents[2].state.p_pos = np.array([-0.3,-0.5])
        world.agents[2].state.p_vel = np.zeros(world.dim_p)
        world.agents[2].state.c = np.zeros(world.dim_c)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        raise NotImplementedError


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        adversaries = self.adversaries(world)
        for adv in adversaries:
            rew += adv.state.p_pos[1]

        rew -= 0.01*(np.abs(agent.action.u[0])+np.abs(agent.action.u[1]))

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0

        rew -= agent.state.p_pos[1]
        rew -= 0.01*(np.abs(agent.action.u[0])+np.abs(agent.action.u[1]))

        return rew

    def observation(self, agent, world):
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel)
