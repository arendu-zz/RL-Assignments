### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters
import pdb
import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)
np.seterr(all='raise') 


def converged(Vk, Vk1, tol=1e-3):
    return np.linalg.norm(Vk - Vk1) < tol

def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
        # formular for value iteraction is slide 24 from DP.pdf
        # V[s](k+1) = max_a_in_A (R_s^a + \gamma \sum_s' P(s'|a,s )V[s'](k)) 
	V = np.zeros(nS)
        policy = np.zeros((nS), dtype=np.int32)
        for _iter in xrange(max_iteration):
            V_kp1 = np.zeros(nS)
            for s in xrange(nS):
                v_kp1 = np.ones(nA,) * -np.inf 
                for a in xrange(nA):
                    _tmp_a = 0.
                    for prob_sp_s, sp, r_s_a_sp, is_terminal in P[s][a]:
                        _tmp_a  += prob_sp_s * (r_s_a_sp + gamma * V[sp])
                        #v_kp1[a] += prob_sp_s * (r_s_a_sp + gamma * V[sp])
                    v_kp1[a] = _tmp_a
                V_kp1[s] = np.max(v_kp1)
                policy[s] = np.argmax(v_kp1)
            if converged(V, V_kp1, tol):
                break
            else:
                pass
            V = V_kp1
	return V, policy


def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	policy: np.array
		The policy to evaluate. Maps states to actions.
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns
	-------
	value function: np.ndarray
		The value function from the given policy.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
        # V[s](k+1) = max_a_in_A (R_s^a + \gamma \sum_s' P(s'|a,s )V[s'](k)) 
	V = np.zeros(nS)
        for _iter in xrange(max_iteration):
            V_kp1 = np.zeros(nS)
            for s in xrange(nS):
                q_s_kp1 = np.zeros(nA,) 
                for a in xrange(nA):
                    for prob_sp_s, sp, r_s_a_sp, is_terminal in P[s][a]:
                        q_s_kp1[a] += prob_sp_s * (r_s_a_sp + gamma * V[sp])
                a_dist = np.zeros(nA,)
                a_dist[int(policy[s])] = 1. #for non-greedy set a_dist from policy in an appropriate manner
                V_kp1[s] = np.sum(a_dist * q_s_kp1)
            if converged(V, V_kp1, tol):
                break
            else:
                pass
            V = V_kp1
	return V 


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new policy: np.ndarray
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""    
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
        Q =np.zeros((nS,nA)) 
        for s in range(nS):
            for a in xrange(nA):
                for prob_sp_s, sp, r_s_a_sp, is_terminal in P[s][a]:
                    Q[s,a] += prob_sp_s * (r_s_a_sp + gamma * value_from_policy[sp])
        new_policy = np.argmax(Q, axis = 1)
        assert new_policy.shape[0] == nS
	return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""Runs policy iteration.

	You should use the policy_evaluation and policy_improvement methods to
	implement this method.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
        for _ in xrange(max_iteration):
            V = policy_evaluation(P, nS, nA, policy)
            policy = policy_improvement(P, nS, nA, V, policy)
	return V, policy



def example(env):
	"""Show an example of gym
	Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
	"""
	env.seed(0); 
	from gym.spaces import prng; prng.seed(10) # for print the location
	# Generate the episode
	ob = env.reset()
	for t in range(100):
		env.render()
		a = env.action_space.sample()
		ob, rew, done, _ = env.step(a)
		if done:
			break
	assert done
	env.render();

def render_single(env, policy):
	"""Renders policy once on environment. Watch your agent play!

		Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	"""

	episode_reward = 0
	ob = env.reset()
	for t in range(100):
		env.render()
		time.sleep(0.5) # Seconds between frames. Modify as you wish.
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	assert done
	env.render();
	print "Episode reward: %f" % episode_reward


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
	env = gym.make("Deterministic-4x4-FrozenLake-v0")
	print env.__doc__
	print "Here is an example of state, action, reward, and next state"
	example(env)
        from pprint import pprint
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
        print 'V_vi'
        pprint(V_vi)
        print 'P_vi'
        pprint(p_vi)

	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
        print 'V_pi'
        pprint(V_pi)
        print 'P_pi'
        pprint(p_pi)
	
