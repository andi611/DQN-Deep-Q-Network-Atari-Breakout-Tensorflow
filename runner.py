# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ runner.py ]
#   Synopsis     [ main program that runs the training and testing of the DQN agent ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
#   Reference    [ URL ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import argparse
from agent_dqn import Agent_DQN
from environment import Environment
seed = 11037


def parse():
	parser = argparse.ArgumentParser(description="runner")
	parser.add_argument('--env_name', default=None, help='environment name')
	parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
	parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
	parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
	parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
	parser.add_argument('--video_dir', default=None, help='output video directory')
	parser.add_argument('--do_render', action='store_true', help='whether render environment')
	args = parser.parse_args()
	return args


def run(args):
	if args.train_dqn:
		env_name = args.env_name or 'BreakoutNoFrameskip-v4'
		env = Environment(env_name, args, atari_wrapper=True)
		agent = Agent_DQN(env, args)
		agent.train()

	if args.test_dqn:
		env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
		agent = Agent_DQN(env, args)
		test(agent, env, total_episodes=100)


def test(agent, env, total_episodes=30):
	rewards = []
	env.seed(seed)
	for i in range(total_episodes):
		state = env.reset()
		done = False
		episode_reward = 0.0

		# playing one game
		while not done:
			action = agent.make_action(state, test=True)
			state, reward, done, info = env.step(action)
			episode_reward += reward

		rewards.append(episode_reward)
	print('Run %d episodes'%(total_episodes))
	print('Mean:', np.mean(rewards))


if __name__ == '__main__':
	args = parse()
	run(args)
