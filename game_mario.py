import os
# Fix NumPy 2.0 uint8 overflow in nes-py/gym-super-mario-bros
os.environ['NPY_PROMOTION_STATE'] = 'legacy'
# Fix OpenMP duplicate runtime initialization conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
import matplotlib.pyplot as plt

from agent1 import Agent

def make_env(render_mode=None):
    # Instantiate the SuperMarioBrosEnv directly to bypass legacy gym.make conflicts
    env = gym_super_mario_bros.SuperMarioBrosEnv()
    if render_mode is not None:
        env.render_mode = render_mode
    # Wrap with JoypadSpace to reduce action space (e.g. RIGHT_ONLY has 5 actions)
    env = JoypadSpace(env, RIGHT_ONLY)
    # Convert frames to grayscale
    env = GrayScaleObservation(env, keep_dim=False)
    # Resize observations to 84x84
    env = ResizeObservation(env, 84)
    # Stack 4 consecutive frames
    env = FrameStack(env, num_stack=4)
    return env

def train_agent(args):
    render_mode = "human" if args.render else None
    env = make_env(render_mode=render_mode)
    state_dim = (4, 84, 84)
    action_dim = env.action_space.n

    agent = Agent(args.param_set, state_dim, action_dim)
    
    # Auto-load checkpoint if it exists
    model_file = args.model_path if args.model_path else agent.MODEL_FILE
    if os.path.exists(model_file):
        print(f"Loading checkpoint: {model_file}")
        agent.load(model_file)
    else:
        print("No checkpoint found. Starting fresh training.")

    print(f"Starting training on {args.param_set}...")
    print(f"Device: {agent.policy_net.conv[0].weight.device}")
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}")

    best_reward = -float('inf')
    reward_history = []
    loss_history = []
    epsilon_history = []

    # Write header to log file
    with open(agent.LOG_FILE, "a") as f:
        f.write(f"--- Training started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write("Episode,Steps,TotalReward,Epsilon,AvgLoss\n")

    for episode in range(1, args.episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_losses = []

        start_time = time.time()

        while True:
            action = agent.act(state, train=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            done = terminated or truncated or (episode_steps >= args.max_steps)

            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            
            if episode_steps % 4 == 0:
                loss = agent.learn()
                if loss is not None:
                    episode_losses.append(loss)

            episode_reward += reward

            if done:
                break

        duration = time.time() - start_time
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        reward_history.append(episode_reward)
        loss_history.append(avg_loss)
        epsilon_history.append(agent.epsilon)

        log_msg = f"Ep {episode}/{args.episodes} | Steps: {episode_steps} | Reward: {episode_reward:.1f} | Epsilon: {agent.epsilon:.4f} | Loss: {avg_loss:.4f} | Time: {duration:.1f}s"
        print(log_msg)

        # Log to file
        with open(agent.LOG_FILE, "a") as f:
            f.write(f"{episode},{episode_steps},{episode_reward:.1f},{agent.epsilon:.6f},{avg_loss:.6f}\n")

        # Save model if new best reward is achieved or reward threshold is met
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save()
            print(f"--> Saved new best model checkpoint with reward: {best_reward:.1f}")

        if episode_reward >= agent.reward_threshold:
            print(f"Reward threshold of {agent.reward_threshold} reached! Training successful.")
            agent.save()
            break

        # Save training plots periodically
        if episode % args.plot_interval == 0:
            save_plots(reward_history, loss_history, agent.param_set)

    env.close()
    save_plots(reward_history, loss_history, agent.param_set)
    print("Training finished.")

def save_plots(rewards, losses, param_set):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Episodic Reward")
    # Running average reward
    if len(rewards) >= 10:
        run_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(rewards)), run_avg, label="10-Ep Moving Avg", color="red")
    plt.title("Reward History")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Average Loss", color="orange")
    plt.title("Loss History")
    plt.xlabel("Episode")
    plt.ylabel("MSE Loss")
    plt.legend()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(current_dir, "runs", f"{param_set}_progress.png")
    plt.savefig(plot_path)
    plt.close()

def play_agent(args):
    # Enable rendering inside the environment instantiation
    env = make_env(render_mode="human")
    state_dim = (4, 84, 84)
    action_dim = env.action_space.n

    agent = Agent(args.param_set, state_dim, action_dim)
    
    model_file = args.model_path if args.model_path else agent.MODEL_FILE
    if not agent.load(model_file):
        print(f"Failed to load model weights. Cannot play.")
        sys.exit(1)

    print(f"Playing agent from checkpoint {model_file}...")
    
    for episode in range(1, args.episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state, train=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated or (steps >= args.max_steps)
            
            state = next_state
            episode_reward += reward
            
            try:
                env.render()
                time.sleep(0.01) # Delay to make visual gameplay normal speed
            except Exception as e:
                # Handle rendering issues gracefully if running headlessly
                pass

            if done:
                break
                
        print(f"Play Episode {episode} finished | Steps: {steps} | Reward: {episode_reward:.1f}")
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Play a DQN Agent on Super Mario Bros")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "play"],
                        help="Mode: 'train' to train a new agent, 'play' to run a trained agent")
    parser.add_argument("--param_set", type=str, default="param1",
                        help="Name of the parameter set from parameters1.yaml to load")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes to run (default: 1000 for train, 10 for play)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint .pt file (optional)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from an existing checkpoint")
    parser.add_argument("--plot_interval", type=int, default=50,
                        help="Plot progress every N episodes")
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Maximum steps allowed per episode (default: 2000)")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during training (always rendered in play mode)")

    args = parser.parse_args()
    
    # Adjust default episodes if in play mode
    if args.mode == "play" and args.episodes == 1000:
        args.episodes = 10

    if args.mode == "train":
        train_agent(args)
    else:
        play_agent(args)
