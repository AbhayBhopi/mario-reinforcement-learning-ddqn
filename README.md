# Mario Reinforcement Learning Agent (Double DQN)

Deep Reinforcement Learning agent trained to play Super Mario Bros using **PyTorch**, **Double Deep Q-Networks (DDQN)**, and **Gymnasium**.

##  Features
- **Convolutional Feature Extractor**: Neural network processing 84x84 grayscale frame stacks.
- **Double DQN Architecture**: Mitigates Q-value overestimation bias.
- **Prioritized Experience Replay**: Stores transition tuples `(state, action, reward, next_state, done)` for continuous policy optimization.
- **Epsilon-Greedy Exploration**: Dynamic decay for exploration vs exploitation balance.

##  Tech Stack
- Python 3.10+
- PyTorch
- Gymnasium & `gym-super-mario-bros`
- OpenCV & NumPy

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision gym-super-mario-bros nes-py PyYAML
   ```

2. **Train / Test the Agent**:
   ```bash
   python game_mario.py
   ```

3. **Manual Keyboard Play**:
   ```bash
   python play_mario_manual.py
   ```
