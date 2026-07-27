import os
# Fix NumPy 2.0 uint8 overflow in nes-py/gym-super-mario-bros
os.environ['NPY_PROMOTION_STATE'] = 'legacy'

import gymnasium as gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import pygame
import sys

def main():
    # Instantiate the SuperMarioBrosEnv directly to bypass legacy gym.make conflicts
    env = gym_super_mario_bros.SuperMarioBrosEnv()
    # Wrap with JoypadSpace to use COMPLEX_MOVEMENT actions (discrete 0-11)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    # Use rgb_array mode to bypass Pyglet window issues entirely
    env.unwrapped.render_mode = 'rgb_array'
    
    state, info = env.reset()
    done = False
    
    # Initialize Pygame and create a display window
    pygame.init()
    screen = pygame.display.set_mode((512, 480))
    pygame.display.set_caption("Super Mario Bros Manual Play")
    
    clock = pygame.time.Clock()
    
    print("Manual Control Instructions:")
    print("  Right Arrow : Move Right")
    print("  Left Arrow  : Move Left")
    print("  Down Arrow  : Duck / Go Down Pipe")
    print("  Up Arrow    : Climb / Go Up")
    print("  Z / Space   : Jump (A button)")
    print("  X / Shift   : Run (B button)")
    print("  ESC / Close Window to Quit")

    while not done:
        action = 0  # default: NOOP
        
        # Poll events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
        
        # Get state of all keys
        keys = pygame.key.get_pressed()
        
        # Map keys to COMPLEX_MOVEMENT actions
        right = keys[pygame.K_RIGHT]
        left = keys[pygame.K_LEFT]
        down = keys[pygame.K_DOWN]
        up = keys[pygame.K_UP]
        jump = keys[pygame.K_z] or keys[pygame.K_SPACE]
        run = keys[pygame.K_x] or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        if left:
            if jump and run:
                action = 9  # left + A + B
            elif jump:
                action = 7  # left + A
            elif run:
                action = 8  # left + B
            else:
                action = 6  # left
        elif right:
            if jump and run:
                action = 4  # right + A + B
            elif jump:
                action = 2  # right + A
            elif run:
                action = 3  # right + B
            else:
                action = 1  # right
        elif down:
            action = 10     # down
        elif up:
            action = 11     # up
        elif jump:
            action = 5      # A (jump)
        else:
            action = 0      # NOOP
            
        # Step env
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Get screen frame as numpy array
        frame = env.render()
        
        # Convert RGB numpy array to Pygame Surface
        if frame is not None:
            surf = pygame.image.frombuffer(frame.tobytes(), (256, 240), 'RGB')
            # Scale frame to 512x480 for better viewing
            scaled_surf = pygame.transform.scale(surf, (512, 480))
            screen.blit(scaled_surf, (0, 0))
            pygame.display.flip()
            
        clock.tick(60)  # Limit to 60 FPS for realistic NES speed
        
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
