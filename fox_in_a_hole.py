import numpy as np

class FoxInAHole():
    def __init__(self, n_holes):
        '''
        Initializes the fox in a hole environemnt parameters.

        Parameters
        ----------
        n_holes (int):
            Number of holes in the environment.
        '''
        self.n_holes = n_holes
        self.reward = 0

    def reset(self):
        '''
        Resets the environment to a random initial state.
        '''
        self.done = False
        self.fox = np.random.randint(0, self.n_holes)
        return self.done

    def step(self):
        '''
        Performs one step in the game environment for the fox.
        '''
        if self.n_holes-1 > self.fox > 0:
            random_movement = np.random.random()
            if random_movement < 0.5:
                self.fox -= 1
            else:
                self.fox += 1
        elif self.fox == 0:
            self.fox += 1
        else:
            self.fox -= 1

    def guess(self, action):
        '''
        Performs one guess by the player. and returns whether it was correct and the reward.

        Parameters
        ----------
        action (int):
            The hole that is being guessed by the player.
        '''
        if action == self.fox:
            self.done = True # the game is won when the fox is found
            self.reward = 0
        else:
            self.reward = -1
        return self.reward, self.done