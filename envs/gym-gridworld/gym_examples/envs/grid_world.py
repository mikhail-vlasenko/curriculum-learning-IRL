from gymnasium import spaces, Env
import pygame
import numpy as np


class GridWorldEnv(Env):
    """
    Valid agent and target locations are in the range [obs_dist, size + obs_dist - 1].
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, grid_size=5, max_steps=None, obs_dist=2, checkers_negative_reward=False):
        self.size = grid_size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.obs_dist = obs_dist
        self.checkers_negative_reward = checkers_negative_reward

        self.rewards = np.random.uniform(-1, 1, size=(self.size + 2 * self.obs_dist, self.size + 2 * self.obs_dist))

        if max_steps is None:
            self.max_steps = 2 * (self.size ** 2)
        else:
            self.max_steps = max_steps

        self.obs_side_length = 1 + 2 * obs_dist
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(self.obs_dist, self.obs_dist + grid_size - 1, shape=(2,), dtype=np.int32),
                "target": spaces.Box(self.obs_dist, self.obs_dist + grid_size - 1, shape=(2,), dtype=np.int32),
                # 0 for tiles out of bounds
                "reward_grid": spaces.Box(-1, 1, shape=(self.obs_side_length, self.obs_side_length), dtype=np.float64),
                # 1 if agent can move to this tile, 0 if not
                "walkable_grid": spaces.MultiBinary(n=(self.obs_side_length, self.obs_side_length)),
                "time_till_end": spaces.Box(0, self.max_steps, shape=(1,), dtype=np.int32),
            }
        )

        self.walkable = np.zeros((self.size + 2 * self.obs_dist, self.size + 2 * self.obs_dist), dtype=np.int32)
        self.walkable[self.obs_dist : self.obs_dist + self.size, self.obs_dist : self.obs_dist + self.size] = 1

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self._time = 0
        self._agent_location = np.zeros(2, dtype=int)
        self._target_location = np.zeros(2, dtype=int)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        idx = self.get_observation_boundaries()
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "reward_grid": self.rewards[
                idx[0][0]:idx[0][1], idx[1][0]:idx[1][1]
            ],
            "walkable_grid": self.walkable[
                idx[0][0]:idx[0][1], idx[1][0]:idx[1][1]
            ],
            "time_till_end": np.array([self.max_steps - self._time]),
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._time = 0

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(self.obs_dist, self.size + self.obs_dist, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                self.obs_dist, self.size + self.obs_dist, size=2, dtype=int
            )

        # initialize random rewards
        self.rewards = np.random.uniform(-1, 1, size=(self.size + 2 * self.obs_dist, self.size + 2 * self.obs_dist))

        if self.checkers_negative_reward:
            # not really checkers, but similar
            # -+-+-
            # +++++
            # -+-+-
            # +++++
            # -+-+-
            self.rewards = np.abs(self.rewards)
            # Create a boolean array where True indicates positions where both i and j are even
            even_indices = np.ix_(np.arange(0, self.size + 2 * self.obs_dist) % 2 == 0,
                                  np.arange(0, self.size + 2 * self.obs_dist) % 2 == 0)

            self.rewards[even_indices] *= -1

        self.rewards[self._agent_location[0], self._agent_location[1]] = 0
        self.rewards[self._target_location[0], self._target_location[1]] = 0
        self.rewards[self.walkable == 0] = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # terminate after some steps
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, self.obs_dist, self.obs_dist + self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = self.get_tile_reward(reset=True)
        observation = self._get_obs()
        info = self._get_info()
        self._time += 1

        if self.render_mode == "human":
            self._render_frame()

        # do not enforce deadline if goal already reached
        if not terminated and self._time >= self.max_steps:
            # also set truncated to true
            reward = -0.5 * self.max_steps
            return observation, reward, True, True, info

        return observation, reward, terminated, False, info

    def get_observation_boundaries(self):
        return [
            [self._agent_location[0] - self.obs_dist, self._agent_location[0] + self.obs_dist + 1],
            [self._agent_location[1] - self.obs_dist, self._agent_location[1] + self.obs_dist + 1],
        ]

    def get_tile_reward(self, reset=False):
        r = self.rewards[self._agent_location[0], self._agent_location[1]]
        if reset:
            self.rewards[self._agent_location[0], self._agent_location[1]] = 0
        return r

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.draw_font = pygame.font.SysFont("monospace", 20)
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # draw rewards
        for x in range(self.obs_dist, self.size + self.obs_dist):
            for y in range(self.obs_dist, self.size + self.obs_dist):
                r = self.rewards[x, y]
                if r != 0:
                    color = (255 - 255 * r, 255, 255 - 255 * r) if r > 0 else (255, 255 + 255 * r, 255 + 255 * r)
                    pygame.draw.rect(
                        canvas,
                        color,
                        pygame.Rect(
                            (pix_square_size * (x - self.obs_dist), pix_square_size * (y - self.obs_dist)),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 255),
            pygame.Rect(
                pix_square_size * (self._target_location - [self.obs_dist, self.obs_dist]),
                (pix_square_size, pix_square_size),
            ),
        )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            ((self._agent_location - [self.obs_dist, self.obs_dist]) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # display remaining steps
        label = self.draw_font.render("Steps left: {}".format(self.max_steps - self._time), True, (0, 0, 0))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            self.window.blit(label, (10, 10))
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
