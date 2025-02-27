#!/usr/bin/python3

from gymnasium import spaces
import gymnasium as gym
import math
import matplotlib.pyplot as plt
import numpy as np
import random

COLLISION_REWARD = -100
INVALID_ACTION_REWARD = -10
IDLE_REWARD = -2
DEST_REACH_REWARD = 25
MAX_TIMESTEP = 50


class MarpAIGym(gym.Env):
    def __init__(self, render_flag=False):
        super(MarpAIGym, self).__init__()
        self.graph = self.create_graph()
        self.action_space = spaces.Discrete(25)  # 5 actions for each AMR
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [-100] * 20, dtype=np.float32),
            high=np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10] + [10] * 20, dtype=np.float32),
            shape=(30,),
            dtype=np.float32,
        )  # amr poses, dest, distance to goal, waypoints, 30 elements
        self.renderflag = render_flag
        self.init_val()

    def create_graph(self):
        return {
            (0, 2): [(0, 2), (1, 2)],
            (1, 2): [(1, 2), (0, 2), (2, 2)],
            (2, 2): [(2, 2), (1, 2), (3, 2), (2, 3), (2, 1)],
            (3, 2): [(3, 2), (2, 2), (4, 2)],
            (4, 2): [(4, 2), (3, 2), (5, 2)],
            (5, 2): [(5, 2), (4, 2)],
            (2, 3): [(2, 3), (2, 2)],
            (2, 1): [(2, 1), (2, 2), (2, 0)],
            (2, 0): [(2, 0), (2, 1)],
        }

    def init_val(self):
        self.amr1_last_pose = (-100, -100)
        self.amr1_pose = (0, 2)
        self.amr2_last_pose = (-100, -100)
        self.amr2_pose = (2, 0)
        self.amr1_dest = (5, 2)
        self.amr2_dest = (2, 3)
        self.amr1_options = self.pad_waypoints(self.graph[self.amr1_pose])
        self.amr2_options = self.pad_waypoints(self.graph[self.amr2_pose])
        self.step_count = 0
        self.episode_total_score = 0
        self.amr_1_distance_to_goal = self.dist(
            self.amr1_pose[0], self.amr1_pose[1], self.amr1_dest[0], self.amr1_dest[1]
        )
        self.amr_2_distance_to_goal = self.dist(
            self.amr2_pose[0], self.amr2_pose[1], self.amr2_dest[0], self.amr2_dest[1]
        )

    def pad_waypoints(self, waypoints, max_size=5, pad_value=(-100, -100)):
        # pad waypoints with (-100, -100) until max_size is reached
        return waypoints + [pad_value] * (max_size - len(waypoints))

    def reset(self, seed=None):
        self.init_val()

        # Return the initial observation
        combined_array = np.concatenate(
            (
                list(self.amr1_pose),
                list(self.amr1_dest),
                list(self.amr2_pose),
                list(self.amr2_dest),
                [self.amr_1_distance_to_goal],
                [self.amr_2_distance_to_goal],
                [coord for waypoint in self.amr1_options for coord in waypoint],
                [coord for waypoint in self.amr2_options for coord in waypoint],
            )
        )
        return combined_array, {}

    def dist(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def calculate_reward(self, amr1_next, amr2_next):
        terminated = False
        truncated = False
        reward = 0
        amr1_reached = False
        amr2_reached = False

        # Ignore movement if selecting the padded value (-100, -100), and apply penalty
        if amr1_next == (-100, -100):
            reward += INVALID_ACTION_REWARD
        else:
            self.amr1_last_pose = self.amr1_pose
            self.amr1_pose = amr1_next
        if amr2_next == (-100, -100):
            reward += INVALID_ACTION_REWARD
        else:
            self.amr2_last_pose = self.amr2_pose
            self.amr2_pose = amr2_next

        # calculate distance to goal
        self.amr_1_distance_to_goal = self.dist(
            self.amr1_pose[0], self.amr1_pose[1], self.amr1_dest[0], self.amr1_dest[1]
        )
        self.amr_2_distance_to_goal = self.dist(
            self.amr2_pose[0], self.amr2_pose[1], self.amr2_dest[0], self.amr2_dest[1]
        )

        # terminate on collision
        if self.amr1_pose == self.amr2_pose:
            terminated = True
            reward += COLLISION_REWARD

        # reward for idling
        if self.amr1_pose == self.amr1_last_pose:
            reward += IDLE_REWARD
        if self.amr2_pose == self.amr2_last_pose:
            reward += IDLE_REWARD

        # reward for reaching destination
        if self.amr1_pose == self.amr1_dest:
            amr1_reached = True
            print("solved amr1")
            reward += DEST_REACH_REWARD
        if self.amr2_pose == self.amr2_dest:
            amr2_reached = True
            print("solved amr2")
            reward += DEST_REACH_REWARD

        # both amrs reached destination, terminate
        if amr1_reached and amr2_reached:
            print("solved both, terminating")
            terminated = True

        # truncated on step count exceeding threshold
        if self.step_count >= MAX_TIMESTEP:
            truncated = True
        return terminated, truncated, reward

    def get_all_state(self):
        combined_array = np.concatenate(
            (
                list(self.amr1_pose),
                list(self.amr1_dest),
                list(self.amr2_pose),
                list(self.amr2_dest),
                [self.amr_1_distance_to_goal],
                [self.amr_2_distance_to_goal],
                [coord for waypoint in self.amr1_options for coord in waypoint],
                [coord for waypoint in self.amr2_options for coord in waypoint],
            )
        )
        observations = combined_array
        return (observations, self.reward, self.terminated, self.truncated, {})

    def step(self, action):
        self.step_count += 1

        # Convert the flat action back to amr1 and amr2 actions
        amr1_action = action // 5  # Integer division to get amr1's action
        amr2_action = action % 5  # Modulo operation to get amr2's action

        # amr1_action, amr2_action = action
        amr1_next = self.amr1_options[amr1_action]
        amr2_next = self.amr2_options[amr2_action]

        self.terminated, self.truncated, self.reward = self.calculate_reward(
            amr1_next, amr2_next
        )  # return bool and float

        self.amr1_options = self.pad_waypoints(self.graph.get(self.amr1_pose, []))
        self.amr2_options = self.pad_waypoints(self.graph.get(self.amr2_pose, []))
        if self.renderflag:
            self.render()

        self.episode_total_score += self.reward
        # print(f"Step {self.step_count}: Reward = {self.reward}, Total Score = {self.episode_total_score}")

        return self.get_all_state()

    def render(self):
        """
        Renders the current state of the environment using matplotlib.
        Displays the AMR1 and AMR2 positions and destinations, and the graph of possible waypoints.
        Updates the same plot window.
        """
        if not hasattr(self, "_initialized_render"):
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self._initialized_render = True

        self.ax.cla()  # Clear axes to avoid overlaying multiple plots

        # Plot the graph waypoints
        for node, neighbors in self.graph.items():
            self.ax.plot(node[0], node[1], "go")  # plot node (green circle)
            for neighbor in neighbors:
                self.ax.plot([node[0], neighbor[0]], [node[1], neighbor[1]], "b-")  # plot edges (blue lines)

        # Plot AMR1 and AMR2 current positions
        self.ax.plot(self.amr1_pose[0], self.amr1_pose[1], "ro", markersize=15, label="AMR1")  # red circle
        self.ax.plot(self.amr2_pose[0], self.amr2_pose[1], "bo", markersize=15, label="AMR2")  # blue circle

        # Plot AMR1 and AMR2 destinations
        self.ax.plot(self.amr1_dest[0], self.amr1_dest[1], "rx", markersize=12, label="AMR1 Dest")  # red cross
        self.ax.plot(self.amr2_dest[0], self.amr2_dest[1], "bx", markersize=12, label="AMR2 Dest")  # blue cross

        # Label the axes and add a title
        self.ax.set_xlim(0, 5)
        self.ax.set_ylim(0, 3)
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.set_title("AMR1 & AMR2 Environment")

        # Display the legend
        self.ax.legend()

        # Update the plot
        plt.draw()
        plt.pause(0.5)  # Pause to allow for updates


if __name__ == "__main__":
    env = MarpAIGym(render_flag=True)
    print(env.reset())
    print(env.step(random.randint(0, 24)))
    print(env.step(random.randint(0, 24)))
