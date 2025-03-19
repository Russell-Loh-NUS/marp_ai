#!/usr/bin/python3

from gymnasium import spaces
import gymnasium as gym
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
import random
import logging

logging.basicConfig(level=logging.WARN)

# rewards
COLLISION_REWARD = -30
INVALID_ACTION_REWARD = -15
IDLE_REWARD = -1
COMEOUT_FROM_DEST_REWARD = -15
CYCLIC_REWARD = -1
MOVING_CLOSER_REWARD = 10
DEST_REACH_REWARD = 50

# others
CYCLIC_HISTORY_SIZE = 3
MAX_TIMESTEP = 20
OBSERVATION_SPACE = spaces.Box(
    low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, -1, -1] + [-100] * 20, dtype=np.float32),
    high=np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1, 10, 1, 1] + [10] * 20, dtype=np.float32),
    shape=(34,),
    dtype=np.float32,
)
ACTION_SPACE = spaces.Discrete(25)


class MarpAIGym(gym.Env):
    def __init__(self, config=None, render_flag=False):
        super(MarpAIGym, self).__init__()
        self.renderflag = render_flag
        self.action_space = ACTION_SPACE
        self.observation_space = OBSERVATION_SPACE
        self.result = {
            "collision": 0,
            "swap": 0,
            "invalid_action": 0,
            "comeout_from_dest": 0,
            "cyclic": 0,
            "idle": 0,
            "dest_reach": 0,
            "max_timestep": 0,
        }

        self.graph = self.create_graph()
        self.center = (2, 2)  # junction
        self.valid_waypoints = list(self.graph.keys())

        # level and difficulties
        self.level = 0
        self.selected_level = 0
        self.experiences = {
            0: {
                "level_up_threshold": 1500,
                "last_solved_episode": 0,
                "solved_counter": 0,
            },  # both amr1 and amr2 dont the junction
            1: {
                "level_up_threshold": 1500,
                "last_solved_episode": 0,
                "solved_counter": 0,
            },  # one amr cross the junction
            2: {
                "level_up_threshold": 1500,
                "last_solved_episode": 0,
                "solved_counter": 0,
            },  # both amr cross the junction, but not turning
            3: {
                "level_up_threshold": 1500,
                "last_solved_episode": 0,
                "solved_counter": 0,
            },  # both amr cross the junction, and turning
            4: {
                "level_up_threshold": 1500,
                "last_solved_episode": 0,
                "solved_counter": 0,
            },  # both amr share path, but no deadlock
            5: {
                "level_up_threshold": 1500,
                "last_solved_episode": 0,
                "solved_counter": 0,
            },  # both amr share path, with deadlock
            6: {"level_up_threshold": 5000, "last_solved_episode": 0, "solved_counter": 0},  # random spawn and dest
            7: {
                "level_up_threshold": 5000,
                "last_solved_episode": 0,
                "solved_counter": 0,
            },  # random spawn and dest, with random map
            8: {"level_up_threshold": 0, "last_solved_episode": 0, "solved_counter": 0},  # continuous
        }

        self.last_generate_data_when_stuck = 0
        self.generate_new_data = True
        self.episode_count = 0

    def create_graph(self):
        return {
            (0, 2): [(0, 2), (1, 2)],
            (1, 2): [(1, 2), (2, 2), (0, 2)],
            (2, 2): [(2, 2), (2, 3), (3, 2), (2, 1), (1, 2)],
            (3, 2): [(3, 2), (4, 2), (2, 2)],
            (4, 2): [(4, 2), (5, 2), (3, 2)],
            (5, 2): [(5, 2), (6, 2), (4, 2)],
            (6, 2): [(6, 2), (5, 2)],
            (2, 3): [(2, 3), (2, 4), (2, 2)],
            (2, 4): [(2, 4), (2, 5), (2, 3)],
            (2, 5): [(2, 5), (2, 4)],
            (2, 1): [(2, 1), (2, 2), (2, 0)],
            (2, 0): [(2, 0), (2, 1)],
        }

    def init_val(self):
        self.amr1_last_pose = (-100, -100)
        self.amr2_last_pose = (-100, -100)
        self.amr1_closest_distance_to_goal = 100.0
        self.amr2_closest_distance_to_goal = 100.0

        # Generate new data if stuck for 300 episodes
        if self.episode_count - self.last_generate_data_when_stuck >= 300:
            print("Stuck for 300 episodes, generating new data")
            self.generate_new_data = True
            self.last_generate_data_when_stuck = self.episode_count

        # Increase difficulty
        if (
            self.level != 8
            and self.experiences[self.level]["solved_counter"] >= self.experiences[self.level]["level_up_threshold"]
        ):
            self.level += 1
            print(f"Level up to {self.level}")
            self.generate_new_data = True

        if self.generate_new_data:
            print("=============================================================================")
            print(
                f"generating new data, episdode count: {self.episode_count}, level: {self.level}, experience: {self.experiences}"
            )
            print(f"result: {self.result}")
            self.picked_amr1_pose, self.picked_amr1_dest, self.picked_amr2_pose, self.picked_amr2_dest = (
                self.pick_start_dest(self.valid_waypoints)
            )
            print(f"amr1: {self.picked_amr1_pose} -> {self.picked_amr1_dest}")
            print(f"amr2: {self.picked_amr2_pose} -> {self.picked_amr2_dest}")
            self.result = {
                "collision": 0,
                "swap": 0,
                "invalid_action": 0,
                "comeout_from_dest": 0,
                "cyclic": 0,
                "idle": 0,
                "dest_reach": 0,
                "max_timestep": 0,
            }
            self.generate_new_data = False
        self.amr1_pose, self.amr1_dest, self.amr2_pose, self.amr2_dest = (
            self.picked_amr1_pose,
            self.picked_amr1_dest,
            self.picked_amr2_pose,
            self.picked_amr2_dest,
        )

        self.amr1_options = self.pad_waypoints(self.graph[self.amr1_pose])
        self.amr2_options = self.pad_waypoints(self.graph[self.amr2_pose])
        self.step_count = 0
        self.episode_total_score = 0
        self.amr1_last_distance_to_goal = 0.0
        self.amr1_distance_to_goal = self.dist(
            self.amr1_pose[0], self.amr1_pose[1], self.amr1_dest[0], self.amr1_dest[1]
        )
        self.amr2_last_distance_to_goal = 0.0
        self.amr2_distance_to_goal = self.dist(
            self.amr2_pose[0], self.amr2_pose[1], self.amr2_dest[0], self.amr2_dest[1]
        )
        self.amr1_reached = False
        self.amr2_reached = False
        self.amr1_recent_poses = []
        self.amr2_recent_poses = []
        self.episode_count += 1

    def pick_start_dest(self, valid_waypoints):
        left_waypoints = sorted([wp for wp in valid_waypoints if wp[0] < self.center[0]])
        right_waypoints = sorted([wp for wp in valid_waypoints if wp[0] > self.center[0]])
        top_waypoints = sorted([wp for wp in valid_waypoints if wp[1] > self.center[1]], key=lambda x: x[1])
        bottom_waypoints = sorted([wp for wp in valid_waypoints if wp[1] < self.center[1]], key=lambda x: x[1])
        amr1_start, amr1_dest, amr2_start, amr2_dest = None, None, None, None

        # 80% to select the same level, 20% to select a random level
        # self.selected_level = self.level
        # if random.random() > 0.8:
        #     self.selected_level = random.randint(0, self.level)
        self.selected_level = 4
        print(f"Selected level: {self.selected_level}")

        if self.selected_level == 0:
            # both amr don't cross the junction
            amr1_side, amr2_side = random.sample(["left", "right", "top", "bottom"], 2)
            amr1_start, amr1_dest = random.sample(eval(f"{amr1_side}_waypoints"), 2)
            amr2_start, amr2_dest = random.sample(eval(f"{amr2_side}_waypoints"), 2)
        elif self.selected_level == 1:
            # one amr cross the junction
            cross_junction = 1 if random.random() < 0.5 else 2
            cross_junction_start, cross_junction_dest, non_cross = random.sample(["left", "right", "top", "bottom"], 3)
            print(f"Cross junction: amr_{cross_junction}")
            print(f"Cross junction start: {cross_junction_start}, dest: {cross_junction_dest}, non_cross: {non_cross}")
            if cross_junction == 1:
                amr2_start, amr2_dest = random.sample(eval(f"{non_cross}_waypoints"), 2)
                amr1_start, amr1_dest = random.choice(eval(f"{cross_junction_start}_waypoints")), random.choice(
                    eval(f"{cross_junction_dest}_waypoints")
                )
            else:
                amr1_start, amr1_dest = random.sample(eval(f"{non_cross}_waypoints"), 2)
                amr2_start, amr2_dest = random.choice(eval(f"{cross_junction_start}_waypoints")), random.choice(
                    eval(f"{cross_junction_dest}_waypoints")
                )
        elif self.selected_level == 2:
            # both amr cross the junction, but not turning, eg: AMR1: Top to Bottom (or reverse) AMR2: Top to Bottom (or reverse)
            if random.random() < 0.5:
                amr1_start, amr1_dest = random.choice(left_waypoints), random.choice(right_waypoints)
            else:
                amr1_start, amr1_dest = random.choice(right_waypoints), random.choice(left_waypoints)
            if random.random() < 0.5:
                amr2_start, amr2_dest = random.choice(top_waypoints), random.choice(bottom_waypoints)
            else:
                amr2_start, amr2_dest = random.choice(bottom_waypoints), random.choice(top_waypoints)

            if random.random() < 0.5:
                temp_start, temp_dest = amr1_start, amr1_dest
                amr1_start, amr1_dest = amr2_start, amr2_dest
                amr2_start, amr2_dest = temp_start, temp_dest
        elif self.selected_level == 3:
            # both amr cross the junction, and turning, eg: AMR1: Left to Top (or reverse) AMR2: Right to Bottom (or reverse)
            if random.random() < 0.5:
                amr1_start, amr1_dest = random.choice(left_waypoints), random.choice(top_waypoints)
            else:
                amr1_start, amr1_dest = random.choice(top_waypoints), random.choice(left_waypoints)
            if random.random() < 0.5:
                amr2_start, amr2_dest = random.choice(right_waypoints), random.choice(bottom_waypoints)
            else:
                amr2_start, amr2_dest = random.choice(bottom_waypoints), random.choice(right_waypoints)

            if random.random() < 0.5:
                temp_start, temp_dest = amr1_start, amr1_dest
                amr1_start, amr1_dest = amr2_start, amr2_dest
                amr2_start, amr2_dest = temp_start, temp_dest
        elif self.selected_level == 4:
            # both amr share path, but no deadlock
            start_side, dest_side = random.sample(["left", "right", "top", "bottom"], 2)
            print(f"{start_side}, {dest_side}")

            start_options = eval(f"{start_side}_waypoints")
            dest_options = eval(f"{dest_side}_waypoints")

            # Select two start positions and preserve order
            amr1_start, amr2_start = random.sample(start_options, 2)
            start_closer_to_center = 0
            if start_side == "left":
                start_closer_to_center = 1 if amr1_start[0] > amr2_start[0] else 2
            elif start_side == "right":
                start_closer_to_center = 1 if amr1_start[0] < amr2_start[0] else 2
            elif start_side == "top":
                start_closer_to_center = 1 if amr1_start[1] < amr2_start[1] else 2
            elif start_side == "bottom":
                start_closer_to_center = 1 if amr1_start[1] > amr2_start[1] else 2
            print(f"Closer to center: amr_{start_closer_to_center}")
            amr1_dest, amr2_dest = random.sample(dest_options, 2)
            dest_closer_to_center = 0
            if dest_side == "left":
                dest_closer_to_center = 1 if amr1_dest[0] > amr2_dest[0] else 2
            elif dest_side == "right":
                dest_closer_to_center = 1 if amr1_dest[0] < amr2_dest[0] else 2
            elif dest_side == "top":
                dest_closer_to_center = 1 if amr1_dest[1] < amr2_dest[1] else 2
            elif dest_side == "bottom":
                dest_closer_to_center = 1 if amr1_dest[1] > amr2_dest[1] else 2
            print(f"Closer to center: amr_{dest_closer_to_center}")
            if dest_closer_to_center == start_closer_to_center:
                amr1_dest, amr2_dest = amr2_dest, amr1_dest
        elif self.selected_level == 5:
            # both amr share path, with deadlock
            start_side, dest_side = random.sample(["left", "right", "top", "bottom"], 2)
            print(f"{start_side}, {dest_side}")

            start_options = eval(f"{start_side}_waypoints")
            dest_options = eval(f"{dest_side}_waypoints")

            # Select two start positions and preserve order
            amr1_start, amr2_start = random.sample(start_options, 2)
            start_closer_to_center = 0
            if start_side == "left":
                start_closer_to_center = 1 if amr1_start[0] > amr2_start[0] else 2
            elif start_side == "right":
                start_closer_to_center = 1 if amr1_start[0] < amr2_start[0] else 2
            elif start_side == "top":
                start_closer_to_center = 1 if amr1_start[1] < amr2_start[1] else 2
            elif start_side == "bottom":
                start_closer_to_center = 1 if amr1_start[1] > amr2_start[1] else 2
            print(f"Closer to center: amr_{start_closer_to_center}")
            amr1_dest, amr2_dest = random.sample(dest_options, 2)
            dest_closer_to_center = 0
            if dest_side == "left":
                dest_closer_to_center = 1 if amr1_dest[0] > amr2_dest[0] else 2
            elif dest_side == "right":
                dest_closer_to_center = 1 if amr1_dest[0] < amr2_dest[0] else 2
            elif dest_side == "top":
                dest_closer_to_center = 1 if amr1_dest[1] < amr2_dest[1] else 2
            elif dest_side == "bottom":
                dest_closer_to_center = 1 if amr1_dest[1] > amr2_dest[1] else 2
            print(f"Closer to center: amr_{dest_closer_to_center}")
            if dest_closer_to_center != start_closer_to_center:
                amr1_dest, amr2_dest = amr2_dest, amr1_dest
        else:
            amr1_start, amr1_dest, amr2_start, amr2_dest = random.sample(valid_waypoints, 4)
            # TODO: this is level 6, lv7 and 8 will be implemented later
        return amr1_start, amr1_dest, amr2_start, amr2_dest

    def pad_waypoints(self, waypoints, max_size=5, pad_value=(-100, -100)):
        # pad waypoints with (-100, -100) until max_size is reached
        return waypoints + [pad_value] * (max_size - len(waypoints))

    def reset(self, seed=None, options=None):
        self.init_val()
        amr1_direction = np.array(self.amr1_dest) - np.array(self.amr1_pose)
        amr2_direction = np.array(self.amr2_dest) - np.array(self.amr2_pose)
        # Normalize to get unit vectors
        amr1_unit_vector = (
            amr1_direction / np.linalg.norm(amr1_direction) if np.linalg.norm(amr1_direction) > 0 else np.array([0, 0])
        )
        amr2_unit_vector = (
            amr2_direction / np.linalg.norm(amr2_direction) if np.linalg.norm(amr2_direction) > 0 else np.array([0, 0])
        )
        # Return the initial observation
        combined_array = np.concatenate(
            (
                list(self.amr1_pose),
                list(self.amr1_dest),
                list(self.amr2_pose),
                list(self.amr2_dest),
                [self.amr1_distance_to_goal],
                list(amr1_unit_vector),
                [self.amr2_distance_to_goal],
                list(amr2_unit_vector),
                [coord for waypoint in self.amr1_options for coord in waypoint],
                [coord for waypoint in self.amr2_options for coord in waypoint],
            )
        )
        return combined_array, {}

    def dist(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def calculate_reward(self, amr1_next, amr2_next):
        global COMEOUT_FROM_DEST_REWARD
        terminated = False
        truncated = False
        reward = 0.0

        # Ignore movement if selecting the padded value (-100, -100), and apply penalty
        if amr1_next == (-100, -100):
            self.result["invalid_action"] += 1
            reward += INVALID_ACTION_REWARD
        else:  # valid action, check for cyclic
            self.amr1_last_pose = self.amr1_pose
            self.amr1_pose = amr1_next
            if self.amr1_pose != self.amr1_dest and self.amr1_pose in self.amr1_recent_poses:
                logging.info(f"amr 1 cyclic {self.amr1_pose}, recentposes: {self.amr1_recent_poses}")
                self.result["cyclic"] += 1
                reward += CYCLIC_REWARD
            if len(self.amr1_recent_poses) == CYCLIC_HISTORY_SIZE:
                self.amr1_recent_poses.pop(0)
            self.amr1_recent_poses.append(self.amr1_pose)

        if amr2_next == (-100, -100):
            self.result["invalid_action"] += 1
            reward += INVALID_ACTION_REWARD
        else:  # valid action, check for cyclic
            self.amr2_last_pose = self.amr2_pose
            self.amr2_pose = amr2_next
            if self.amr2_pose != self.amr2_dest and self.amr2_pose in self.amr2_recent_poses:
                logging.info(f"amr 2 cyclic {self.amr2_pose}, recentposes: {self.amr2_recent_poses}")
                self.result["cyclic"] += 1
                reward += CYCLIC_REWARD
            if len(self.amr2_recent_poses) == CYCLIC_HISTORY_SIZE:
                self.amr2_recent_poses.pop(0)
            self.amr2_recent_poses.append(self.amr2_pose)
        
        # Incentivise AMR to come out of dest if both AMRs meet
        if self.dist(self.amr1_pose[0], self.amr1_pose[1], self.amr2_pose[0], self.amr2_pose[1]) == 1:
            COMEOUT_FROM_DEST_REWARD = 10

        # terminate on collision and swap
        if self.amr1_pose == self.amr2_pose:
            logging.info("collision, terminated")
            self.result["collision"] += 1
            reward += COLLISION_REWARD
            terminated = True
        if self.amr1_last_pose == self.amr2_pose and self.amr1_pose == self.amr2_last_pose:
            logging.info("swap, terminated")
            self.result["swap"] += 1
            reward += COLLISION_REWARD
            terminated = True

        else:
            # reward for coming out from destination
            if self.amr1_reached and self.amr1_pose != self.amr1_dest:
                logging.info("amr 1 comeout from dest")
                self.result["comeout_from_dest"] += 1
                reward += COMEOUT_FROM_DEST_REWARD
                self.amr1_reached = False
            if self.amr2_reached and self.amr2_pose != self.amr2_dest:
                logging.info("amr 2 comeout from dest")
                self.result["comeout_from_dest"] += 1
                reward += COMEOUT_FROM_DEST_REWARD
                self.amr1_reached = False

            # calculate distance to goal, reward for moving closer
            self.amr1_last_distance_to_goal = self.amr1_distance_to_goal
            self.amr1_distance_to_goal = self.dist(
                self.amr1_pose[0], self.amr1_pose[1], self.amr1_dest[0], self.amr1_dest[1]
            )
            self.amr2_last_distance_to_goal = self.amr2_distance_to_goal
            self.amr2_distance_to_goal = self.dist(
                self.amr2_pose[0], self.amr2_pose[1], self.amr2_dest[0], self.amr2_dest[1]
            )
            if self.amr1_distance_to_goal < self.amr1_closest_distance_to_goal:
                self.amr1_closest_distance_to_goal = self.amr1_distance_to_goal
                reward += MOVING_CLOSER_REWARD
            if self.amr2_distance_to_goal < self.amr2_closest_distance_to_goal:
                self.amr2_closest_distance_to_goal = self.amr2_distance_to_goal
                reward += MOVING_CLOSER_REWARD

            # reward for idling
            if self.amr1_pose == self.amr1_last_pose:
                self.result["idle"] += 1
                reward += IDLE_REWARD
            if self.amr2_pose == self.amr2_last_pose:
                self.result["idle"] += 1
                reward += IDLE_REWARD

            # reward for reaching destination, but only the first time it reaches goal
            if not self.amr1_reached and self.amr1_pose == self.amr1_dest:
                self.amr1_reached = True
                COMEOUT_FROM_DEST_REWARD = -15 # Reset reward
                reward += DEST_REACH_REWARD
                logging.info("solved amr1")
            if not self.amr2_reached and self.amr2_pose == self.amr2_dest:
                self.amr2_reached = True
                COMEOUT_FROM_DEST_REWARD = -15 # Reset reward
                reward += DEST_REACH_REWARD
                logging.info("solved amr2")

            # both amrs reached destination, terminate
            if self.amr1_pose == self.amr1_dest and self.amr2_pose == self.amr2_dest:
                reward += DEST_REACH_REWARD
                self.result["dest_reach"] += 1
                self.experiences[self.selected_level]["solved_counter"] += 1
                reward += DEST_REACH_REWARD
                self.experiences[self.selected_level]["last_solved_episode"] = self.episode_count
                self.generate_new_data = True
                terminated = True
                print(f"solved in {self.step_count} steps")

        # truncated on step count exceeding threshold
        if self.step_count >= MAX_TIMESTEP:
            self.result["max_timestep"] += 1
            truncated = True
        return terminated, truncated, reward

    def get_all_state(self):
        amr1_direction = np.array(self.amr1_dest) - np.array(self.amr1_pose)
        amr2_direction = np.array(self.amr2_dest) - np.array(self.amr2_pose)
        # Normalize to get unit vectors
        amr1_unit_vector = (
            amr1_direction / np.linalg.norm(amr1_direction) if np.linalg.norm(amr1_direction) > 0 else np.array([0, 0])
        )
        amr2_unit_vector = (
            amr2_direction / np.linalg.norm(amr2_direction) if np.linalg.norm(amr2_direction) > 0 else np.array([0, 0])
        )
        combined_array = np.concatenate(
            (
                list(self.amr1_pose),
                list(self.amr1_dest),
                list(self.amr2_pose),
                list(self.amr2_dest),
                [self.amr1_distance_to_goal],
                list(amr1_unit_vector),
                [self.amr2_distance_to_goal],
                list(amr2_unit_vector),
                [coord for waypoint in self.amr1_options for coord in waypoint],
                [coord for waypoint in self.amr2_options for coord in waypoint],
            )
        )
        observations = combined_array
        return (observations, self.reward, self.terminated, self.truncated, {})

    def step(self, action):
        self.step_count += 1

        # Convert the flat action back to amr1 and amr2 actions
        amr1_action = action % 5  # Integer division to get amr1's action
        amr2_action = action // 5  # Modulo operation to get amr2's action
        logging.info(f"amr1_action: {amr1_action}, amr2_action: {amr2_action}")
        amr1_next = self.amr1_options[amr1_action]
        amr2_next = self.amr2_options[amr2_action]

        self.terminated, self.truncated, self.reward = self.calculate_reward(amr1_next, amr2_next)

        self.amr1_options = self.pad_waypoints(self.graph.get(self.amr1_pose, []))
        self.amr2_options = self.pad_waypoints(self.graph.get(self.amr2_pose, []))
        if self.renderflag:
            self.render()

        self.episode_total_score += self.reward
        return self.get_all_state()

    def render(self):
        if not hasattr(self, "_initialized_render"):
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self._initialized_render = True
        self.ax.cla()  # Clear axes to avoid overlaying multiple plots
        # Plot the graph waypoints
        for node, neighbors in self.graph.items():
            self.ax.plot(node[0], node[1], "o", color="grey", markersize=35, alpha=0.5)
            for neighbor in neighbors:
                if neighbor != (-100, -100):
                    self.ax.plot([node[0], neighbor[0]], [node[1], neighbor[1]], "k-")  # plot edges (blue lines)

        # Function to draw rounded square robots with labels
        def draw_robot(x, y, color, label):
            size = 0.35
            robot_patch = patches.FancyBboxPatch(
                (x - size / 2, y - size / 2), size, size, boxstyle="round,pad=0.1", edgecolor=color, facecolor=color
            )
            self.ax.add_patch(robot_patch)
            self.ax.text(x, y, label, color="white", fontsize=10, ha="center", va="center", fontweight="bold")

        # Draw AMR1 and AMR2 as rounded squares
        draw_robot(self.amr1_pose[0], self.amr1_pose[1], "red", "AMR1")
        draw_robot(self.amr2_pose[0], self.amr2_pose[1], "blue", "AMR2")
        # Plot AMR1 and AMR2 destinations as larger red and blue circles
        (amr1_dest_plot,) = self.ax.plot(
            self.amr1_dest[0], self.amr1_dest[1], "ro", markersize=35, alpha=0.5, label="AMR1 Dest"
        )  # red destination
        (amr2_dest_plot,) = self.ax.plot(
            self.amr2_dest[0], self.amr2_dest[1], "bo", markersize=35, alpha=0.5, label="AMR2 Dest"
        )  # blue destination
        self.ax.set_aspect("equal")
        # Label the axes and add a title
        self.ax.set_xlim(-0.5, 6.6)
        self.ax.set_ylim(-0.5, 5.5)
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.set_title("MARP AI Visualization")
        # Manually add legend for AMR1 and AMR2 patches
        legend_patches = [
            patches.Patch(color="red", label="AMR1"),
            patches.Patch(color="blue", label="AMR2"),
            amr1_dest_plot,  # AMR1 destination (from plot)
            amr2_dest_plot,  # AMR2 destination (from plot)
        ]
        self.ax.legend(handles=legend_patches, loc="upper right")
        legend = self.ax.legend(
            handles=legend_patches,
            loc="upper right",
            handler_map={amr1_dest_plot: HandlerLine2D(numpoints=1)},  # Reduce marker size
            markerscale=0.3,
        )
        # Update the plot
        plt.draw()
        plt.pause(0.5)  # Pause to allow for updates


if __name__ == "__main__":
    env = MarpAIGym(render_flag=True)
    print(env.reset())
    for i in range(25):
        print(i)
        env.step(i)
