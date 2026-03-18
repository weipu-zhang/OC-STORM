import cv2
import numpy as np
import platform
import time
import keyboard  # send keyboard commands
import os
import ray
import mss  # screenshot
from typing import Dict
import json

if platform.system() == "Windows":
    import pygetwindow as gw
else:
    print("pygetwindow not imported, not Windows system.")
    gw = None


# get Hollow Knight game window by title
def get_game_window(title):
    windows = gw.getWindowsWithTitle(title)
    return windows[0] if windows else None


# capture image from specified window
def capture_window_image(window, sct):
    if not window:
        return None

    # detect if window is already active
    window.restore()
    window.activate()

    bbox = window.box
    # convert to (x1, y1, x2, y2)
    bbox = (bbox.left, bbox.top, bbox.left + bbox.width, bbox.top + bbox.height)
    # remove window border
    bbox = (bbox[0] + 15, bbox[1] + 64, bbox[2] - 16, bbox[3] - 16)
    screenshot = sct.grab(bbox)
    frame = np.array(screenshot)
    return frame


def tail(filename, n=1):
    with open(filename, "rb") as file:
        file.seek(0, 2)  # move to the end of the file
        filesize = file.tell()
        lines_found = []
        while filesize > 0 and len(lines_found) <= n:
            file.seek(filesize - 1)
            next_char = file.read(1)
            if next_char == b"\n":
                lines_found.append(file.readline().decode())
            filesize -= 1
        if filesize == 0:
            file.seek(0)
            lines_found.append(file.readline().decode())
        return lines_found[-n:]


def tail_until(filename, last_id):
    with open(filename, "rb") as file:
        file.seek(0, 2)  # seek to end of file
        filesize = file.tell()
        lines_found = []
        ignore_last_line = True
        while filesize > 0:
            file.seek(filesize - 1)
            next_char = file.read(1)
            if next_char == b"\n":
                if ignore_last_line:
                    ignore_last_line = False
                else:
                    line = file.readline().decode()
                    current_id = int(line.split(":")[0])
                    if current_id <= last_id:
                        break
                    lines_found.append(line)
            filesize -= 1

        if len(lines_found) == 0:
            new_last_id = last_id
        else:
            new_last_id = int(lines_found[0].split(":")[0])
        return lines_found, new_last_id


class ActionSpace:
    def __init__(self, action_list) -> None:
        self.action_list = action_list
        self.n = len(action_list)

    def sample(self):
        action = np.random.randint(self.n)
        return action


class IndependentActionSpace:
    def __init__(self, independent_action_list) -> None:
        self.independent_action_list = independent_action_list
        self.dim = len(independent_action_list)
        self.choices_per_dim = 2

    def sample(self):
        action = np.random.randint(0, self.choices_per_dim, self.dim)
        return action

    def explain(self, action):  # explain action as chars to keyboard commands
        action_list = []
        for i in range(self.dim):
            if action[i] == 1:
                action_list.append(self.independent_action_list[i])
        return "+".join(action_list)

    def place_holder_action(self):
        return np.zeros(self.dim, dtype=np.int32)


class RewardScaler:
    """
    Scale rewards based on whether enemy is a major target or not
    Major targets: 1x reward
    Minor enemies: 0.3x reward
    """

    def __init__(self, boss_name):
        self.boss_name = boss_name
        self.boss_name_to_target_entities_dict = {
            "GodTamer": ["Lobster"],
            "HornetProtector": ["Hornet Boss 1"],
            "HornetSentinel": ["Hornet Boss 2"],
            "MegaMossCharger": ["Mega Moss Charger"],
            "MantisLords": ["Mantis Lord", "Mantis Lord S1", "Mantis Lord S2"],
            "BattleSisters": ["Mantis Lord", "Mantis Lord S1", "Mantis Lord S2", "Mantis Lord S3"],
            "MageLord": ["Mage Lord", "Mage Lord Phase2"],
            "Mawlek": ["Mawlek Body"],
            "HKPrime": ["HK Prime"],
            "GrimmBoss": ["Grimm Boss"],
            "FlukeMother": ["Fluke Mother"],
        }
        assert boss_name in self.boss_name_to_target_entities_dict, f"Boss name not supported: {boss_name}"
        self.major_targets = set(self.boss_name_to_target_entities_dict[boss_name])

    def get_scale_factor(self, enemy_name) -> float:
        """
        Return reward scale factor for the given enemy
        """
        if enemy_name in self.major_targets:
            return 1.0
        else:
            return 0.3


@ray.remote(resources={"env_runner": 1})
class HKEnv:
    """
    Hollow Knight Gym-like Environment
    Default observation space: (711, 1275, 3)
    """

    def __init__(
        self, boss_name, obs_size=None, color_convert=True, target_fps=15, possible_actions=None, debug_log_path=None
    ) -> None:
        self.boss_name = boss_name

        self.reward_scaler = RewardScaler(boss_name)

        # base attack is related to charm configuration, see:
        # https://hollowknight.fandom.com/wiki/Damage_Values_and_Enemy_Health_(Hollow_Knight)
        self.attack_normalize_factor = 32
        self.healing_normalize_factor = 1

        self.health = None  # initialized in reset()

        # action space
        self.independent_action_list = possible_actions
        self.action_space = IndependentActionSpace(self.independent_action_list)

        # if the last episode is a win battle
        self.win_last_battle = False

        # game window
        self.game_window = get_game_window("Hollow Knight")
        if not self.game_window:
            assert False, f"Hollow Knight not found."
        self.sct = mss.mss()

        # log file
        if debug_log_path is not None:
            self.log_file_path = debug_log_path
        else:
            self.log_file_path = (
                os.getenv("APPDATA").replace("Roaming", "LocalLow")
                + "\\Team Cherry\\Hollow Knight"
                + "\\custom_log.log"
            )
        assert os.path.exists(self.log_file_path), f"Log file not found: {self.log_file_path}"

        # env config
        self.obs_size = obs_size
        self.color_convert = color_convert
        self.target_time_per_frame = 1.0 / target_fps

        # cache the last action, judge if the action is the same as last step
        # used in send_action()
        self.last_action = self.action_space.place_holder_action()

    def send_action(self, action, last_action):
        action_penalty = 0
        for i in range(len(action)):
            if action[i] != last_action[i]:
                if action[i] == 1:
                    keyboard.press(self.independent_action_list[i])  # 0 -> 1
                else:
                    keyboard.release(self.independent_action_list[i])  # 1 -> 0
                    if self.independent_action_list[i] == "j" or self.independent_action_list[i] == "i":
                        action_penalty += 0.01  # small penalty for attack/spell
        return action_penalty

    def release_action(self):
        for i in range(len(self.independent_action_list)):
            keyboard.release(self.independent_action_list[i])

    def get_last_log_step(self):
        last_lines = tail(self.log_file_path, 1)
        return int(last_lines[0].split(":")[0])

    def read_last_lines(self):
        last_lines, last_id = tail_until(self.log_file_path, self.last_log_step)
        self.last_log_step = last_id
        return last_lines

    def wait_for_scene_change(self, log_step_before, need_at_GG_workshop, max_wait_time=5.0, check_interval=0.1):
        """
        Wait for a scene change event in the log file.

        Args:
            log_step_before: The log step ID before the action that triggers scene change
            need_at_GG_workshop: If True, wait for GG_Workshop scene. If False, wait for any non-GG_Workshop scene
            max_wait_time: Maximum time to wait in seconds
            check_interval: How often to check the log file in seconds

        Returns:
            bool: True if target scene was detected, False if timeout
        """
        elapsed_time = 0.0

        while elapsed_time < max_wait_time:
            time.sleep(check_interval)
            elapsed_time += check_interval

            # Read new log lines since the action
            new_lines, _ = tail_until(self.log_file_path, log_step_before)

            # Check for scene change
            for line in new_lines:
                if "<SceneChanged>" in line:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        scene_name = parts[1].strip().strip("<>")

                        # Check if this is the scene we're looking for
                        if need_at_GG_workshop:
                            if scene_name == "GG_Workshop":
                                return True
                        else:
                            # any non-GG_Workshop scene
                            if scene_name != "GG_Workshop":
                                return True

        # Timeout: warn and return False
        target_desc = "GG_Workshop" if need_at_GG_workshop else "non-GG_Workshop scene"
        print(f"WARNING: Scene change timeout after {max_wait_time}s, expected {target_desc}")
        return False

    def reset(
        self, first_run=False, wait=0
    ):  # wait should be 6, coverd by training time, should be specified in training script
        # Game interaction part >>>
        self.release_action()
        if not first_run:
            # Wait for scene change to GG_Workshop (max 3 seconds)
            # self.wait_for_scene_change(
            #     self.last_terminate_log_step, need_at_GG_workshop=True, max_wait_time=10.0
            # )  # long time-out for some speical animation
            # waiting for the scene is ready to take inputs
            time.sleep(2.0)

            self.game_window.restore()
            self.game_window.activate()
            keyboard.press_and_release("w")
            # for standing up
            time.sleep(2.5)

        self.game_window.restore()
        self.game_window.activate()
        keyboard.press_and_release("w")
        time.sleep(1)
        self.game_window.restore()
        self.game_window.activate()

        # Get log step before pressing 'k' to start battle
        log_step_before_battle = self.get_last_log_step()
        keyboard.press_and_release("k")

        # Wait for scene change to battle scene (any non-GG_Workshop scene)
        self.wait_for_scene_change(
            log_step_before_battle,
            need_at_GG_workshop=False,
            max_wait_time=5.0,  # Any non-GG_Workshop scene
        )

        frame = capture_window_image(self.game_window, self.sct)
        # <<< Game interaction part

        # Environment part >>>
        self.last_log_step = self.get_last_log_step()
        self.health = 9
        self.epsisode_step = 0
        print(f"Resetting from LogID: {self.last_log_step}")

        if self.obs_size is not None:
            frame = cv2.resize(frame, self.obs_size)
        if self.color_convert:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_timer = time.time()

        # reset last action cache
        self.last_action = self.action_space.place_holder_action()
        # <<< Environment part

        return frame, {"episode_frame_number": self.epsisode_step}

    def precise_sleep(self, duration):  # accurate sleep, use time.sleep in Ray is not accurate
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < duration:
            pass

    def parse_log_line(self, line):
        """
        Parse a single log line and return state changes.

        Returns:
            tuple: (delta_reward, current_health, life_loss, battle_ended, release_keys)
                - delta_reward: float, reward change from this line
                - current_health: int or None, current health if updated, None otherwise
                - life_loss: bool, True if player took damage
                - battle_ended: bool, True if scene changed back to GG_Workshop
                - release_keys: bool, True if should release keyboard immediately
        """
        delta_reward = 0.0
        current_health = None
        life_loss = False
        battle_ended = False

        if "<SceneChanged>" in line:  # scene change
            # Format: <SceneChanged>|<scene_name>
            parts = line.split("|")
            if len(parts) >= 2:
                scene_name = parts[1].strip().strip("<>")
                # This is correct is this function is called in step()
                if scene_name == "GG_Workshop":
                    battle_ended = True
                    # release keys immediately when scene changes back
                    self.release_action()

        elif "<EnableEnemy>" in line:  # enemy spawn
            # Format: <EnableEnemy>|<name>|<hp>
            # This is just for tracking, no reward/penalty
            pass

        elif "<EnemyHealthChange>" in line:  # hit enemy
            # Format: <EnemyHealthChange>|<name>|<currentHP>|<deltaHP>
            parts = line.split("|")
            if len(parts) >= 4:
                target_name = parts[1].strip().strip("<>")
                enemy_current_health = int(parts[2].strip().strip("<>"))
                delta_hp = int(parts[3].strip().strip("<>"))

                # Only reward for damage dealt (negative delta_hp), ignore healing
                if delta_hp < 0:
                    normalized_delta_health = delta_hp / self.attack_normalize_factor
                    scale_factor = self.reward_scaler.get_scale_factor(target_name)
                    delta_reward += -1 * normalized_delta_health * scale_factor  # enemy health decrease -> reward

        elif "<DamageTaken>" in line:  # player hurt by enemy
            # Format: <DamageTaken>|<currentHealth>|<damageAmount>
            # Note: damageAmount is negative (e.g., -1, -2)
            parts = line.split("|")
            if len(parts) >= 3:
                current_health = int(parts[1].strip().strip("<>"))
                damage_amount = int(parts[2].strip().strip("<>"))  # negative value
                life_loss = True

        elif "<Heal>" in line:  # player heal
            # Format: <Heal>|<currentHealth>|<amount>
            parts = line.split("|")
            if len(parts) >= 3:
                current_health = int(parts[1].strip().strip("<>"))
                amount = int(parts[2].strip().strip("<>"))
                normalized_amount = amount / self.healing_normalize_factor
                delta_reward += 2 * normalized_amount

        return delta_reward, current_health, life_loss, battle_ended

    def step(self, action):
        self.epsisode_step += 1

        # FPS control
        time_diff = time.time() - self.frame_timer
        if time_diff < self.target_time_per_frame:
            self.precise_sleep(self.target_time_per_frame - time_diff)

        self.frame_timer = time.time()

        frame = capture_window_image(self.game_window, self.sct)

        # # send action immediately after frame capture
        action_penalty = self.send_action(action, self.last_action)

        # read new lines in log file
        last_lines = self.read_last_lines()
        # calc reward
        # reward = -action_penalty
        reward = 0
        battle_ended = False
        life_loss = False

        # Parse all log lines and aggregate results
        for line in last_lines:  # all the .strip() are removing "\r\n"
            delta_reward, current_health, line_life_loss, line_battle_ended = self.parse_log_line(line)

            # Accumulate reward (additive)
            reward += delta_reward

            # Update health directly from log (not delta-based)
            if current_health is not None:
                self.health = current_health

            # life_loss is True if ANY line has damage (OR logic)
            life_loss = life_loss or line_life_loss

            # battle_ended is True if ANY line indicates battle end (OR logic)
            if line_battle_ended:
                battle_ended = True

        # Determine win_battle and terminate based on battle_ended
        win_battle = False
        terminate = False
        if battle_ended:
            terminate = True
            if self.health > 0:
                win_battle = True
                self.win_last_battle = True
            else:
                self.win_last_battle = False

        truncated = False

        if self.obs_size is not None:
            frame = cv2.resize(frame, self.obs_size)
        if self.color_convert:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if terminate:  # prevent meaningless action between episodes
            self.release_action()
            # self.last_terminate_log_step = self.get_last_log_step()

        # update last action cache
        self.last_action = action
        return (
            frame,
            reward,
            terminate,
            truncated,
            {
                "episode_frame_number": self.epsisode_step,
                "win_battle": win_battle,
                "health": self.health,
                "life_loss": life_loss,
            },
        )

    def pause(self):
        self.game_window.restore()
        self.game_window.activate()
        keyboard.press_and_release("esc")

    def resume(self):
        self.game_window.restore()
        self.game_window.activate()
        keyboard.press_and_release("esc")


class LocalAbstractHKEnv:
    def __init__(self, env):
        self.first_run = True
        self.env = env

    def release_action(self):
        self.env.release_action.remote()

    def reset(self):
        obs, info = ray.get(self.env.reset.remote(first_run=self.first_run))
        self.first_run = False
        return obs, info

    def step(self, action):
        return ray.get(self.env.step.remote(action))


if __name__ == "__main__":
    # Test log parsing functionality without game interaction
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_log_path = "test_input.txt"

    print(f"Testing log parsing with: {demo_log_path}")

    # Create env instance with debug log path (no game window needed for parsing test)
    # Get the original class from Ray's ActorClass wrapper
    original_class = (
        HKEnv._ray_original_class if hasattr(HKEnv, "_ray_original_class") else HKEnv.__ray_metadata__.modified_class
    )
    env = original_class(
        boss_name="GrimmBoss",
        obs_size=(64, 64),
        possible_actions=["w", "a", "s", "d", "j", "k", "l", "i"],
        debug_log_path=demo_log_path,
    )

    # Test log reading functions
    print("\n=== Testing tail() function ===")
    last_lines = tail(demo_log_path, 3)
    print(f"Last 3 lines:")
    for line in last_lines:
        print(f"  {line.strip()}")

    print("\n=== Testing get_last_log_step() ===")
    last_step = env.get_last_log_step()
    print(f"Last log step: {last_step}")

    print("\n=== Testing tail_until() function ===")
    # Simulate reading from step 300
    lines_since_300, new_last_id = tail_until(demo_log_path, 300)
    print(f"Lines since step 300: {len(lines_since_300)} lines")
    print(f"New last ID: {new_last_id}")
    if lines_since_300:
        print(f"First line: {lines_since_300[-1].strip()}")
        print(f"Last line: {lines_since_300[0].strip()}")

    print("\n=== Testing log parsing logic on real data ===")
    # Initialize trackers
    env.scene_tracker.reset()
    env.health = 9

    # Read all lines from demo.txt
    with open(demo_log_path, "r") as f:
        all_lines = f.readlines()

    print(f"Total lines in log file: {len(all_lines)}")

    # Parse all real logs and write to output.txt
    output_path = os.path.join(script_dir, "output.txt")
    total_reward = 0
    battle_ended = False
    win_battle = False

    with open(output_path, "w", encoding="utf-8") as out_f:
        for line in all_lines:
            # Write original line
            out_f.write(f"RAW: {line.strip()}\n")

            # Use the parse_log_line function
            delta_reward, current_health, life_loss, line_battle_ended, release_keys = env.parse_log_line(line)

            # Accumulate results
            total_reward += delta_reward
            if current_health is not None:
                env.health = current_health

            if line_battle_ended:
                battle_ended = True
                if env.health > 0:
                    win_battle = True

            # Write parsed result
            out_f.write(
                f"PARSED: delta_reward={delta_reward:.3f}, current_health={current_health}, "
                f"life_loss={life_loss}, battle_ended={line_battle_ended}, "
                f"release_keys={release_keys}, health={env.health}, cumulative_reward={total_reward:.3f}\n"
            )

    print(f"Output written to: {output_path}")
    print(f"\n=== Parsing Statistics ===")
    print(f"Total lines processed: {len(all_lines)}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Final health: {env.health}")
    print(f"Battle ended: {battle_ended}")
    print(f"Win battle: {win_battle}")

    print("\n=== Test completed successfully! ===")
