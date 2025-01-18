import sys
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
import time
import torch
import numpy as np
import os
import dill
import hydra

os.environ["WANDB_SILENT"] = "True"

env_path = "/home/robot/UR_Robot_Arm_Show/tele_ws/src/tele_ctrl_jeff/scripts/"
sys.path.append(env_path)
from inference import InferenceEnv

sys.path.append("/home/robot/UR_Robot_Arm_Show")
from utils.arm_robot import ArmRobot
from utils.dataset_playback import send_action_by_p2v

sys.path.append("/home/robot/ArmRobot")
from observation.Image_process_utils import process_image_npy


class Inference:
    """
    The deployment is running on the local computer of the robot.
    """

    def __init__(
        self,
        robot: ArmRobot,
        obs_horizon=2,
        action_horizon=8,
        device="gpu",
    ):
        self.robot = robot

        # camera

        args = {"fps": 20, "visualize": False, "path": ""}
        self.obs_env = InferenceEnv(args)
        print("self.obs_env: ", self.obs_env)

        # horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # inference device
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.image_params = {
            "rgb": {
                "size": (960, 540),
                "crop": (230, 0, 770, 540),
                "resize": (240, 240),
            },
            "wrist": {
                "size": (640, 480),
                "crop": (80, 0, 560, 480),
                "resize": (240, 240),
            },
            "scene": {
                "size": (640, 480),
                "crop": (100, 160, 540, 480),
                "resize": (240, 240),
            },
        }
        self.loop_policy = None

    def reset_robot(self):
        print("reset_robot")

        action = [
            -0.1897960292037411,
            -0.38273282360124394,
            0.4351520715693574,
            0.4165155311323526,
            0.9091285998773582,
            3.390982377428181e-05,
            7.790528867994886e-06,
            0.0,
        ]
        self.robot.init_action(action)

        self.obs_list = []

    def construct_obs(self, obs_list):

        if len(obs_list) < self.obs_horizon:
            env_obs = [obs_list[0]] * self.obs_horizon
        else:
            env_obs = obs_list[-self.obs_horizon :]

        wrist_image_list = []
        rgb_image_list = []
        state_list = []
        for obs in env_obs:

            wrist_image = process_image_npy(obs["wrist"], "wrist", self.image_params)
            wrist_image = np.transpose(wrist_image, (2, 0, 1))
            wrist_image_list.append(wrist_image)
            rgb_image = process_image_npy(obs["rgb"], "rgb", self.image_params)
            rgb_image = np.transpose(rgb_image, (2, 0, 1))
            rgb_image_list.append(rgb_image)

            state_list.append(obs["state"])

        wrist = np.stack(wrist_image_list)
        rgb = np.stack(rgb_image_list)
        state = np.stack(state_list)

        model_input = {}
        model_input["wrist_image"] = (
            torch.from_numpy(wrist).unsqueeze(0).to(self.device)
        )
        model_input["image"] = torch.from_numpy(rgb).unsqueeze(0).to(self.device)
        model_input["agent_pos"] = torch.from_numpy(state).unsqueeze(0).to(self.device)

        return model_input

    def get_obs_dict(self):
        data = self.obs_env.get_state_obs()
        data["state"] = self.robot.get_state()

        return data

    def filter_actions(self, actions):
        # actions = actions["action"].detach().cpu().numpy()[0]
        actions = actions["action"].detach().cpu().numpy()[0][2::4]

        """
        diff
        """
        # state = self.robot.get_state()
        # action = actions[0]
        # action[0] += state[0]
        # action[1] += state[1]
        # action[2] += state[2]
        # for i in range(1, len(actions)):
        #     actions[i][0] += actions[i - 1][0]
        #     actions[i][1] += actions[i - 1][1]
        #     actions[i][2] += actions[i - 1][2]

        return actions

    def step_one(self, action):
        # print("step: ", action.shape)
        # print(action[:3])

        self.robot.send_action(action)

    def run(self, policy):
        self.reset = False
        step_count = 0

        obs = self.get_obs_dict()
        self.obs_list.append(obs)
        # print(obs.keys())

        while step_count < 10000:
            if self.reset:
                break
            with torch.no_grad():

                model_input = self.construct_obs(self.obs_list)

                actions = policy.predict_action(model_input)

            actions = self.filter_actions(actions)

            # self.obs_env.send_path(actions, self.robot)
            # obs = self.get_obs_dict()
            # self.obs_list.append(obs)
            # self.obs_list.append(obs)
            for action in actions:
                self.step_one(action)
                time.sleep(0.04)

                obs = self.get_obs_dict()
                self.obs_list.append(obs)
                step_count += 1

            # print(f"step: {step_count}")

        if self.reset:
            self.reset_robot()

    def run_loop(self):
        self.reset = False
        step_count = 0

        obs = self.get_obs_dict()
        self.obs_list.append(obs)

        while step_count < 10000:
            if self.reset:
                self.reset = False
                self.reset_robot()

            if self.loop_policy is None:
                time.sleep(1)
                continue

            with torch.no_grad():

                model_input = self.construct_obs(self.obs_list)

                actions = self.loop_policy.predict_action(model_input)

            actions = self.filter_actions(actions)
            for action in actions:
                self.step_one(action)

                obs = self.get_obs_dict()
                self.obs_list.append(obs)
                step_count += 1


def load_policy(ckpt_path):
    # load checkpoint
    print(f"load: {ckpt_path}")
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    print(f"loaded: {cfg.name}")

    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device("cuda")
    policy.eval().to(device)
    policy.num_inference_steps = 16  # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    return policy


def main(ckpt_path):

    policy = load_policy(ckpt_path)

    print("==== model loaded")
    robot = ArmRobot()

    env = Inference(
        robot=robot,
        obs_horizon=2,
        action_horizon=policy.n_action_steps,
        device="cpu",
    )

    env.reset_robot()

    env.run(policy)


def test():

    print("==== test ==")
    robot = ArmRobot()

    env = Inference(
        robot=robot,
        obs_horizon=2,
        action_horizon=16,
        device="cpu",
    )

    env.reset_robot()

    env.run(None)


def test_env():
    args = {"fps": 20, "visualize": False, "path": ""}

    collector = InferenceEnv(args)
    time.sleep(2)
    for i in range(10):
        data = collector.get_state_obs()
        print(data.keys())
        time.sleep(1)


def test01():
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/cube2bowl_2024.12.28/dex_pt-sample_08.04.33/checkpoints/epoch=0200-train_loss=0.002.ckpt"
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/cube2bowl_2024.12.28/dex_pt-sample_08.04.33/checkpoints/epoch=0150-train_loss=0.003.ckpt"
    # ok
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/cube2bowl_2024.12.28/dex_pt-sample_08.04.33/checkpoints/epoch=0300-train_loss=0.001.ckpt"
    # not good
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/cube2bowl_2024.12.28/dex_pt-sample_08.04.33/checkpoints/epoch=0350-train_loss=0.001.ckpt"
    # overfit
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/cube2bowl_2024.12.28/dex_pt-sample_08.04.33/checkpoints/epoch=0450-train_loss=0.000.ckpt"
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/cube2bowl_2024.12.28/dex_pt-sample_no-crop_16.34.48/checkpoints/epoch=0300-train_loss=0.000.ckpt"
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/cube2bowl_2024.12.28/dex_pt-sample_no-crop_16.34.48/checkpoints/epoch=0150-train_loss=0.001.ckpt"
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/11.15.44_train_cube2bowl_box_no-crop/checkpoints/epoch=0150-train_loss=0.006.ckpt"
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/11.15.44_train_cube2bowl_box_no-crop/checkpoints/epoch=0350-train_loss=0.002.ckpt"
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/11.15.44_train_cube2bowl_box_no-crop/checkpoints/epoch=0550-train_loss=0.001.ckpt"

    # ok
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/dp_240_checkpoints/300.ckpt"
    # file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/dp_240_checkpoints/550.ckpt"
    # file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/dp_240_checkpoints/150.ckpt"

    # bowl bad
    # file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/240_bowl/epoch=0300-train_loss=0.004.ckpt"
    # file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/240_bowl/epoch=0550-train_loss=0.001.ckpt"

    # file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/bowl_2024.12.31/dex_31_cube2bowl_13.24.01/checkpoints/epoch=0300-train_loss=0.010.ckpt"
    # # 60
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/cube2bowl_2024.12.31/dex_no-resize_13.24.15/checkpoints/epoch=0400-train_loss=0.006.ckpt"
    # file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/cube2bowl_2024.12.31/dex_noise_13.24.01/checkpoints/epoch=0400-train_loss=0.006.ckpt"

    # ok bowl
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/bowl_2024.12.31/dex_exp_10.11.20/checkpoints/epoch=0400-train_loss=0.007.ckpt"
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/bowl_2024.12.31/dex_exp_10.11.20/checkpoints/epoch=0200-train_loss=0.014.ckpt"
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/bowl_2024.12.31/dex_exp_10.11.20/checkpoints/epoch=0250-train_loss=0.014.ckpt"
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/bowl_2024.12.31/dex_exp_10.11.20/checkpoints/epoch=0300-train_loss=0.012.ckpt"
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/bowl_2024.12.31/dex_exp_10.11.20/checkpoints/epoch=0550-train_loss=0.003.ckpt"
    main(file)
    # test()


def test02():
    # ok
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/take_all_2025.01.09/epoch=0400-train_loss=0.004.ckpt"
    # file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/take_all_2025.01.09/noise/epoch=0400-train_loss=0.007.ckpt"

    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/take_all_2025.01.09/open/epoch=0400-train_loss=0.015.ckpt"

    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/take_all_2025.01.09/open_take/epoch=0400-train_loss=0.004.ckpt"
    main(file)


def test03():
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/demo_2025.01.14/open/epoch=0400-train_loss=0.026.ckpt"
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/demo_2025.01.14/bell/epoch=0400-train_loss=0.040.ckpt"
    # file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/demo_2025.01.14/mm/epoch=0400-train_loss=0.044.ckpt"
    # file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/demo_2025.01.14/kele/epoch=0400-train_loss=0.040.ckpt"
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/demo_2025.01.14/ship/epoch=0400-train_loss=0.032.ckpt"
    # file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/demo_2025.01.14/snake/epoch=0400-train_loss=0.040.ckpt"
    # file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/demo_2025.01.14/persimmon/epoch=0400-train_loss=0.037.ckpt"
    main(file)


def test03_():
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/demo_2025.01.14/open/epoch=0400-train_loss=0.026.ckpt"
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/demo_2025.01.14/bell/epoch=0400-train_loss=0.040.ckpt"
    main(file)


def test04():
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/dex_c230_diff_12.05.58/epoch=0400-train_loss=0.005.ckpt"
    main(file)


def test05():
    file = "/media/robot/30F73268F87D0FEF/Checkpoints/dp/demo_2025.01.14/open/epoch=0400-train_loss=0.026.ckpt"
    main(file)


if __name__ == "__main__":
    test05()
