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


sys.path.append("/home/robot/ArmRobot")
from observation.Image_process_utils import process_image_npy
from hardware.arm_robot import ArmRobot


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
                "resize": (512, 512),
            },
            "wrist": {
                "size": (640, 480),
                "crop": (0, 0, 640, 480),
                "resize": (512, 512),
            },
            "scene": {
                "size": (640, 480),
                "crop": (100, 160, 540, 480),
                "resize": (512, 512),
            },
        }

    def reset_robot(self):

        action = [
            -0.18980697676771513,
            -0.3827291399992533,
            0.31114132703422326,
            -0.417985318626756,
            -0.9083933376685246,
            0.008868446494732254,
            0.005582844140526736,
            0.0,
        ]
        self.robot.init_action(action)

        self.obs_list = []

    def construct_obs(self, obs_list):

        if len(obs_list) < self.obs_horizon:
            env_obs = [obs_list[0]] * self.obs_horizon
        else:
            env_obs = obs_list[-self.obs_horizon :]  # FIXME
        # for k, v in env_obs.items():
        #     print(f"{k} : {v.shape}")

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
        return actions["action"].detach().cpu().numpy()[0]

    def step_one(self, action):
        print("step: ", action.shape)
        self.robot.send_action(action)

    def run(self, policy):
        step_count = 0

        while step_count < 1000:
            with torch.no_grad():

                obs = self.get_obs_dict()
                print(obs.keys())
                self.obs_list.append(obs)

                model_input = self.construct_obs(self.obs_list)

                actions = policy.predict_action(model_input)

            actions = self.filter_actions(actions)

            for action in actions:
                self.step_one(action)
                time.sleep(0.1)
                step_count += 1

            print(f"step: {step_count}")


def main(ckpt_path):

    # load checkpoint

    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    assert "diffusion" in cfg.name

    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device("cuda")
    policy.eval().to(device)
    policy.num_inference_steps = 16  # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

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


if __name__ == "__main__":
    main(
        "/media/robot/30F73268F87D0FEF/Checkpoints/dp/checkpoints/epoch=0150-train_loss=0.016.ckpt"
    )
    # test()
