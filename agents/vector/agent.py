import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .world_model import WorldModel
from .policy import ActorCritic

from utils.replay_buffer import ReplayBuffer


class Agent(nn.Module):
    def __init__(
        self,
        action_dims,
        num_objects,
        latent_width,
        transformer_max_length,
        transformer_hidden_dim,
        transformer_num_layers,
        transformer_num_heads,
        policy_num_layers,
        policy_hidden_dim,
        gamma,
        lambd,
        entropy_coef,
    ) -> None:
        super().__init__()
        self.action_dims = action_dims

        self.world_model = WorldModel(
            action_dims=action_dims,
            num_objects=num_objects,
            latent_width=latent_width,
            transformer_max_length=transformer_max_length,
            transformer_hidden_dim=transformer_hidden_dim,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
        )
        self.actor_critic = ActorCritic(
            feat_dim=(latent_width * latent_width + transformer_hidden_dim) * num_objects,
            num_layers=policy_num_layers,
            hidden_dim=policy_hidden_dim,
            action_dims=action_dims,
            gamma=gamma,
            lambd=lambd,
            entropy_coef=entropy_coef,
        )

        self.update_steps = -1
        self.ppo_ratio = 4

    def sample_policy(self, context_state, context_action, greedy=False):
        self.world_model.eval()
        self.actor_critic.eval()
        with torch.no_grad():
            model_context_state = torch.stack(list(context_state), dim=0).unsqueeze(0)
            context_latent = self.world_model.encode_obs(model_context_state)
            context_latent, current_latent = (
                context_latent[:, :-1],
                context_latent[:, -1:],
            )  # first dim is batch_size=1, second dim is batch_length
            model_context_action = np.stack(list(context_action), axis=0)
            model_context_action = torch.Tensor(model_context_action).unsqueeze(0).cuda()
            last_hidden = self.world_model.calc_last_hidden(context_latent, model_context_action)
            action = self.actor_critic.sample_as_env_action(torch.cat([current_latent, last_hidden], dim=-1), greedy)

            action = action.flatten()  # HK 1, 1, 7 -> 7 or Atari 1, 1, 1 -> 1
        return action

    def update(
        self,
        replay_buffer: ReplayBuffer,
        training_steps,
        batch_size,
        batch_length,
        imagine_batch_size,
        imagine_context_length,
        imagine_batch_length,
        logger,
    ):
        self.update_steps += 1

        # train world model
        self.world_model.train()
        obs, action, reward, termination = replay_buffer.sample(batch_size, batch_length)
        self.world_model.update(obs, action, reward, termination, logger=logger)

        if self.update_steps % self.ppo_ratio == 0:
            # imagine data
            self.world_model.eval()
            self.actor_critic.eval()
            sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
                imagine_batch_size * self.ppo_ratio, imagine_context_length
            )
            with torch.no_grad():
                latent, action, log_prob, reward_hat, termination_hat = self.world_model.imagine_data(
                    self.actor_critic,
                    sample_obs,
                    sample_action,
                    imagine_batch_size=imagine_batch_size * self.ppo_ratio,  # 512 * 4
                    imagine_batch_length=imagine_batch_length,
                    log_video=False,
                    logger=logger,
                )

            # train actor critic
            self.actor_critic.train()
            for i in range(self.ppo_ratio):  # for in range(4)
                self.actor_critic.update(
                    latent=latent[i * imagine_batch_size : (i + 1) * imagine_batch_size],
                    action=action[i * imagine_batch_size : (i + 1) * imagine_batch_size],  # [B, L, 1]
                    old_log_prob=log_prob[i * imagine_batch_size : (i + 1) * imagine_batch_size],
                    reward=reward_hat[i * imagine_batch_size : (i + 1) * imagine_batch_size],
                    termination=termination_hat[i * imagine_batch_size : (i + 1) * imagine_batch_size],
                    logger=logger,
                )
