import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as np
import os
import gym
from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output

def evaluate(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda'

    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'

    loadpath = os.path.join(Config.bucket, logger.prefix, 'base_stage', 'checkpoint')
    
    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_{self.step}.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')
    
    state_dict = torch.load(loadpath, map_location=Config.device)

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
    )

    render_config = utils.Config(
        Config.renderer,
        savepath='render_config.pkl',
        env=Config.dataset,
    )

    dataset = dataset_config()
    renderer = render_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    base_stage_model_config = utils.Config(
        Config.model,
        savepath='base_stage_model_config.pkl',
        horizon=Config.horizon,
        transition_dim=observation_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        returns_condition=Config.returns_condition,
        dim=Config.dim,
        condition_dropout=Config.condition_dropout,
        calc_energy=Config.calc_energy,
        device=Config.device,
        if_base = True,
    )

    refinement_stage_model_config = utils.Config(
        Config.model,
        savepath='refinement_stage_model_config.pkl',
        horizon=Config.horizon,
        transition_dim=observation_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        returns_condition=Config.returns_condition,
        dim=Config.dim,
        condition_dropout=Config.condition_dropout,
        calc_energy=Config.calc_energy,
        device=Config.device,
        if_base = False,
    )

    base_stage_diffusion_config = utils.Config(
        Config.diffusion,
        savepath='base_stage_diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        ar_inv=Config.ar_inv,
        train_only_inv=Config.train_only_inv,
        hidden_dim=Config.hidden_dim,
        if_base = True,
        clip_mode=Config.clip_mode,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        condition_guidance_w=Config.condition_guidance_w,
        device=Config.device,
    )

    base_stage_trainer_config = utils.Config(
        utils.Trainer,
        savepath='base_stage_trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
        save_checkpoints=Config.save_checkpoints,
    )


    base_stage_model = base_stage_model_config()
    refinement_stage_model = refinement_stage_model_config()

    base_stage_diffusion = base_stage_diffusion_config(base_stage_model)

    base_stage_trainer = base_stage_trainer_config(base_stage_diffusion, dataset, renderer)

    base_stage_trainer = base_stage_trainer_config(base_stage_diffusion, dataset, renderer)
    
    logger.print(utils.report_parameters(base_stage_model), color='green')
    base_stage_trainer.step = state_dict['step']
    base_stage_trainer.model.load_state_dict(state_dict['model'])
    base_stage_trainer.ema_model.load_state_dict(state_dict['ema'])

    loadpath = os.path.join(Config.bucket, logger.prefix, 'refinement_stage', 'checkpoint')
    
    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_{self.step}.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')
    
    state_dict = torch.load(loadpath, map_location=Config.device)
    
    refinement_stage_diffusion_config = utils.Config(
        Config.diffusion,
        base_stage_ema_model = base_stage_trainer.ema_model,
        savepath='refinement_stage_diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        ar_inv=Config.ar_inv,
        train_only_inv=Config.train_only_inv,
        hidden_dim=Config.hidden_dim,
        if_base = False,
        clip_mode=Config.clip_mode,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        condition_guidance_w=Config.condition_guidance_w,
        device=Config.device,
    )
    
    refinement_stage_trainer_config = utils.Config(
        utils.Trainer,
        savepath='refinement_stage_trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
        save_checkpoints=Config.save_checkpoints,
    )

    refinement_stage_diffusion = refinement_stage_diffusion_config(refinement_stage_model)
    
    refinement_stage_trainer = refinement_stage_trainer_config(refinement_stage_diffusion, dataset, renderer)

    logger.print(utils.report_parameters(refinement_stage_model), color='green')
    refinement_stage_trainer.step = state_dict['step']
    refinement_stage_trainer.model.load_state_dict(state_dict['model'])
    refinement_stage_trainer.ema_model.load_state_dict(state_dict['ema'])

    num_eval = 10
    device = Config.device

    env_list = [gym.make(Config.dataset) for _ in range(num_eval)]
    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]

    assert refinement_stage_trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    returns = to_device(Config.test_ret * torch.ones(num_eval, 1), device)

    t = 0
    obs_list = [env.reset()[None] for env in env_list]
    obs = np.concatenate(obs_list, axis=0)
    recorded_obs = [deepcopy(obs[:, None])]

    while sum(dones) <  num_eval:
        obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=device)}
        samples = refinement_stage_trainer.ema_model.conditional_sample(conditions, returns=returns)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2*observation_dim)
        action = refinement_stage_trainer.ema_model.inv_model(obs_comb)

        samples = to_np(samples)
        action = to_np(action)

        action = dataset.normalizer.unnormalize(action, 'actions')

        if t == 0:
            normed_observations = samples[:, :, :]
            observations = dataset.normalizer.unnormalize(normed_observations, 'observations')
            savepath = os.path.join('images', 'sample-planned.png')
            renderer.composite(savepath, observations)

        obs_list = []
        for i in range(num_eval):
            this_obs, this_reward, this_done, _ = env_list[i].step(action[i])
            obs_list.append(this_obs[None])
            if this_done:
                if dones[i] == 1:
                    pass
                else:
                    dones[i] = 1
                    episode_rewards[i] += this_reward
                    logger.print(f"Episode ({i}): {episode_rewards[i]}", color='green')
            else:
                if dones[i] == 1:
                    pass
                else:
                    episode_rewards[i] += this_reward

        obs = np.concatenate(obs_list, axis=0)
        recorded_obs.append(deepcopy(obs[:, None]))
        t += 1

    recorded_obs = np.concatenate(recorded_obs, axis=1)
    savepath = os.path.join('images', f'sample-executed.png')
    renderer.composite(savepath, recorded_obs)
    episode_rewards = np.array(episode_rewards)

    logger.print(f"average_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}", color='green')
    logger.log_metrics_summary({'average_ep_reward':np.mean(episode_rewards), 'std_ep_reward':np.std(episode_rewards)})
