import diffuser.utils as utils
import torch

def main(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    # logger.remove('*.pkl')
    # logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.log_text("""
                    charts:
                    - yKey: loss
                      xKey: steps
                    - yKey: a0_loss
                      xKey: steps
                    """, filename=".charts.yml", dedent=True, overwrite=True)

    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)
    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#

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
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
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

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    if Config.diffusion == 'models.GaussianInvDynDiffusion':
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

    else:
        model_config = utils.Config(
            Config.model,
            savepath='model_config.pkl',
            horizon=Config.horizon,
            transition_dim=observation_dim + action_dim,
            cond_dim=observation_dim,
            dim_mults=Config.dim_mults,
            returns_condition=Config.returns_condition,
            dim=Config.dim,
            condition_dropout=Config.condition_dropout,
            calc_energy=Config.calc_energy,
            device=Config.device,
        )

        diffusion_config = utils.Config(
            Config.diffusion,
            savepath='diffusion_config.pkl',
            horizon=Config.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=Config.n_diffusion_steps,
            loss_type=Config.loss_type,
            clip_denoised=Config.clip_denoised,
            predict_epsilon=Config.predict_epsilon,
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

    # ---------------------------------------------------------------------------------------#
    # -------------------------------- instantiate base_stage--------------------------------#
    # ---------------------------------------------------------------------------------------#

    base_stage_model = base_stage_model_config()
    refinement_stage_model = refinement_stage_model_config()

    base_stage_diffusion = base_stage_diffusion_config(base_stage_model)

    base_stage_trainer = base_stage_trainer_config(base_stage_diffusion, dataset, renderer)

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(base_stage_model)

    logger.print('Testing forward...', end=' ', flush=True)
    batch = utils.batchify(dataset[0], Config.device)
    loss, _ = base_stage_diffusion.loss(*batch)
    loss.backward()
    logger.print('✓')

    # -------------------------------------------------------------------------------------------#
    # --------------------------------- main loop for base_stage---------------------------------#
    # -------------------------------------------------------------------------------------------#

    n_epochs = int(Config.n_train_steps // Config.n_steps_per_epoch)

    for i in range(n_epochs):
        logger.print(f'Epoch {i} / {n_epochs} | {logger.prefix}')
        base_stage_trainer.train(n_train_steps=Config.n_steps_per_epoch)
    
    print("Done training base stage")
    
    # -------------------------------------------------------------------------------------------------#
    # -------------------------------- instantiate for refinement_stage--------------------------------#
    # -------------------------------------------------------------------------------------------------#
    
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

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(refinement_stage_model)

    logger.print('Testing forward...', end=' ', flush=True)
    batch = utils.batchify(dataset[0], Config.device)
    loss, _ = refinement_stage_diffusion.loss(*batch)
    loss.backward()
    logger.print('✓')

    # -------------------------------------------------------------------------------------------------#
    # --------------------------------- main loop for refinement_stage---------------------------------#
    # -------------------------------------------------------------------------------------------------#

    n_epochs = int(Config.n_train_steps // Config.n_steps_per_epoch)

    for i in range(n_epochs):
        logger.print(f'Epoch {i} / {n_epochs} | {logger.prefix}')
        refinement_stage_trainer.train(n_train_steps=Config.n_steps_per_epoch)



