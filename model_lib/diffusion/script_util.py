import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    predict_prev=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    p2_gamma=0,
    p2_k=1,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    if predict_xstart:
        pred = gd.ModelMeanType.START_X
    elif predict_prev:
        pred = gd.ModelMeanType.PREVIOUS_X
    else:
        pred = gd.ModelMeanType.EPSILON 
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(pred
            # gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        p2_gamma=p2_gamma,
        p2_k=p2_k,
    )



def create_sft_gaussian_diffusion(  # create shifted gaussian diffusion for clip prior
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    predict_prev = False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    p2_gamma=0,
    p2_k=1,
    vocab = None,
    beta_min = 0.05,
    beta_max = 0.1,
    mean_path = None,  # path to load mean file
    std_path = None,  # path to load std file
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps, beta_min=beta_min, beta_max=beta_max)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    if predict_xstart:
        pred = gd.ModelMeanType.START_X
    elif predict_prev:
        pred = gd.ModelMeanType.PREVIOUS_X
    else:
        pred = gd.ModelMeanType.EPSILON 

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(pred
            # gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            # gd.ModelMeanType.PREVIOUS_X if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        p2_gamma=p2_gamma,
        p2_k=p2_k,
        mean_path=mean_path,
        std_path=std_path,
        vocab=vocab,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
