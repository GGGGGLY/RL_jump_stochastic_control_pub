"""Semi-gradient TD targets with the optional correction in manuscript (3.17)."""

import torch


def corrected_td_residual(reward, value, next_value, value_gradient,
                          diffusion, jump_value, compensated_jump, discount):
    """Subtract Brownian and compensated jump increments from the TD error.

    ``jump_value - value`` contains one value increment per jump channel. For
    multiple independent channels, pass a [batch, channels] tensor of increments
    by broadcasting the scalar current value. ``diffusion`` is sigma*dW, not
    sigma alone. Only -value contributes gradients to the critic; the actor must
    detach this residual before using it in a score-function loss.
    """
    with torch.no_grad():
        brownian = (value_gradient * diffusion).sum(dim=-1, keepdim=True)
        jumps = ((jump_value - value) * compensated_jump).sum(dim=-1, keepdim=True)
        target = reward + discount * next_value - brownian - jumps
    return target - value
