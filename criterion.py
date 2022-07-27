from torch import nn

def criterion(inputs, target, join=False, mode='CE', balance_factor=0.5):
    losses = {}
    for name, x in inputs.items():
        if mode == 'CE':
            losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if join:
        return losses['out'] + balance_factor * losses['backbone']
    else:
        return losses['out']