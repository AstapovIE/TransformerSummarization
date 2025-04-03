import torch

def subsequent_mask(size, device):
    mask = torch.ones(size, size, device=device).triu_()
    return mask.unsqueeze(0) == 0

def make_mask(source_inputs, target_inputs, pad_idx, device):
    source_mask = (source_inputs != pad_idx).unsqueeze(-2)
    target_mask = (target_inputs != pad_idx).unsqueeze(-2)
    target_mask = target_mask & subsequent_mask(target_inputs.size(-1), device).type_as(target_mask)
    return source_mask, target_mask


def convert_batch(batch, device, pad_idx=1):
    source_inputs, target_inputs = batch.source.transpose(0, 1), batch.target.transpose(0, 1)
    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx, device)

    return source_inputs, target_inputs, source_mask, target_mask


def new_convert_batch(batch, device, pad_idx=1):
    source_inputs = batch.source.transpose(0, 1)
    target_inputs = batch.target.transpose(0, 1)

    # Маска источника (учитывает только pad-токены)
    source_mask = (source_inputs != pad_idx).unsqueeze(1)

    return source_inputs, target_inputs, source_mask