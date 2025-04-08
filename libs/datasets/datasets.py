import os
import torch
try:
    from .data_utils import trivial_batch_collator
except ImportError:
    from data_utils import trivial_batch_collator

datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator

def make_dataset(name, is_training, split, **kwargs):
   """
       A simple dataset builder
   """
   dataset = datasets[name](is_training, split, **kwargs)
   return dataset


def make_data_loader(dataset, is_training, generator, batch_size, num_workers, drop_last=None):
    """
        A simple dataloder builder
    """
    if is_training:
        if drop_last is None:
            drop_last = True
    else:
        drop_last = False
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        shuffle=is_training,
        drop_last=drop_last,
        generator=generator,
        persistent_workers=True
    )
    return loader