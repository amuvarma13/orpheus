from torch.utils.data import DataLoader, Dataset
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
from torch.utils.data.distributed import DistributedSampler
import torch
from transformers import Trainer
import wandb

class FSDPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_train_dataloader(self):
        sampler = AlternatingDistributedSampler(
            self.train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=False, 
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def log(self, logs, callback=None):
        super().log(logs)
        if self.is_world_process_zero():
            global_step = self.state.global_step
            if global_step % 2 == 0:
                wandb.log({"text_loss": logs["loss"], "step": global_step})
            else:
                wandb.log({"audio_loss": logs["loss"], "step": global_step})

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        self.save_and_push_model(output_dir)
 
    def save_and_push_model(self, output_dir):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state_dict = self.model.state_dict()
        self.model.save_pretrained(output_dir, state_dict=cpu_state_dict)

class AlternatingDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

class BatchedAlternatingDataset(Dataset):
    def __init__(self, dataset1, dataset2, batch_total):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_total = batch_total
        self.length = 2 * min(len(dataset1), len(dataset2))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        super_batch = index // (2 * self.batch_total)
        position_in_super_batch = index % (2 * self.batch_total)
        
        if position_in_super_batch < self.batch_total:
            dataset_index = super_batch * self.batch_total + position_in_super_batch
            return self.dataset1[dataset_index]
        else:
            dataset_index = super_batch * self.batch_total + (position_in_super_batch - self.batch_total)
            return self.dataset2[dataset_index]
