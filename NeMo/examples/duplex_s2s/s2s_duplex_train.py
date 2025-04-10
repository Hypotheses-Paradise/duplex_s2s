# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import hydra
import torch
from lightning.pytorch import Callback, Trainer

from nemo.collections.duplex_s2s.data.datamodule import S2SDataModule
from nemo.collections.duplex_s2s.models.duplex_s2s_model import DuplexS2SModel,DuplexS2SModelSpeechDecoder
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# During the training, the checkpoint format is standard PTL ckpt
# After the training -> convert to HF instead of .nemo ?
# Add a callback that does the above conversion at every checkpoint save


class PROFILING(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch, batch_idx: int
    ) -> None:
        if batch_idx == 10:
            print("STARTING PROFILE")
            #import nvtx
            #self.profile = nvtx.Profile()
            #self.profile.enable()
            torch.cuda.profiler.cudart().cudaProfilerStart()

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int
    ) -> None:
        if batch_idx == 15:
            print("STOPPING PROFILE")
            #self.profile.disable()
            torch.cuda.profiler.cudart().cudaProfilerStop()


# @hydra.main(config_path="conf", config_name="s2s_duplex")
@hydra_runner(config_path="conf", config_name="s2s_duplex")
def train(cfg):
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    # TODO: decide on exp_manager or adopting NeMo 2.0 API with _setup function, or sth else ?
    trainer = Trainer(
        **resolve_trainer_cfg(cfg.trainer),
        #callbacks=[PROFILING()],
    )
    exp_manager(trainer, cfg.get("exp_manager", None))

    with trainer.init_module():
    #    model = DuplexS2SModel(cfg.model)
        model = DuplexS2SModelSpeechDecoder(cfg.model)

    # exp_manager / NeMo2 _setup provide:
    # * PEFT (possibly from HF)
    # * save/load checkpoint (exp_manager -> .nemo only)
    # * resume
    # * W&B loggers etc
    datamodule = S2SDataModule(cfg.data, tokenizer=model.tokenizer)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
