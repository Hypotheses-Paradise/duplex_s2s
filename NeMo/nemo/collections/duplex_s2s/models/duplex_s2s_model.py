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
import builtins
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import Any

import hydra
import torch
from lightning import LightningModule
from omegaconf import open_dict
from torch import Tensor
from torch.distributed import init_device_mesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from torchmetrics.text import SacreBLEUScore
from transformers import AutoModel, AutoModelForCausalLM

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.duplex_s2s.modules import AudioPerceptionModule
from nemo.collections.duplex_s2s.modules import SpeechDecoder
from nemo.collections.tts.models import AudioCodecModel
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging

class DuplexS2SModel(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self._audio_codec = (
            _load_pretrained(AudioCodecModel, self.cfg.pretrained_audio_codec).to(torch.bfloat16).eval()
        )
        del self._audio_codec.discriminator  # free up some memory
        self._codebook_size = self._audio_codec.vector_quantizer.codebook_size_per_group
        self._num_codebooks = self._audio_codec.vector_quantizer.num_groups

        # We load the pretrained HF LLM using "ForCausalLM" variant so that we can obtain the
        # pretrained LM head weights.
        # However, for S2S we need to access the activations before LM head directly
        # to feed them to the audio codec head.
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        llm = AutoModelForCausalLM.from_pretrained(self.cfg.pretrained_llm).to(torch.bfloat16).train()
        self.llm = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
        self.lm_head = llm.lm_head
        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_tokens = self.llm.embed_tokens
        del self.llm.embed_tokens

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        asr = _load_pretrained(ASRModel, self.cfg.pretrained_asr).to(torch.bfloat16).eval()
        with open_dict(self.cfg):
            self.cfg.perception.preprocessor = asr.cfg.preprocessor
            self.cfg.perception.encoder = asr.cfg.encoder
            self.cfg.perception.output_dim = self.llm.config.hidden_size
        self.perception = AudioPerceptionModule(self.cfg.perception).train()
        self.perception.load_state_dict(asr.state_dict(), strict=False)

        self.embed_audio_tokens = torch.nn.ModuleList(
            [
                # codebook size + 2 (speech BOS and EOS)
                torch.nn.Embedding(self.speech_vocab_size, self.embed_tokens.embedding_dim)
                for _ in range(self._num_codebooks)
            ]
        )
        self.audio_head = torch.nn.Linear(self.llm.config.hidden_size, self.speech_vocab_size * self._num_codebooks)

    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        return self._codebook_size + 2

    @property
    def speech_bos_id(self) -> int:
        return self._codebook_size

    @property
    def speech_eos_id(self) -> int:
        return self._codebook_size + 1

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    def forward(
        self,
        input,
    ) -> dict[str, Tensor]:
        """
        Implements a fully offline forward pass through the entire model.
        The flow is the following:

                                                     |-> |audio_head| -> |audio codes|
        |source speech + prev target text| -> |llm| -|
                                                     |-> |lm_head|    -> |token ids  |
        """
        # input_embeds and out: (B, T, H)
        out = self.llm(inputs_embeds=input["input_embeds"])
        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out['last_hidden_state'])  # (B, T, num_text_tokens)
        audio_logits = self.audio_head(out['last_hidden_state']).view(
            B, T, self.speech_vocab_size, self._num_codebooks
        )
        return {
            "text_logits": text_logits,
            "audio_logits": audio_logits,
        }

    def prepare_inputs(self, batch: dict):
        """
        Performs additional processing on the mini-batch collected from dataloader.
        Notably:
        * Convert source audio to speech representations.
        * Convert target audio to target audio tokens.
        * Convert target text to embeddings.
        * Combine the input audio and target text embeddings.
        * Take care of any necessary slicing to align the shapes of source audio,
            target audio, and target token ids.
        """

        def print(*args, **kwargs):
            if hasattr(self, "device_mesh") and self.device_mesh is not None:
                builtins.print(f"[{self.device_mesh.get_coordinate()}]", *args, **kwargs)
            else:
                builtins.print(f"[{torch.distributed.get_rank()}]", *args, **kwargs)

        # Source audio encoding.
        # Input audio: (B, T_samples)
        # Encoded: (B, T, H)
        source_encoded, source_encoded_lens = self.perception(
            input_signal=batch["source_audio"], input_signal_length=batch["source_audio_lens"]
        )

        # Target text preparation.
        # Target tokens: (B, T)
        target_tokens = batch["target_tokens"]
        if target_tokens.shape[1] < source_encoded.shape[1]:
            pad_id = self.tokenizer.pad
            if pad_id is None:
                pad_id = self.tokenizer.unk_id
            if pad_id is None:
                warnings.warn(
                    "The text tokenizer has no <pad> or <unk> tokens available, using id 0 for padding (this may lead to silent bugs)."
                )
                pad_id = 0
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (torch.ones(source_encoded.shape[0], 1, device=source_encoded.device) * pad_id).to(torch.long),
                ],
                dim=-1,
            )

        # Target audio encoding.
        # Input target audio: (B, T_samples')
        # Output target codes: (B, K, T)
        with torch.no_grad(), _safe_audio_codec_inference():
            target_codes, target_codes_lens = self._audio_codec.encode(
                audio=batch["target_audio"], audio_len=batch["target_audio_lens"]
            )
            target_codes = target_codes.transpose(1, 2)  # (B, K, T) -> (B, T, K)

        # Note: Because we are using separate models for source and target representations,
        #       despite best-effort attempt to align their frame rates, they may be off by a few frames.
        #       We'll fix it by truncating to shortest sequence, and emit a warning if the discrepancy is too high.
        if (tl := target_codes.shape[1]) != (sl := source_encoded.shape[1]):
            if tl < sl:
                diff = sl - tl
                source_encoded = source_encoded[:, :tl]
                target_tokens = target_tokens[:, :tl]
                torch.clamp_(source_encoded_lens, max=tl)
            else:
                diff = tl - sl
                target_codes = target_codes[:, :sl]
                torch.clamp_(target_codes_lens, max=sl)
            if diff > 2:
                logging.warning(
                    f"A mismatch between source ({sl}) and target ({tl}) sequence length greater than 2 detected. "
                    f"This may indicate significant desynchronization in longer sessions."
                )

        # Insert speech BOS and speech EOS after we know input/output text/audio shapes are matching.
        btt = target_tokens[..., None]  # broadcast target tokens to num_codebooks dim
        target_codes = torch.where(btt == self.text_bos_id, self.speech_bos_id, target_codes)
        target_codes = torch.where(btt == self.text_eos_id, self.speech_eos_id, target_codes)

        # Combine target audio and text into a single tensor to slice them together.
        # It will also help us truncate the sequence lengths to be divisible by TP world size,
        # when TP is enabled.
        # Input ids: (B, T, K+1)
        input_ids = torch.cat([target_codes, target_tokens[..., None]], dim=-1)
        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_ids.shape[1] - 1) % tp_world_size) != 0:
                # Truncate some tokens from the end to make the sequence lenght shape divisible by tensor parallelism
                # world size. Otherwise, sequence parallelism will change the input shape making leading to mismatches.
                input_ids = input_ids[:, :-remainder]
                source_encoded = source_encoded[:, :-remainder]

        text_inputs = input_ids[:, :-1, -1]  # (B, T-1)
        text_labels = input_ids[:, 1:, -1]  # (B, T-1)
        audio_inputs = input_ids[:, :-1, :-1]  # (B, T-1, K)
        audio_labels = input_ids[:, 1:, :-1]  # (B, T-1, K)

        # Input embeds: (B, T-1, H)
        input_embeds = self.embed_tokens(text_inputs)
        for cbidx in range(self._num_codebooks):
            input_embeds.add_(self.embed_audio_tokens[cbidx](audio_inputs[..., cbidx]))
        input_embeds.add_(source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 0.3))

        return {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens,
            "output_lens": target_codes_lens,
            "text_labels": text_labels,
            "audio_labels": audio_labels,
        }

    def training_step(self, batch: dict, batch_idx: int):
        inputs = self.prepare_inputs(batch)

        forward_outputs = self(inputs)
        num_frames = inputs["input_lens"].sum()
        with loss_parallel():
            text_loss = (
                torch.nn.functional.cross_entropy(
                    forward_outputs["text_logits"].transpose(1, 2),
                    inputs["text_labels"],
                    reduction="sum",
                )
                / num_frames
            )
            audio_loss = torch.nn.functional.cross_entropy(
                forward_outputs["audio_logits"].transpose(1, 2),
                inputs["audio_labels"],
                reduction="sum",
            ) / (num_frames * self._num_codebooks)

        loss = text_loss + audio_loss

        B, T = inputs["input_embeds"].shape[:2]
        print(f"{loss=} {B=} {T=}")

        self.log_dict(
            {
                "loss": loss,
                "learning_rate": torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr']),
                "text_loss": text_loss,
                "audio_loss": audio_loss,
                "batch_size": B,
                "sequence_length": T,
                "num_frames": num_frames.to(torch.float32),  # avoid warning
                "padding_ratio": num_frames / (B * T),
            },
            on_step=True,
        )

        return loss

    def on_validation_epoch_start(self) -> None:
        # Cleaning up GPU memory before we load ASRModel, because it may already
        # be quite fragmented and close to the limit after observing many
        # dynamic shapes during the training epoch.
        torch.cuda.memory.empty_cache()
        self.asr = ASRModel.from_pretrained(self.cfg.scoring_asr).to(torch.bfloat16).eval()
        WithOptionalCudaGraphs.disable_cuda_graphs_recursive(self.asr, attribute_path="decoding.decoding")
        # Setup a separate BLEU metric for each validation dataloader through CombinedLoader.
        # See: https://lightning.ai/docs/pytorch/LTS/guides/data.html#accessing-dataloaders-within-lightningmodule
        self._partial_val_losses = []
        self.bleu = {}
        for name in self.trainer.val_dataloaders.keys():
            self.bleu[name] = SacreBLEUScore().to(self.device)

    def on_validation_epoch_end(self) -> None:
        for name, bleu in self.bleu.items():
            self.log(f"val_asr_bleu_{name}", bleu.compute(), on_epoch=True, sync_dist=True)
            bleu.reset()
        self.asr = None  # free up GPU memory
        val_loss = torch.mean(torch.stack(self._partial_val_losses))
        self._partial_val_losses = None
        self.log("val_loss", val_loss, on_epoch=True, sync_dist=True)
        torch.cuda.memory.empty_cache()

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted
            inputs = self.prepare_inputs(dataset_batch)
            forward_outputs = self(inputs)
            num_frames = inputs["input_lens"].sum()
            with loss_parallel():
                text_loss = (
                    torch.nn.functional.cross_entropy(
                        forward_outputs["text_logits"].transpose(1, 2),
                        inputs["text_labels"],
                        reduction="sum",
                    )
                    / num_frames
                )
                audio_loss = torch.nn.functional.cross_entropy(
                    forward_outputs["audio_logits"].transpose(1, 2),
                    inputs["audio_labels"],
                    reduction="sum",
                ) / (num_frames * self._num_codebooks)

            loss = text_loss + audio_loss
            self._partial_val_losses.append(loss)

            B = inputs["input_embeds"].shape[0]
            self.log(f"val_loss_{name}", loss, on_epoch=True, sync_dist=True, batch_size=B)
            self.log(f"val_text_loss_{name}", text_loss, on_epoch=True, sync_dist=True, batch_size=B)
            self.log(f"val_audio_loss_{name}", audio_loss, on_epoch=True, sync_dist=True, batch_size=B)

            # ASR BLEU
            import torchaudio

            with torch.inference_mode():
                predicted_audio_tokens = torch.argmax(forward_outputs["audio_logits"], dim=2).transpose(1, 2)
                with _safe_audio_codec_inference():
                    predicted_audio, predicted_audio_lens = self._audio_codec.decode(
                        tokens=predicted_audio_tokens, tokens_len=inputs["output_lens"]
                    )
                ans = self.asr.transcribe(
                    list(torchaudio.functional.resample(predicted_audio, 22050, 16000)),
                    batch_size=predicted_audio.shape[0],
                    verbose=False,
                )
            self.bleu[name].update([hyp.text for hyp in ans], [[tt] for tt in dataset_batch["target_texts"]])

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def _get_bos_embedding(self):
        """
        Return the BOS embedding: sum of text BOS token embedding and audio BOS token embedding.
        Intended to be added to the input speech representation in the initial frame during inference.
        The returned shape is (1, embedding_dim).
        """
        raise NotImplementedError()
        embed = self.embed_tokens(torch.full((1,), fill_value=self.tokenizer.bos_id, device=self.device))
        embed += self.audio_bos_embedding
        return embed

    def offline_inference(self, input_signal: torch.Tensor, input_signal_lens: torch.Tensor):
        raise NotImplementedError()
        # Run through ASR simulating streaming, and pre-multiply by input channel weight
        # input_embeds: (B, T, H)
        input_embeds, input_embed_lens = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_lens,
        )
        input_embeds *= self.cfg.get("duplex_user_channel_weight", 0.3)

        # Pre-allocate the memory for outputs.
        # from transformers import DynamicCache
        # cache = DynamicCache()
        B, T = input_embeds.shape[:2]
        gen_audio = torch.empty(B, T, self._num_codebooks, device=self.device, dtype=torch.long)
        gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)

        # Construct initial input frame using BOS token and BOS output audio frame
        # and run the first prediction step
        input_embeds[:, 0] += self._get_bos_embedding()
        ans = self(input_embeds[:, :1])
        # ans = self.llm(input_embeds[:, :1], past_key_values=cache, use_cache=True, return_dict=True)
        # text_logits = self.lm_head(ans.logits)
        # audio_logits = self.audio_head(ans.logits)
        # cache = ans.past_key_values
        gen_text[:, 0] = ans["text_logits"].argmax(dim=-1)
        gen_audio[:, 0] = ans["audio_logits"].argmax(dim=-2)

        for t in range(1, input_embeds.shape[1]):
            input_embeds[:, t] += self.embed_tokens(gen_text[:, t])
            for cbidx in range(self._num_codebooks):
                input_embeds[:, t] += self.embed_audio_tokens[cbidx](gen_audio[:, t, cbidx])
            ans = self(input_embeds[:, :t])
            gen_text[:, 0] = ans["text_logits"].argmax(dim=-1)
            gen_audio[:, 0] = ans["audio_logits"].argmax(dim=-2)

        return gen_text, gen_audio

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        parameters = chain(
            self.perception.parameters(),
            self.llm.parameters(),
            self.lm_head.parameters(),
            self.audio_head.parameters(),
        )
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, parameters, _convert_='all')
        lr_scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1},
        }

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {"name": "target_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "target_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
            ],
        }

    def configure_model(self) -> None:
        self._use_fsdp = False
        self._use_tp = False
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            # TODO: Distributing embeddings with TP in this setup is tricky
            #       because we're adding with the output of a non-parallelized
            #       speech encoder.
            # for m in (self.embed_tokens, self.embed_audio_tokens):
            #     parallelize_module(
            #         m,
            #         tp_mesh,
            #         ColwiseParallel(
            #             # input_layouts=Shard(1),
            #             # # Optional: Shard the output along the class dimension to compute the loss in parallel.
            #             # # See `loss_parallel` in `train.py`
            #             # output_layouts=Shard(1),
            #             # use_local_output=False,
            #         ),
            #     )

            # # Parallelize the first embedding and the last linear out projection
            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(self.llm, tp_mesh, plan)

            # Parallelize each transformer block
            for transformer_block in self.llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                # Apply the plan for the current transformer block
                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.lm_head, self.audio_head):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        # Optional: Shard the output along the class dimension to compute the loss in parallel.
                        # See `loss_parallel` in `train.py`
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1  # Hybrid-sharding not supported
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh, "mp_policy": MixedPrecisionPolicy(torch.bfloat16)}

            for idx, layer in enumerate(self.llm.layers):
                # layer.self_attn = checkpoint_wrapper(layer.self_attn)
                # layer.mlp = checkpoint_wrapper(layer.mlp)
                self.llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            for idx in range(self._num_codebooks):
                self.embed_audio_tokens[idx] = fully_shard(self.embed_audio_tokens[idx], **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            # self.lm_head = checkpoint_wrapper(self.lm_head)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            # self.audio_head = checkpoint_wrapper(self.audio_head)
            self.audio_head = fully_shard(self.audio_head, **fsdp_config)

            # for idx, layer in enumerate(self.perception.encoder.layers):
            #     self.perception.encoder.layers[idx] = fully_shard(layer, **fsdp_config)
            # self.perception = checkpoint_wrapper(self.perception)
            self.perception = fully_shard(self.perception, **fsdp_config)


class DuplexS2SModelSpeechDecoder(DuplexS2SModel):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.speech_decoder = SpeechDecoder(
            speech_decoder_parms=dict(self.cfg.speech_decoder_parms),
            lantent_dim=self.llm.config.hidden_size,
            num_audio_codebooks=self._num_codebooks,
            num_audio_tokens_per_codebook=self._codebook_size
        )

    def forward(
        self,
        input,
    ) -> dict[str, Tensor]:
        """
        Implements a fully offline forward pass through the entire model.
        The flow is the following:

                                                     |-> |audio_head| -> |audio codes|
        |source speech + prev target text| -> |llm| -|
                                                     |-> |lm_head|    -> |token ids  |
        """
        out = self.llm(inputs_embeds=input["input_embeds"])
        B, T = input["input_embeds"].shape[:2]
        speech_mask = input['loss_mask'][:, :, -1].reshape(input['loss_mask'].size(0), input['loss_mask'].size(1))
        # import pdb; pdb.set_trace()
        text_logits = self.lm_head(out['last_hidden_state'])
        _, audio_logits = self.speech_decoder(out['last_hidden_state'].transpose(0,1), speech_mask, input_audio_tokens=input['input_audio_tokens'])
        audio_logits = audio_logits.view(B, T, self._codebook_size, self._num_codebooks)


        return {
            "text_logits": text_logits,
            "audio_logits": audio_logits,
        }


    def prepare_inputs(self, batch: dict):

        """
        Performs additional processing on the mini-batch collected from dataloader.
        Notably:
        * Convert source audio to speech representations.
        * Convert target audio to target audio tokens.
        * Convert target text to embeddings.
        * Combine the input audio and target text embeddings.
        * Take care of any necessary slicing to align the shapes of source audio,
          target audio, and target token ids.


        """


        source_encoded, source_encoded_lens = self.perception(
            input_signal=batch["source_audio"], input_signal_length=batch["source_audio_lens"]
        )

        target_tokens = batch["target_tokens"]
        if target_tokens.shape[1] < source_encoded.shape[1]:
            pad_id = self.tokenizer.pad
            if pad_id is None:
                pad_id = self.tokenizer.unk_id
            if pad_id is None:
                pad_id = 0  # fallback
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (torch.ones(source_encoded.shape[0], 1, device=source_encoded.device) * pad_id).to(torch.long),
                ],
                dim=-1,
            )


        with torch.no_grad(), _safe_audio_codec_inference():
            target_codes, target_codes_lens = self._audio_codec.encode(
                audio=batch["target_audio"], audio_len=batch["target_audio_lens"]
            )
            # shape: [B, D, T] =>  [B, T, D]
            target_codes = target_codes.transpose(1, 2)


        if (tl := target_codes.shape[1]) != (sl := source_encoded.shape[1]):
            if tl < sl:
                diff = sl - tl
                source_encoded = source_encoded[:, :tl]
                target_tokens = target_tokens[:, :tl]
                torch.clamp_(source_encoded_lens, max=tl)
            else:
                diff = tl - sl
                target_codes = target_codes[:, :sl]
                torch.clamp_(target_codes_lens, max=sl)
            if diff > 2:
                logging.warning(
                    f"A mismatch between source ({sl}) and target ({tl}) sequence length greater than 2 detected. "
                    f"This may indicate significant desynchronization in longer sessions."
                )


        input_ids = torch.cat([target_codes, target_tokens[..., None]], dim=-1)

        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_ids.shape[1] - 1) % tp_world_size) != 0:

                input_ids = input_ids[:, :-remainder]
                source_encoded = source_encoded[:, :-remainder]



        text_inputs = input_ids[:, :-1, -1]
        input_embeds = self.embed_tokens(text_inputs)


        # embedding = source_encoded * duplex_user_channel_weight

        input_embeds = input_embeds + source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 0.3)


        text_labels = input_ids[:, 1:, -1]  # [B, T-1]
        audio_labels = input_ids[:, 1:, :-1]  # [B, T-1, audio_dim]

        labels = torch.cat([text_labels.unsqueeze(-1), audio_labels], dim=-1)  # [B, T-1, 1 + audio_dim]


        loss_mask = torch.ones_like(labels)


        input_audio_tokens = input_ids[:, :-1, :-1]  # [B, T-1, audio_dim]


        return {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens,
            "output_lens": target_codes_lens,
            "text_labels": text_labels,
            "audio_labels": audio_labels,
            "loss_mask": loss_mask,
            "input_audio_tokens": input_audio_tokens,
         }



    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {"name": "target_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "target_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
            ],
        }

    def configure_model(self) -> None:
        self._use_fsdp = False
        self._use_tp = False
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            # TODO: Distributing embeddings with TP in this setup is tricky
            #       because we're adding with the output of a non-parallelized
            #       speech encoder.
            # for m in (self.embed_tokens, self.embed_audio_tokens):
            #     parallelize_module(
            #         m,
            #         tp_mesh,
            #         ColwiseParallel(
            #             # input_layouts=Shard(1),
            #             # # Optional: Shard the output along the class dimension to compute the loss in parallel.
            #             # # See `loss_parallel` in `train.py`
            #             # output_layouts=Shard(1),
            #             # use_local_output=False,
            #         ),
            #     )

            # # Parallelize the first embedding and the last linear out projection
            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(self.llm, tp_mesh, plan)

            # Parallelize each transformer block
            for transformer_block in self.llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                # Apply the plan for the current transformer block
                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.lm_head, self.audio_head):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        # Optional: Shard the output along the class dimension to compute the loss in parallel.
                        # See `loss_parallel` in `train.py`
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1  # Hybrid-sharding not supported
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh, "mp_policy": MixedPrecisionPolicy(torch.bfloat16)}

            for idx, layer in enumerate(self.llm.layers):
                # layer.self_attn = checkpoint_wrapper(layer.self_attn)
                # layer.mlp = checkpoint_wrapper(layer.mlp)
                self.llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            # self.embed_audio_tokens = fully_shard(self.embed_audio_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            # self.lm_head = checkpoint_wrapper(self.lm_head)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            # self.audio_head = checkpoint_wrapper(self.audio_head)
            self.audio_head = fully_shard(self.audio_head, **fsdp_config)

            # for idx, layer in enumerate(self.perception.encoder.layers):
            #     self.perception.encoder.layers[idx] = fully_shard(layer, **fsdp_config)
            # self.perception = checkpoint_wrapper(self.perception)
            self.perception = fully_shard(self.perception, **fsdp_config)


def _load_pretrained(cls, model_path_or_name: str):
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        return cls.restore_from(model_path_or_name)
    else:
        return cls.from_pretrained(model_path_or_name)


@contextmanager
def _safe_audio_codec_inference():
    """
    Works around an issue where PTL setting of precision='bf16-true'
    interferes with padding shape computations inside of audio codec convolutional layers.
    This is because bf16-true temporarily changes the default float dtype to bf16,
    which cannot represent integers used in shape computations, and truncates them.
    """
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
        try:
            yield
        finally:
            torch.set_default_dtype(default_dtype)
