# s2s-dev

This repo contains the training recipes that use the new (as of Mar 19th 2025) scalable S2S codebase. **The code is still in development and likely to change.**

The recipes and scripts here are confirmed to work on `draco-oci-iad`.

## Structure

### Running experiments

Use the scripts below to run experiments. **Remember to override the paths and W&B keys with your own in .sub and .yaml files!**

* `auto_launcher_with_seed.sh` - use it to submit SLURM jobs on the cluster, e.g. `./auto_launcher_with_seed.sh -n8 s2s_tinyllama_repro.sub` submits 8 consecutive jobs. This script generates a random seed for each submitted SLURM job, leveraging `shard_seed="randomized"` option in lhotse to ensure each data parallel rank is seeded differently, but each tensor parallel rank is seeded identically.
* `s2s_tinyllama_repro.sub` - SLURM submission script for a single job, contains a minimal amount of boilerplate code. This particular script attempts to reproduce Zhehuai's TinyLlama-1B S2S recipe.
* `s2s_tinyllama_repro.yaml` - The experiment configuration. Prefer copying and modifying this file rather than overriding options in the SLURM script to have the entire configuration available at a glance and readily versioned.
* `train_icfg_12mar2025.yaml` - YAML configuration specifying all training data sources and their weights. This config is a 99% subset of Zhehuai's TinyLlama-1B S2S recipe training data.

### Debugging

* `get_interactive_node.sh` - use it to fetch a node for interactive jobs and debugging.
* `run.sh` - uses `torchrun` to launch a job with a specified configuration, useful for debugging, quick checks, and profiling (nsys command available inside).

## S2S codebase

The new S2S codebase is available in the following branch: https://github.com/NVIDIA/NeMo/tree/duplex-s2s-new (PR: https://github.com/NVIDIA/NeMo/pull/12617).
Below is an outline of how it's structured and what's the rationale.

They key idea of this codebase is that it should be fairly small and allow to swap out different LLMs easily, as well as scale up their size for training.
It is also expected to be efficient, hence some profiling scripts and code are included. The swapping out of LLMs is confirmed to work (succesfully done with TinyLlama, Llama3, Gemma1, Qwen2.5).

### Training script and config

**examples/duplex_s2s/s2s_duplex_train.py** 
This is the main training script. It's very brief and similar to ASR collection training scripts. The main difference is that it instantiates a Lightning DataModule separately from the model. It currently uses `exp_manager` for checkpointing, resumption, and W&B logger setup, but that might change in the future.

**examples/duplex_s2s/conf/s2s_duplex.yaml**
This is an example configuration I use to develop this code on my workstation. You might need to replace some paths to make that work on yours.

### Collection: duplex_s2s 

**nemo/collections/duplex_s2s** 
This is the root directory for this small development collection. I will refer to paths within this root below.

**data/datamodule.py**
Contains definition of a Lightning DataModule adequate for S2S (but probably can be re-used more broadly if made to accept a `Dataset` class).
It takes care of setting up the proper DP ranks for dataloaders, and instantiating them.
Keep in mind the actual dataset paths and blend are defined by the YAML config, not Python code.

**data/dataset.py**
Defines S2S dataset class that converts Lhotse Cuts to a dict of tensors. It mostly follows the logic from the original S2S codebase but is simplified and shorter. 
It returns collated mini-batches of source/target audio (wav) and source/target tokens (tokenized text). 
The data format is the following: 
* each cut contains `.recording` and `.target_audio` attributes used for loading source and target audio correspondingly
* each cut contains a list of `supervisions` with objects of type `lhotse.SupervisionSegment` that represent turns with corresponding text and speaker information.
* the text of each supervision is tokenized and identified as model's output (`supervision.speaker == 'agent'`) or model's input.
* `target_tokens` and `source_tokens` are created to have length equal to `lhotse.utils.compute_num_frames(cut.duration, frame_length, cut.sampling_rate)` - the `frame_length` is typically 80ms in our setups.
* we use `lhotse.utils.compute_num_frames(supervision.start, frame_length)` to determine the token offset for assigning the turn-specific tokens
* if the token sequence is too long vs the audio, we might emit warnings in some cases we are able to detect, but generally these tokens will be truncated and are likely an invalid training example. This should be fixed sometime.

**modules/perception.py**
Verbatim copy from `multimodal/speech_llm` collection. This is a wrapper on ASR encoder that adds a few layers of "modality adapter" and performs SpecAugment (if configured). 

**models/duplex_s2s_model.py**
This is the heart of this new codebase - the model definition itself. Key insights:
* The constructor `__init__` initializes pretrained ASR encoder/perception; LLM (using HuggingFace AutoModel); and audio codec. It also adds separate text token and audio token prediction heads.
* `forward` accepts input representations (i.e. the sum of audio perception and text embedding outputs) and runs an offline forward pass through the LM and boths token prediction heads, returns logits for each.
* `training_step` builds the input representations with `prepare_inputs`, runs `forward`, computes the losses, and logs some information.
* `prepare_inputs` runs source audio through `perception` (with grads on; learnable); target audio through audio codec (grads off; non-trainable); truncates source/target audio and target text sequences if needed (mismatch in token sequence length up to 2 is ignored, otherwise we emit warnings); does extra truncation if tensor parallelism is enabled to avoid shape mismatches with sequence parallelism; and returns a dict with input and label tensors suitable for `forward`/`training_step`.
* Validation first clears GPU memory (to avoid OOM), loads a scoring ASR model into GPU, and initializes metric aggregation. After validation is finished, the scoring ASR model is deleted, final metrics are logged, and cleared too, together with GPU memory.
* Validation is configured to use multiple dataloaders in parallel (see [Lightning's CombinedLoader with max_size mode](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.utilities.combined_loader.html#lightning.pytorch.utilities.combined_loader.CombinedLoader). That means the batch in `validation_step` is not a single batch: it is a dict where the key is the dataset name, and the value is its mini-batch. This is why we loop over `batch.items()`. Even for a single validation dataset, we should provide its name. This is configured via `data.validation_ds.datasets` field in the YAML config. This solution enables us to track the metrics across each dataset elegantly, as well as specifying individual override options in the config (each validation/test dataloader inherits options from `validation_ds` namespace, minus `datasets` key, and then overrides them with the options in `validation_ds.datasets.<current-dataset-name>` namespace).
* Currently supported metrics are validation loss (per dataset + aggregated) and ASR BLEU.
* Preliminary support for OOMptimizer is added, but not tested thoroughly yet, so I've been using only `batch_duration` so far.

The final method I'm about to describe is the most important for scaling model sizes: `configure_model`.
In the YAML configuration we can select a `ModelParallelStrategy` under `trainer.strategy`, which makes Lightning configure an object called `trainer.device_mesh`.
The mesh has a "shape" where each dimension is either data-parallel or model-parallel dimension, allowing us to combine FSDP2 with TP/SP.
We control the shape by specifying the world size of `data_parallel` and `tensor_parallel` in the YAML config, under `trainer.strategy`.
A shape of 1 means a given dimension is disabled (allowing us to do pure FSDP2; pure TP; or a mix for 2D parallelism).

Inside `configure_model`, our task is to tell PyTorch how do we want to distribute different layers of the model across GPUs.

With FSDP2, we generally call `fully_shard` on modules which tells PyTorch to distribute (shard) all parameters in that module across `data_parallel` world size,
all-gather them just before the computation starts, and de-allocate after we leave that layer. The more granularly we call this on the model's modules, the more memory we save, but the larger is the communication overhead. Current state seems to be an OK balance that lets us train ~30B models with just FSDP2.

With TP/SP, we similarly call `parallelize_module` on certain modules together with a "parallelization plan". This is a bit more involved and requires you to understand how TP/SP works. I won't go into details here, but I'll link a doc with useful resources below. The current implementation seems to be working but hasn't been tested thoroughly yet - unlike FSDP2.

Note that since different HF LLMs have minor variations in architecture, we might need to keep a registry of `configure_model` functions that parallelize/shard the models appropriately, and use the appropriate function for a given model.

I've collected useful resources about this topic here: https://docs.google.com/document/d/1sHWZrXs1hjpsXlJbmQx7HR9pbps8uadY9Bn38CitICE

### Profiling

To profile, uncomment the nsys command in `run.sh`, and uncomment the line adding `PROFILING()` callback in the `Trainer` in the training script. 
The resulting profile can be opened in NVIDIA Nsight systems GUI for analysis.
