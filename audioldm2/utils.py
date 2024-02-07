import contextlib
import importlib
from huggingface_hub import hf_hub_download
import numpy as np
from inspect import isfunction
import os
import soundfile as sf
import time
import wave
import torch
import audioldm2.utilities.audio as Audio
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from safetensors import deserialize, safe_open, serialize, serialize_file

import progressbar

def random_uniform(start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5  # Manually limit the maximum amplitude into 0.5

def read_list(fname):
    result = []
    with open(fname, "r", encoding="utf-8") as f:
        for each in f.readlines():
            each = each.strip('\n')
            result.append(each)
    return result

def get_duration(fname):
    with contextlib.closing(wave.open(fname, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)
    
def pad_spec(log_mel_spec, target_length):
    n_frames = log_mel_spec.shape[0]
    p = target_length - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        log_mel_spec = m(log_mel_spec)
    elif p < 0:
        log_mel_spec = log_mel_spec[0 : target_length, :]

    if log_mel_spec.size(-1) % 2 != 0:
        log_mel_spec = log_mel_spec[..., :-1]

    return log_mel_spec

def wav_feature_extraction(waveform, target_length, STFT):
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    log_mel_spec, stft, energy = Audio.tools.get_mel_from_wav(waveform, STFT)

    log_mel_spec = torch.FloatTensor(log_mel_spec.T)
    stft = torch.FloatTensor(stft.T)

    log_mel_spec, stft = pad_spec(log_mel_spec, target_length), pad_spec(stft, target_length)
    return log_mel_spec, stft

def build_dsp(config):
    STFT = Audio.stft.TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )
    return STFT

def get_bit_depth(fname):
    with contextlib.closing(wave.open(fname, "r")) as f:
        bit_depth = f.getsampwidth() * 8
        return bit_depth


def get_time():
    t = time.localtime()
    return time.strftime("%d_%m_%Y_%H_%M_%S", t)


def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_wave(waveform, savepath, name="outwav"):
    if type(name) is not list:
        name = [name] * waveform.shape[0]

    for i in range(waveform.shape[0]):
        if(waveform.shape[0] > 1):
            fname = "%s.wav" % (
                    os.path.basename(name[i])
                    if (not ".mp3" in name[i] and not ".wav" in name[i])
                    else os.path.basename(name[i]).split(".")[0],
                )
        else:
            fname = "%s.wav" % os.path.basename(name[i]) if (not ".mp3" in name[i] and not ".wav" in name[i]) else os.path.basename(name[i]).split(".")[0]
            
        path = os.path.join(
            savepath, fname
        )
        print("Save audio to %s" % path)
        sf.write(path, waveform[i, 0], samplerate=16000)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    try:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))
    except:
        import ipdb
        ipdb.set_trace()

# Ashley: New add
checkpoint_dict_replacements = {
    'cond_stage_model.transformer.text_model.embeddings.': 'cond_stage_model.transformer.embeddings.',
    'cond_stage_model.transformer.text_model.encoder.': 'cond_stage_model.transformer.encoder.',
    'cond_stage_model.transformer.text_model.final_layer_norm.': 'cond_stage_model.transformer.final_layer_norm.',
}

def load_file(filename: Union[str, os.PathLike], device="cpu") -> Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into torch format.

    Args:
        filename (`str`, or `os.PathLike`):
            The name of the file which contains the tensors
        device (`Dict[str, any]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations

    Returns:
        `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor`

    Example:

    ```python
    from safetensors.torch import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    result = {}
    with safe_open(filename, framework="pt", device=device) as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result

def transform_checkpoint_dict_key(k):
    for text, replacement in checkpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k

def read_state_dict(checkpoint_file, print_global_state=False):
    _, extension = os.path.splitext(checkpoint_file)
    if extension.lower() == ".safetensors":
        pl_sd = load_file(checkpoint_file, device='cpu')
    else:
        pl_sd = torch.load(checkpoint_file, map_location='cpu')

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd

def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd

##end

def default_audioldm_config(model_name="audioldm2-full"):
    basic_config = get_basic_config()
    if("-large-" in model_name):
        basic_config["model"]["params"]["unet_config"]["params"]["context_dim"] = [768, 1024, None]
        basic_config["model"]["params"]["unet_config"]["params"]["transformer_depth"] = 2
    if("-speech-" in model_name):
        basic_config["model"]["params"]["unet_config"]["params"]["context_dim"] = [768]
        basic_config["model"]["params"]["cond_stage_config"] = {
        "crossattn_audiomae_generated": {
          "cond_stage_key": "all",
          "conditioning_key": "crossattn",
          "target": "audioldm2.latent_diffusion.modules.encoders.modules.SequenceGenAudioMAECond",
          "params": {
            "always_output_audiomae_gt": False,
            "learnable": True,
            "use_gt_mae_output": True,
            "use_gt_mae_prob": 1,
            "base_learning_rate": 0.0002,
            "sequence_gen_length": 512,
            "use_warmup": True,
            "sequence_input_key": [
              "film_clap_cond1",
              "crossattn_vits_phoneme"
            ],
            "sequence_input_embed_dim": [
              512,
              192
            ],
            "batchsize": 16,
            "cond_stage_config": {
              "film_clap_cond1": {
                "cond_stage_key": "text",
                "conditioning_key": "film",
                "target": "audioldm2.latent_diffusion.modules.encoders.modules.CLAPAudioEmbeddingClassifierFreev2",
                "params": {
                  "sampling_rate": 48000,
                  "embed_mode": "text",
                  "amodel": "HTSAT-base"
                }
              },
              "crossattn_vits_phoneme": {
                "cond_stage_key": "phoneme_idx",
                "conditioning_key": "crossattn",
                "target": "audioldm2.latent_diffusion.modules.encoders.modules.PhonemeEncoder",
                "params": {
                  "vocabs_size": 183,
                  "pad_token_id": 0,
                  "pad_length": 310
                }
              },
              "crossattn_audiomae_pooled": {
                "cond_stage_key": "ta_kaldi_fbank",
                "conditioning_key": "crossattn",
                "target": "audioldm2.latent_diffusion.modules.encoders.modules.AudioMAEConditionCTPoolRand",
                "params": {
                  "regularization": False,
                  "no_audiomae_mask": True,
                  "time_pooling_factors": [
                    1
                  ],
                  "freq_pooling_factors": [
                    1
                  ],
                  "eval_time_pooling": 1,
                  "eval_freq_pooling": 1,
                  "mask_ratio": 0
                }
              }
            }
          }
        }
    }
    if("48k" in model_name):
        basic_config=get_audioldm_48k_config()
    if("t5" in model_name):
        basic_config=get_audioldm_crossattn_t5_config()
    return basic_config

class MyProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

def download_checkpoint(checkpoint_name="audioldm2-full"):
    if("audioldm2-speech" in checkpoint_name):
        model_id = "haoheliu/audioldm2-speech"
    else:
        model_id = "haoheliu/%s" % checkpoint_name
    ckpt_path = '/home/ashley/.cache/huggingface/hub/models--haoheliu--audioldm2-music-665k/snapshots/dc4c0191c097c725f77080d426649b9b3941a3d2/audioldm2-music-665k.pth'
    if os.path.exists(ckpt_path):
        checkpoint_path = ckpt_path
    else:
        checkpoint_path = hf_hub_download(
            repo_id=model_id,
            filename=checkpoint_name+".pth"
        )
    return checkpoint_path


def get_basic_config():
    return {
        "metadata_root": "data/metadata/dataroot.json",
        "log_directory": "./log/audiomae_pred",
        "precision": "high",
        "data": {
            "train": [
                "musiccaps",
                "FMA-intro",
                "FMA-chorus",
                "FMA-outro",
                # "audiocaps",
                # "audioset",
                # "wavcaps",
                # "audiostock_music_250k",
                # "free_to_use_sounds",
                # "epidemic_sound_effects",
                # "vggsound",
                # "million_song_dataset",
            ],
            "val": "musiccaps",
            "test": "musiccaps",
            # "val": "audiocaps",
            # "test": "audiocaps",
            "extra_cond": [
                "chord", "melody"
            ],
            "class_label_indices": "audioset",
            "dataloader_add_ons": [
                # "extract_kaldi_fbank_feature",
                # "extract_vits_phoneme_and_flant5_text",
                "waveform_rs_48k",
            ],
        },
        "variables": {
            "sampling_rate": 16000,
            "mel_bins": 64,
            "latent_embed_dim": 8,
            "latent_t_size": 256,
            "latent_f_size": 16,
            "in_channels": 8,
            "optimize_ddpm_parameter": True,
            "warmup_steps": 5000,
        },
        "step": {
            "validation_every_n_epochs": 1,
            "save_checkpoint_every_n_steps": 5000,
            "limit_val_batches": 10,
            "max_steps": 1500000,
            "save_top_k": 2,
        },
        "preprocessing": {
            "audio": {
                "sampling_rate": 16000,
                "max_wav_value": 32768,
                "duration": 10.24,
            },
            "stft": {"filter_length": 1024, "hop_length": 160, "win_length": 1024},
            "mel": {"n_mel_channels": 64, "mel_fmin": 0, "mel_fmax": 8000},
        },
        "augmentation": {"mixup": 0},
        "model": { # 
            "target": "audioldm2.latent_diffusion.models.ddpm.LatentDiffusion",
            "params": {
                "first_stage_config": {
                    "base_learning_rate": 0.000008,
                    "target": "audioldm2.latent_encoder.autoencoder.AutoencoderKL",
                    "params": {
                        "sampling_rate": 16000,
                        "batchsize": 4,
                        "monitor": "val/rec_loss",
                        "image_key": "fbank",
                        "subband": 1,
                        "embed_dim": 8,
                        "time_shuffle": 1,
                        "lossconfig": {
                            "target": "audioldm2.latent_diffusion.modules.losses.LPIPSWithDiscriminator",
                            "params": {
                                "disc_start": 50001,
                                "kl_weight": 1000,
                                "disc_weight": 0.5,
                                "disc_in_channels": 1,
                            },
                        },
                        "ddconfig": {
                            "double_z": True,
                            "mel_bins": 64,
                            "z_channels": 8,
                            "resolution": 256,
                            "downsample_time": False,
                            "in_channels": 1,  # same as batchsize?
                            "out_ch": 1,
                            "ch": 128,
                            "ch_mult": [1, 2, 4],
                            "num_res_blocks": 2,
                            "attn_resolutions": [],
                            "dropout": 0,
                        },
                    },
                },
                "base_learning_rate": 0.00001,
                "warmup_steps": 5000,
                "optimize_ddpm_parameter": True,
                "sampling_rate": 16000,
                "batchsize": 16,
                "linear_start": 0.0015,
                "linear_end": 0.0195,
                "num_timesteps_cond": 1,
                "log_every_t": 200,
                "timesteps": 1000,
                "unconditional_prob_cfg": 0.1,
                "parameterization": "eps",
                "first_stage_key": "fbank",
                "latent_t_size": 256,
                "latent_f_size": 16,
                "channels": 8,
                "monitor": "val/loss_simple_ema",
                "scale_by_std": True,
                "unet_config": {
                    "target": "audioldm2.latent_diffusion.modules.diffusionmodules.openaimodel.UNetModel",
                    "params": {
                        "image_size": 64,
                        "context_dim": [768, 1024],
                        "in_channels": 8,
                        "out_channels": 8,
                        "model_channels": 128,
                        "attention_resolutions": [8, 4, 2],
                        "num_res_blocks": 2,
                        "channel_mult": [1, 2, 3, 5],
                        "num_head_channels": 32,
                        "use_spatial_transformer": True,
                        "transformer_depth": 1,
                        # Ashley: New Add
                        "use_checkpoint": True
                    },
                },
                "evaluation_params": {
                    "unconditional_guidance_scale": 3.5,
                    "ddim_sampling_steps": 200,
                    "n_candidates_per_samples": 3,
                },
                "cond_stage_config": {
                    "crossattn_audiomae_generated": {
                        "cond_stage_key": "all",
                        "conditioning_key": "crossattn",
                        "target": "audioldm2.latent_diffusion.modules.encoders.modules.SequenceGenAudioMAECond",
                        "params": {
                            "always_output_audiomae_gt": False,
                            "learnable": True,
                            "device": "cuda",
                            "use_gt_mae_output": True,
                            "use_gt_mae_prob": 0.25,
                            "base_learning_rate": 0.0002,
                            "sequence_gen_length": 8,
                            "use_warmup": True,
                            "sequence_input_key": [
                                "film_clap_cond1",
                                "crossattn_flan_t5",
                            ],
                            "sequence_input_embed_dim": [512, 1024],
                            "batchsize": 16,
                            "cond_stage_config": {
                                "film_clap_cond1": {
                                    "cond_stage_key": "text",
                                    "conditioning_key": "film",
                                    "target": "audioldm2.latent_diffusion.modules.encoders.modules.CLAPAudioEmbeddingClassifierFreev2",
                                    "params": {
                                        "sampling_rate": 48000,
                                        "embed_mode": "text",
                                        "amodel": "HTSAT-base",
                                    },
                                },
                                "crossattn_flan_t5": {
                                    "cond_stage_key": "text",
                                    "conditioning_key": "crossattn",
                                    "target": "audioldm2.latent_diffusion.modules.encoders.modules.FlanT5HiddenState",
                                },
                                "crossattn_audiomae_pooled": {
                                    "cond_stage_key": "ta_kaldi_fbank",
                                    "conditioning_key": "crossattn",
                                    "target": "audioldm2.latent_diffusion.modules.encoders.modules.AudioMAEConditionCTPoolRand",
                                    "params": {
                                        "regularization": False,
                                        "no_audiomae_mask": True,
                                        "time_pooling_factors": [8],
                                        "freq_pooling_factors": [8],
                                        "eval_time_pooling": 8,
                                        "eval_freq_pooling": 8,
                                        "mask_ratio": 0,
                                    },
                                },
                            },
                        },
                    },
                    "crossattn_flan_t5": {
                        "cond_stage_key": "text",
                        "conditioning_key": "crossattn",
                        "target": "audioldm2.latent_diffusion.modules.encoders.modules.FlanT5HiddenState",
                    },
                },
            },
        },
    }

def get_audioldm_48k_config():
    return {
            "variables": {
                "sampling_rate": 48000,
                "latent_embed_dim": 16,
                "mel_bins": 256,
                "latent_t_size": 128,
                "latent_f_size": 32,
                "in_channels": 16,
                "optimize_ddpm_parameter": True,
                "warmup_steps": 5000
            },
            "step": {
                "validation_every_n_epochs": 1,
                "save_checkpoint_every_n_steps": 5000,
                "limit_val_batches": 10,
                "max_steps": 1500000,
                "save_top_k": 2
            },
            "preprocessing": {
                "audio": {
                "sampling_rate": 48000,
                "max_wav_value": 32768,
                "duration": 10.24
                },
                "stft": {
                "filter_length": 2048,
                "hop_length": 480,
                "win_length": 2048
                },
                "mel": {
                "n_mel_channels": 256,
                "mel_fmin": 20,
                "mel_fmax": 24000
                }
            },
            "augmentation": {
                "mixup": 0
            },
            "model": {
                "target": "audioldm2.latent_diffusion.models.ddpm.LatentDiffusion",
                "params": {
                "first_stage_config": {
                    "base_learning_rate": 0.000008,
                    "target": "audioldm2.latent_encoder.autoencoder.AutoencoderKL",
                    "params": {
                    "sampling_rate": 48000,
                    "batchsize": 4,
                    "monitor": "val/rec_loss",
                    "image_key": "fbank",
                    "subband": 1,
                    "embed_dim": 16,
                    "time_shuffle": 1,
                    "lossconfig": {
                        "target": "audioldm2.latent_diffusion.modules.losses.LPIPSWithDiscriminator",
                        "params": {
                        "disc_start": 50001,
                        "kl_weight": 1000,
                        "disc_weight": 0.5,
                        "disc_in_channels": 1
                        }
                    },
                    "ddconfig": {
                        "double_z": True,
                        "mel_bins": 256,
                        "z_channels": 16,
                        "resolution": 256,
                        "downsample_time": False,
                        "in_channels": 1,
                        "out_ch": 1,
                        "ch": 128,
                        "ch_mult": [
                        1,
                        2,
                        4,
                        8
                        ],
                        "num_res_blocks": 2,
                        "attn_resolutions": [],
                        "dropout": 0
                    }
                    }
                },
                "base_learning_rate": 0.0001,
                "warmup_steps": 5000,
                "optimize_ddpm_parameter": True,
                "sampling_rate": 48000,
                "batchsize": 16,
                "linear_start": 0.0015,
                "linear_end": 0.0195,
                "num_timesteps_cond": 1,
                "log_every_t": 200,
                "timesteps": 1000,
                "unconditional_prob_cfg": 0.1,
                "parameterization": "eps",
                "first_stage_key": "fbank",
                "latent_t_size": 128,
                "latent_f_size": 32,
                "channels": 16,
                "monitor": "val/loss_simple_ema",
                "scale_by_std": True,
                "unet_config": {
                    "target": "audioldm2.latent_diffusion.modules.diffusionmodules.openaimodel.UNetModel",
                    "params": {
                    "image_size": 64,
                    "extra_film_condition_dim": 512,
                    "context_dim": [
                        None
                    ],
                    "in_channels": 16,
                    "out_channels": 16,
                    "model_channels": 128,
                    "attention_resolutions": [
                        8,
                        4,
                        2
                    ],
                    "num_res_blocks": 2,
                    "channel_mult": [
                        1,
                        2,
                        3,
                        5
                    ],
                    "num_head_channels": 32,
                    "use_spatial_transformer": True,
                    "transformer_depth": 1
                    }
                },
                "evaluation_params": {
                    "unconditional_guidance_scale": 3.5,
                    "ddim_sampling_steps": 200,
                    "n_candidates_per_samples": 3
                },
                "cond_stage_config": {
                    "film_clap_cond1": {
                    "cond_stage_key": "text",
                    "conditioning_key": "film",
                    "target": "audioldm2.latent_diffusion.modules.encoders.modules.CLAPAudioEmbeddingClassifierFreev2",
                    "params": {
                        "sampling_rate": 48000,
                        "embed_mode": "text",
                        "amodel": "HTSAT-base"
                    }
                    }
                }
                }
            }
            }

def get_audioldm_crossattn_t5_config():
    return {
        "variables": {
            "sampling_rate": 16000,
            "mel_bins": 64,
            "latent_embed_dim": 8,
            "latent_t_size": 256,
            "latent_f_size": 16,
            "in_channels": 8,
            "optimize_ddpm_parameter": True,
            "warmup_steps": 5000
        },
        "step": {
            "validation_every_n_epochs": 1,
            "save_checkpoint_every_n_steps": 5000,
            "max_steps": 1500000,
            "save_top_k": 2
        },
        "preprocessing": {
            "audio": {
            "sampling_rate": 16000,
            "max_wav_value": 32768,
            "duration": 10.24
            },
            "stft": {
            "filter_length": 1024,
            "hop_length": 160,
            "win_length": 1024
            },
            "mel": {
            "n_mel_channels": 64,
            "mel_fmin": 0,
            "mel_fmax": 8000
            }
        },
        "augmentation": {
            "mixup": 0
        },
        "model": {
            "target": "audioldm2.latent_diffusion.models.ddpm.LatentDiffusion",
            "params": {
            "first_stage_config": {
                "base_learning_rate": 0.000008,
                "target": "audioldm2.latent_encoder.autoencoder.AutoencoderKL",
                "params": {
                "sampling_rate": 16000,
                "batchsize": 4,
                "monitor": "val/rec_loss",
                "image_key": "fbank",
                "subband": 1,
                "embed_dim": 8,
                "time_shuffle": 1,
                "lossconfig": {
                    "target": "audioldm2.latent_diffusion.modules.losses.LPIPSWithDiscriminator",
                    "params": {
                    "disc_start": 50001,
                    "kl_weight": 1000,
                    "disc_weight": 0.5,
                    "disc_in_channels": 1
                    }
                },
                "ddconfig": {
                    "double_z": True,
                    "mel_bins": 64,
                    "z_channels": 8,
                    "resolution": 256,
                    "downsample_time": False,
                    "in_channels": 1,
                    "out_ch": 1,
                    "ch": 128,
                    "ch_mult": [
                    1,
                    2,
                    4
                    ],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0
                }
                }
            },
            "base_learning_rate": 0.0001,
            "warmup_steps": 5000,
            "optimize_ddpm_parameter": True,
            "sampling_rate": 16000,
            "batchsize": 16,
            "linear_start": 0.0015,
            "linear_end": 0.0195,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "unconditional_prob_cfg": 0.1,
            "parameterization": "eps",
            "first_stage_key": "fbank",
            "latent_t_size": 256,
            "latent_f_size": 16,
            "channels": 8,
            "monitor": "val/loss_simple_ema",
            "scale_by_std": True,
            "unet_config": {
                "target": "audioldm2.latent_diffusion.modules.diffusionmodules.openaimodel.UNetModel",
                "params": {
                "image_size": 64,
                "context_dim": [
                    1024
                ],
                "in_channels": 8,
                "out_channels": 8,
                "model_channels": 128,
                "attention_resolutions": [
                    8,
                    4,
                    2
                ],
                "num_res_blocks": 2,
                "channel_mult": [
                    1,
                    2,
                    3,
                    5
                ],
                "num_head_channels": 32,
                "use_spatial_transformer": True,
                "transformer_depth": 1
                }
            },
            "evaluation_params": {
                "unconditional_guidance_scale": 3.5,
                "ddim_sampling_steps": 200,
                "n_candidates_per_samples": 3
            },
            "cond_stage_config": {
                "crossattn_flan_t5": {
                "cond_stage_key": "text",
                "conditioning_key": "crossattn",
                "target": "audioldm2.latent_diffusion.modules.encoders.modules.FlanT5HiddenState"
                }
            }
            }
        }
        }