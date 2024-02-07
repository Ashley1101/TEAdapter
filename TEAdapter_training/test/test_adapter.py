import os
import cv2
import json
import torch
import argparse
import soundfile as sf
from enum import Enum, unique
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast
from audioldm2.latent_diffusion.modules.encoders.adapter import Adapter

# from audioldm2.latent_diffusion.inference_base import (diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models)
# from audioldm2.latent_diffusion.modules.extra_condition import api
# from audioldm2.latent_diffusion.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)
from data.metadata.dataset_chord import ChordDataset
from data.metadata.dataset_melody import MelodyDataset
from data.metadata.dataset_fusion import AudioDataset
from audioldm2.utils import get_basic_config, save_wave, read_state_dict
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only
from audioldm2 import build_model, seed_everything, duration_to_latent_t_size, make_batch_for_text_to_audio

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

SAMPLE_RATE = 16000
LATENT_T_PER_SECOND = 25.6
DEFAULT_NEGATIVE_PROMPT = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                          'fewer digits, cropped, worst quality, low quality'

# environment variants
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '1101'
os.environ["TOKENIZERS_PARALLELISM"] = "true"

@unique
class ExtraCondition(Enum):
    chord = 0
    melody = 1

def get_adapters(opt, cond_type: ExtraCondition):
    adapter = {}
    cond_weight = getattr(opt, f'{cond_type.name}_weight', None)
    if cond_weight is None:
        cond_weight = getattr(opt, 'cond_weight')
    adapter['cond_weight'] = cond_weight

    adapter['model'] = Adapter(cin=1 * 16, channels=[128, 256, 384, 640][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to(opt.device)
    ckpt_path = getattr(opt, f'{cond_type.name}_adapter_ckpt', None)
    if ckpt_path is None:
        ckpt_path = getattr(opt, 'adapter_ckpt')
    state_dict = read_state_dict(ckpt_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('adapter.'):
            new_state_dict[k[len('adapter.'):]] = v
        else:
            new_state_dict[k] = v

    adapter['model'].load_state_dict(new_state_dict)

    return adapter

def get_adapter_feature(inputs, adapters):
    ret_feat_map = None
    ret_feat_seq = None
    if not isinstance(inputs, list):
        inputs = [inputs]
        adapters = [adapters]

    for input, adapter in zip(inputs, adapters):
        cur_feature = adapter['model'](input)
        if isinstance(cur_feature, list):
            if ret_feat_map is None:
                ret_feat_map = list(map(lambda x: x * adapter['cond_weight'], cur_feature))
            else:
                ret_feat_map = list(map(lambda x, y: x + y * adapter['cond_weight'], ret_feat_map, cur_feature))
        else:
            if ret_feat_seq is None:
                ret_feat_seq = cur_feature * adapter['cond_weight']
            else:
                ret_feat_seq = torch.cat([ret_feat_seq, cur_feature * adapter['cond_weight']], dim=1)

    return ret_feat_map, ret_feat_seq

def save_wave(waveform, savepath, name="outwav"):
    if type(name) is not list:
        name = [name] * waveform.shape[0]

    for i in range(waveform.shape[0]):
        if(waveform.shape[0] > 1):
            fname = "%s.wav" % (
                    os.path.basename(name[i])
                    if (not ".mp3" in name[i] and not ".wav" in name[i])
                    else os.path.basename(name[i]).split(".")[0]
                )
        else:
            fname = "%s.wav" % os.path.basename(name[i]) if (not ".mp3" in name[i] and not ".wav" in name[i]) else os.path.basename(name[i]).split(".")[0]
            
        path = os.path.join(
            savepath, fname
        )
        print("Save audio to %s" % path)
        if waveform.ndim == 3:
            sf.write(path, waveform[i, 0], samplerate=16000, format='wav')
        else:
            sf.write(path, waveform[0], samplerate=16000, format='wav')
        # sf.write(path, waveform[i, 0], samplerate=48000)


def parsr_args():
    """get the base argument parser for inference scripts"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        # default="./output",
    )

    parser.add_argument(
        "-t",
        "--text",
        type=str,
        required=False,
        # default="A female vocalist sings this captivating melody. The tempo is medium witlenthusiastic drumming, soft keyboard harmony, steady bass lines and a middle eastern instrumentThe song is rhythmic, catchy, peppy, vivacious and has a dance groove. This song is a RegionalPop song.",
        default="Compose a narrative-rich music piece using emotionally expressive melodies and symphonic instruments to tell a story.",
        help="Text prompt to the model for audio generation",
    )

    parser.add_argument(
        "-tl",
        "--text_list",
        type=str,
        required=False,
        default="",
        help="A file that contains text prompt to the model for audio generation",
    )

    parser.add_argument(
        '--neg_prompt',
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help='negative prompt',
    )

    parser.add_argument(
        '--cond_path',
        type=str,
        default=None,
        help='condition image path',
    )

    parser.add_argument(
        '--cond_inp_type',
        type=str,
        default='image',
        help='the type of the input condition image, take depth T2I as example, the input can be raw image, '
        'which depth will be calculated, or the input can be a directly a depth map image',
    )

    parser.add_argument(
        "--plms",
        default=False,
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        required=False,
        default=200,
        help="The sampling step for DDIM",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="The checkpoint you gonna use",
        default="audioldm2-music-665k",
        choices=["audioldm_48k", "audioldm_16k_crossattn_t5", "audioldm2-full", "audioldm2-music-665k", "audioldm2-full-large-1150k","audioldm2-speech-ljspeech","audioldm2-speech-gigaspeech"]
    )

    parser.add_argument(
        '--vae_ckpt',
        type=str,
        default=None,
        help='vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded',
    )

    parser.add_argument(
        '--which_cond',
        type=list,
        # required=True,
        choices=[e.name for e in ExtraCondition],
        help='which condition modality you want to test',
        default=['melody']
    )

    parser.add_argument(
        '--adapter_ckpt',
        type=str,
        default='experiments/melody_intro/models/model_ad_20000.pth',
        # default='experiments/chord_archived_20231214_234740/models/model_ad_250000.pth',
        # default='experiments/chord/models/model_ad_10000.pth', 
        # default='experiments/melody/models/model_ad_11000.pth', 
        help='path to checkpoint of adapter',
    )

    parser.add_argument(
        '--chord_adapter_ckpt',
        type=str,
        default='experiments/chord/models/model_ad_10000.pth', 
        # default='experiments/melody/models/model_ad_11000.pth', 
        help='path to checkpoint of adapter',
    )

    parser.add_argument(
        '--melody_adapter_ckpt',
        type=str,
        default='experiments/melody_chorus/models/model_ad_20000.pth',
        # default='experiments/melody/models/model_ad_11000.pth', 
        help='path to checkpoint of adapter',
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/stable-diffusion/sd-v1-inference.yaml',
        help='path to config which constructs SD model',
    )

    parser.add_argument(
        '--C',
        type=int,
        default=4,
        help='latent channels',
    )

    parser.add_argument(
        '--f',
        type=int,
        default=8,
        help='downsampling factor',
    )

    parser.add_argument(
        "-gs",
        "--guidance_scale",
        type=float,
        required=False,
        default=6.0,
        help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity), unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "-dur",
        "--duration",
        type=float,
        required=False,
        default=10.00,
        help="The duration of the samples",
    )

    parser.add_argument(
        "-s_rate",
        "--sample_rate",
        type=float,
        required=False,
        default=16000,
        help="The sample rate of the output waveform",
    )

    parser.add_argument(
        '--cond_tau',
        type=float,
        default=1.0,
        help='timestamp parameter that determines until which step the adapter is applied, '
        'similar as Prompt-to-Prompt tau',
    )

    parser.add_argument(
        '--style_cond_tau',
        type=float,
        default=1.0,
        help='timestamp parameter that determines until which step the adapter is applied, '
             'similar as Prompt-to-Prompt tau',
    )
    parser.add_argument(
        '--chord_weight',
        type=float,
        default=0.4,
        help='the adapter features are multiplied by the cond_weight. The larger the cond_weight, the more aligned '
        'the generated image and condition will be, but the generated quality may be reduced',
    )

    parser.add_argument(
        '--melody_weight',
        type=float,
        default=0.6,
        help='the adapter features are multiplied by the cond_weight. The larger the cond_weight, the more aligned '
        'the generated image and condition will be, but the generated quality may be reduced',
    )

    parser.add_argument(
        '--cond_weight',
        type=float,
        default=1.0,
        help='the adapter features are multiplied by the cond_weight. The larger the cond_weight, the more aligned '
        'the generated image and condition will be, but the generated quality may be reduced',
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
        help="Change this value (any integer number) will lead to a different generation result.",
    )

    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        required=False,
        default=16,
        help="Generate how many samples at the same time",
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        required=False,
        help="The device for computation. If not specified, the script will automatically choose the device based on your environment.",
        default="auto",
    )
    
    parser.add_argument(
        '--launcher',
        default='pytorch',
        type=str,
        help='node rank for distributed training'
    )

    parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        help='node rank for distributed training'
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        # default=1,
    )

    parser.add_argument(
        "--test_on_dataset",
        type=bool,
        default=False,
        help="choose whether to test based on dataset or free text.",
    )

    parser.add_argument(
        "--cond_filepath",
        type=str,
        default="outputs/tmp_['melody']/091219_2_melody.wav",
        help="path of reference music file.",
    )

    parser.add_argument(
        "--clap_score_path",
        type=str,
        default='outputs',
        help="output path of CLAP score.",
    )

    parser.add_argument(
        "--original_generation",
        type=bool,
        default=False,
        help="path of reference music file.",
    )
    opt = parser.parse_args()
    return opt

def main():
    opt = parsr_args()

    # distributed setting
    init_dist(opt.launcher)
    torch.backends.cudnn.benchmark = True
    device = 'cuda'
    torch.cuda.set_device(opt.local_rank)
    torch.set_float32_matmul_precision("high")
    
    which_cond = opt.which_cond
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    audioldm2_config = get_basic_config()
    audioldm2_config["model"]["params"]["cond_stage_config"]["crossattn_audiomae_generated"]["params"]["use_gt_mae_output"] = False
    audioldm2_config["step"]["limit_val_batches"] = None

    # prepare models
    audioldm2 = build_model(model_name=opt.model_name)
    audioldm2_model = torch.nn.parallel.DistributedDataParallel(
        audioldm2,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)
    
    if opt.original_generation is False:
        # Adapter fusion
        if len(which_cond)>1:
            adapter = []
            for cond_name in which_cond:
                adapter.append(get_adapters(opt, getattr(ExtraCondition, cond_name)))
        else:
            adapter = get_adapters(opt, getattr(ExtraCondition, which_cond[0]))
    else:
        adapter = None

    # test on dataset
    if opt.test_on_dataset:
        # test_dataset = ChordDataset(config=audioldm2_config, split="test", use_gt=False)
        # test_dataset = MelodyDataset(config=audioldm2_config, split="test", use_gt=False)
        test_dataset = AudioDataset(extra_cond_opt=["chord", "melody"],
                                    config=audioldm2_config, 
                                    split="test", 
                                    use_gt=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchsize,
            shuffle=(test_sampler is None),
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=test_sampler)
        print(
            "The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s"
            % (len(test_dataset), len(test_dataloader), opt.batchsize)
        )

        # Generate samples by original model
        if opt.original_generation:
            if opt.save_path is None:
                opt.save_path = f'outputs/test'
            os.makedirs(opt.save_path, exist_ok=True) 
            CLAP_scores = {}
            # inference
            with torch.no_grad():
                    # audioldm2_model.module.ema_scope(), \
                    # autocast('cuda'):
                for _, data in enumerate(test_dataloader):
                    seed_everything(opt.seed)

                    waveform = audioldm2_model.module.generate_batch(
                        data,
                        unconditional_guidance_scale=opt.guidance_scale,
                        ddim_steps=opt.ddim_steps,
                        n_gen=opt.n_samples,
                        duration=opt.duration,
                        use_plms=opt.plms,
                    )

                    # result = diffusion_inference(opt, audioldm2_model, adapter_features, append_to_context)
                    similarity = audioldm2_model.module.clap.cos_similarity(
                        torch.FloatTensor(waveform).squeeze(1), data['text']
                    )
                    # assert(
                    #     len(data['fname']) == len(similarity)
                    #     ), "Error: The number of files and the length of the similarity list are inconsistent. Please check if there exists issue during the CLAP scores calculation process."
                    if len(data['fname']) > 1:
                        CLAP_scores.update({fname: sim for fname, sim in zip(data['fname'], similarity)})
                    else:
                        CLAP_scores.update({data['fname'][0]: similarity})
                    save_wave(waveform, opt.save_path, name=data['fname'])

        # Generate samples with extra conditional adapter    
        else:
            if opt.save_path is None:
                if type(which_cond) is list:
                    opt.save_path = f'outputs/test-fusion_0.4c_0.6m'
                else:
                    opt.save_path = f'outputs/test-{which_cond}'
            os.makedirs(opt.save_path, exist_ok=True) 
            CLAP_scores = {}

            # inference
            with torch.no_grad():
                    # audioldm2_model.module.ema_scope(), \
                    # autocast('cuda'):
                for _, data in enumerate(test_dataloader):
                    seed_everything(opt.seed)
                    cond = []
                    if type(which_cond) is list:
                        for cond_name in which_cond:                            
                            cond.append(data['extra_cond_log_mel_spec'][cond_name].unsqueeze(1).to(device))
                    else:
                        cond = data['extra_cond_log_mel_spec'][cond_name].unsqueeze(1).to(device)
                    adapter_features, append_to_context = get_adapter_feature(cond, adapter)

                    waveform = audioldm2_model.module.generate_batch(
                        batch=data,
                        unconditional_guidance_scale=opt.guidance_scale,
                        ddim_steps=opt.ddim_steps,
                        n_gen=opt.n_samples,
                        duration=opt.duration,
                        use_plms=opt.plms,
                        features_adapter=adapter_features,
                        append_to_context=append_to_context,
                        cond_tau=opt.cond_tau,
                        style_cond_tau=opt.style_cond_tau,
                    )

                    # result = diffusion_inference(opt, audioldm2_model, adapter_features, append_to_context)
                    similarity = audioldm2_model.module.clap.cos_similarity(
                        torch.FloatTensor(waveform).squeeze(1), data['text']
                    )
                    # assert(
                    #     len(data['fname']) == len(similarity)
                    #     ), "Error: The number of files and the length of the similarity list are inconsistent. Please check if there exists issue during the CLAP scores calculation process."
                    if len(data['fname']) > 1:
                        CLAP_scores.update({fname: sim for fname, sim in zip(data['fname'], similarity)})
                    else:
                        CLAP_scores.update({data['fname'][0]: similarity})
                    save_wave(waveform, opt.save_path, name=data['fname'])

        total_score = 0.0
        for fname in CLAP_scores:
            total_score += CLAP_scores[fname]
        mean_score = total_score * 1.0 / len(CLAP_scores)  
        CLAP_scores['mean_score'] = mean_score
        print("average CLAP_score is: %.3f" % mean_score)
        for key, value in CLAP_scores.items():
            if torch.is_tensor(value):
                CLAP_scores[key] = value.item()

        if opt.original_generation:
            output_file = os.path.join(opt.clap_score_path, 'clap.json')
        else:
            output_file = os.path.join(opt.clap_score_path, f'clap_{which_cond}.json')
        with open(output_file, "w") as outfile:
            json.dump(CLAP_scores, outfile)

    # generate several samples
    else:
        if opt.save_path is None:
            opt.save_path = f'outputs/tmp_{which_cond}'
        os.makedirs(opt.save_path, exist_ok=True) 
        track_id = os.path.splitext(os.path.basename(opt.cond_filepath))[0]

        seed_everything(opt.seed)
        waveform = None
        data = make_batch_for_text_to_audio(opt.text, 
                                            config=audioldm2_config,
                                            waveform=waveform, 
                                            batchsize=1,
                                            cond_extracted=True,
                                            extra_cond_type='melody',
                                            extra_cond_file=opt.cond_filepath,
                                            sampling_rate=opt.sample_rate,
                                            output_path=opt.save_path
                                            )                          
        cond = data['extra_cond_log_mel_spec'].unsqueeze(1).to(device)
        adapter_features, append_to_context = get_adapter_feature(cond, adapter)

        with torch.no_grad():
            # Generate multiple samples at a time and filter out the best
            # The condition to the diffusion wrapper can have many format
            waveform = audioldm2_model.module.generate_batch(
                batch=data,
                unconditional_guidance_scale=opt.guidance_scale,
                ddim_steps=opt.ddim_steps,
                n_gen=opt.n_samples,
                duration=opt.duration,
                use_plms=opt.plms,
                features_adapter=adapter_features,
                append_to_context=append_to_context,
                cond_tau=opt.cond_tau,
                style_cond_tau=opt.style_cond_tau,
            )

        save_wave(waveform, savepath=opt.save_path, name=f"{track_id}_outro")
                
          
if __name__ == '__main__':
    main()
