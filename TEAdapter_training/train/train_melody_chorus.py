import argparse
import logging
import os
import os.path as osp
import torch
from audioldm2.pipeline import build_model
from audioldm2.utils import get_basic_config
from basicsr.utils import (get_env_info, get_root_logger, get_time_str, scandir)
from basicsr.utils.options import copy_opt_file, dict2str
from omegaconf import OmegaConf

from data.metadata.dataset_melody import MelodyDataset
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only
from audioldm2.latent_diffusion.modules.encoders.adapter import Adapter
from audioldm2.latent_diffusion.models.ddpm import LatentDiffusion
from audioldm2.pipeline import duration_to_latent_t_size
from audioldm2.utils import get_basic_config
from torch.utils.tensorboard import SummaryWriter
# from audioldm2.latent_diffusion.util import load_model_from_config

# environment variants
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '4353'

@master_only
def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(path, 'models'))
    os.makedirs(osp.join(path, 'training_states'))
    os.makedirs(osp.join(path, 'visualization'))


def load_resume_state(opt):
    resume_state_path = None
    if opt.auto_resume:
        state_path = osp.join('experiments', opt.name, 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt.resume_state_path = resume_state_path

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
    return resume_state


def parsr_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--bsize",
    #     type=int,
    #     #default=8,
    #     default=1,
    # )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        # default=1,
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--auto_resume",
        action='store_true',
        help="use plms sampling",
        default=False
    )
    # parser.add_argument(
    #     "--ckpt",
    #     type=str,
    #     # default="models/sd-v1-4.ckpt",
    #     default="models/v1-5-pruned-emaonly.ckpt",
    #     help="path to checkpoint of model",
    # )
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     default="configs/stable-diffusion/sd-v1-train.yaml",
    #     help="path to config which constructs model",
    # )
    parser.add_argument(
        "--name",
        type=str,
        default="melody",
        help="experiment name",
    )
    parser.add_argument(
        "--print_fq",
        type=int,
        default=100,
        help="frequency of output loss",
    )
    # parser.add_argument(
    #     "--H",
    #     type=int,
    #     default=512,
    #     help="image height, in pixel space",
    # )
    # parser.add_argument(
    #     "--W",
    #     type=int,
    #     default=512,
    #     help="image width, in pixel space",
    # )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--gpus",
        default=[0, 1],
        help="gpu idx",
    )
    parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        help='node rank for distributed training'
    )
    parser.add_argument(
        '--launcher',
        default='pytorch',
        type=str,
        help='node rank for distributed training'
    )
    # Ashley: New Add
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        required=False,
        default="",
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
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        default="./output",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="The checkpoint you gonna use",
        default="audioldm2-music-665k",
        choices=["audioldm2-full", "audioldm2-music-665k", "audioldm2-full-large-650k"]
    )

    parser.add_argument(
        "-bsize",
        "--batchsize",
        type=int,
        required=False,
        default=16,
        help="Generate how many samples at the same time",
    )

    # parser.add_argument(
    #     "--ddim_steps",
    #     type=int,
    #     required=False,
    #     default=200,
    #     help="The sampling step for DDIM",
    # )

    # parser.add_argument(
    #     "-gs",
    #     "--guidance_scale",
    #     type=float,
    #     required=False,
    #     default=3.5,
    #     help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
    # )

    # parser.add_argument(
    #     "-n",
    #     "--n_candidate_gen_per_text",
    #     type=int,
    #     required=False,
    #     default=3,
    #     help="Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation",
    # )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
        help="Change this value (any integer number) will lead to a different generation result.",
    )

    parser.add_argument(
        "-v_freq"
        "--val_every_n_epochs",
        type=int,
        required=False,
        default=500,
        help="Frequency of evaluation on the validation set",
    )
    opt = parser.parse_args()
    return opt


def main():
    opt = parsr_args()
    # config = OmegaConf.load(f"{opt.config}")
    
    # distributed setting
    init_dist(opt.launcher)
    torch.backends.cudnn.benchmark = True
    device = 'cuda'
    torch.cuda.set_device(opt.local_rank)
    torch.set_float32_matmul_precision("high")

    # Create an instance of SummaryWriter 
    writer = SummaryWriter()

    # # stable diffusion
    # model = load_model_from_config(config, f"{opt.ckpt}").to(device)

    # Ashley: Update
    audioldm2 = build_model(model_name=opt.model_name)
    # audioldm2 = torch.compile(audioldm2)

    # chord encoder
    model_ad = Adapter(cin=1 * 16, channels=[128, 256, 384, 640][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to(device)

    # to gpus
    model_ad = torch.nn.parallel.DistributedDataParallel(
        model_ad,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)
    audioldm2_model = torch.nn.parallel.DistributedDataParallel(
        audioldm2,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)
    
    # dataset
    audioldm2_config = get_basic_config()
    train_dataset = MelodyDataset(config=audioldm2_config, dataset_name=["musiccaps","FMA-chorus"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batchsize,
        shuffle=(train_sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=train_sampler)

    # optimizer
    params = list(model_ad.parameters())
    optimizer = torch.optim.AdamW(params, lr=audioldm2_config['model']['params']['base_learning_rate'])

    experiments_root = osp.join('experiments', f"{opt.name}_chorus")

    # resume state
    resume_state = load_resume_state(opt)
    if resume_state is None:
        mkdir_and_rename(experiments_root)
        start_epoch = 0
        current_iter = 0
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_chorus_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(audioldm2_config))
    else:
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_chorus_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(audioldm2_config))
        resume_optimizers = resume_state['optimizers']
        optimizer.load_state_dict(resume_optimizers)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']

    # # copy the yml file to the experiment root
    # copy_opt_file(opt.config, experiments_root)

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    for epoch in range(start_epoch, opt.epochs):
        train_dataloader.sampler.set_epoch(epoch)
        # train
        for _, data in enumerate(train_dataloader):
            current_iter += 1
            with torch.no_grad():
                # c = audioldm2_model.module.get_learned_conditioning(c=data['text'], key="crossattn_audiomae_generated", unconditional_cfg=0.1)
                z, c = audioldm2_model.module.get_input(
                                data,
                                "fbank", 
                                unconditional_prob_cfg=0.0,  # Do not output unconditional information in the c
                            )
                c = audioldm2_model.module.filter_useful_cond_dict(c)

                # inputs_dict = self.get_input()
                # inputs = inputs_dict[self.image_key]
                # waveform = inputs_dict["waveform"]
                # reconstructions, posterior = self(inputs)

            optimizer.zero_grad()
            audioldm2_model.zero_grad()
            melody_data = data['extra_cond_log_mel_spec'][opt.name].unsqueeze(1)
            features_adapter = model_ad(melody_data.to(device))
            l_pixel, loss_dict = audioldm2_model.module(z, c=c, features_adapter=features_adapter)
            l_pixel.backward()
            optimizer.step()

            writer.add_scalar('train/loss_simple_step', loss_dict['val/loss_simple'], current_iter) 
            writer.add_scalar('train/loss_vlb_step', loss_dict['val/loss_vlb'], current_iter)  
            writer.add_scalar('train/loss_step', loss_dict['val/loss'], current_iter) 
            
            if (current_iter + 1) % opt.print_fq == 0:
                logger.info(loss_dict)

            # save checkpoint
            rank, _ = get_dist_info()
            if (rank == 0) and ((current_iter + 1) % 1000 == 0):
                save_filename = f'model_ad_{current_iter + 1}.pth'
                save_path = os.path.join(experiments_root, 'models', save_filename)
                save_dict = {}
                state_dict = model_ad.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    save_dict[key] = param.cpu()
                torch.save(save_dict, save_path)
                # save state
                state = {'epoch': epoch, 'iter': current_iter + 1, 'optimizers': optimizer.state_dict()}
                save_filename = f'{current_iter + 1}.state'
                save_path = os.path.join(experiments_root, 'training_states', save_filename)
                torch.save(state, save_path)
            
            if (current_iter + 1) % len(train_dataset) == 0:
                writer.add_scalar('train/loss_simple_epoch', loss_dict['val/loss_simple'], epoch) 
                writer.add_scalar('train/loss_vlb_epoch', loss_dict['val/loss_vlb'], epoch)  
                writer.add_scalar('train/loss_epoch', loss_dict['val/loss'], epoch) 


if __name__ == '__main__':
    main()
