# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os

import fire
import kaldiio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wespeaker.dataset.dataset import Dataset
from wespeaker.dataset.dataset_utils import apply_cmvn, spec_aug
from wespeaker.frontend import *
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.utils import parse_config_or_kwargs, validate_path


def extract(config='conf/config.yaml', **kwargs):
    # parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)

    model_path = configs['model_path']
    embed_ark = configs['embed_ark']
    batch_size = configs.get('batch_size', 1)
    num_workers = configs.get('num_workers', 1)

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    test_conf = copy.deepcopy(configs['dataset_args'])
    # model: frontend (optional) => speaker model
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    frontend_type = test_conf.get('frontend', 'fbank')
    if frontend_type != 'fbank' and frontend_type != 'raw'and frontend_type != "raw_fbank":
        frontend_args = frontend_type + "_args"
        print('Initializing frontend model (this could take some time) ...')
        frontend = frontend_class_dict[frontend_type](
            **test_conf[frontend_args], sample_rate=test_conf['resample_rate'])
        model.add_module("frontend", frontend)
    print('Loading checkpoint ...')
    load_checkpoint(model, model_path)
    print('Finished !!! Start extracting ...')
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model.to(device).eval()

    # test_configs
    # test_conf = copy.deepcopy(configs['dataset_args'])
    test_conf['speed_perturb'] = False
    if 'fbank_args' in test_conf:
        test_conf['fbank_args']['dither'] = 0.0
    if 'raw_args' in test_conf: # Fix for raw (Wav2Vec2) frontend
        pass 
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['aug_prob'] = configs.get('aug_prob', 0.0)
    test_conf['filter'] = False

    dataset = Dataset(configs['data_type'],
                      configs['data_list'],
                      test_conf,
                      spk2id_dict={},
                      whole_utt=(batch_size == 1),
                      reverb_lmdb_file=configs.get('reverb_data', None),
                      noise_lmdb_file=configs.get('noise_data', None),
                      repeat_dataset=False)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            prefetch_factor=4)

    validate_path(embed_ark)
    embed_ark = os.path.abspath(embed_ark)
    embed_scp = embed_ark[:-3] + "scp"

    with torch.no_grad():
        with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
                                 embed_scp) as writer:
            for _, batch in tqdm(enumerate(dataloader)):
                utts = batch['key']
                if frontend_type == 'fbank':
                    features = batch['feat']
                    features = features.float().to(device)  # (B,T,F)
                elif frontend_type == 'raw':
                    wavs = batch['wav']  # (B,1,W)
                    features = wavs.squeeze(1).float().to(device) # (B,W)
                    # No frontend processing needed, features ARE the input for Wav2Vec2
                elif frontend_type == 'raw_fbank':
                    # 這是你的融合模式：直接從 batch 拿 CPU 算好的 feat，並準備好 wav
                    features = batch['feat'].float().to(device)  # (B,T,F)
                else:  # 's3prl'
                    wavs = batch['wav']  # (B,1,W)
                    wavs = wavs.squeeze(1).float().to(device)  # (B,W)
                    wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                        wavs.shape[0]).to(device)  # (B)
                    features, _ = model.frontend(wavs, wavs_len)

                # apply cmvn
                if test_conf.get('cmvn', True):
                    features = apply_cmvn(features,
                                          **test_conf.get('cmvn_args', {}))
                # spec augmentation
                if test_conf.get('spec_aug', False):
                    features = spec_aug(features, **test_conf['spec_aug_args'])

                model_class = model.module.__class__.__name__ if hasattr(model, 'module') else model.__class__.__name__
                if model_class == 'FusionSpeakerModel' or model_class == 'FusionWavLMModel':
                    # 假設 batch['key'] 是 uttid 清單
                    fusion_input = {'key': batch['key'], 'feat': features}
                    if 'wav' in batch:
                        features_wav = batch['wav'].squeeze(1).float().to(device)
                        fusion_input['wav'] = features_wav  # 如果有 WAV，直接傳入
                    
                    outputs = model(fusion_input)
                else:
                    outputs = model(features)
                # Forward through model
                #outputs = model(features)  # embed or (embed_a, embed_b)
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                embeds = embeds.cpu().detach().numpy()  # (B,F)

                for i, utt in enumerate(utts):
                    embed = embeds[i]
                    writer(utt, embed)


if __name__ == '__main__':
    fire.Fire(extract)
