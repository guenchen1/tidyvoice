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

import tableprint as tp

import torch
import torchnet as tnt
from wespeaker.dataset.dataset_utils import apply_cmvn, spec_aug


def run_epoch(dataloader, epoch_iter, model, criterion, optimizer, scheduler,
              margin_scheduler, epoch, logger, scaler, device, configs, writer=None):
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        utts = batch['key']
        targets = batch['label']
        targets = targets.long().to(device)  # (B)
        wavs = None  # 預設 None，確保後續 if wavs is not None 不會出錯
        if frontend_type == 'fbank':
            features = batch['feat']  # (B,T,F)
            features = features.float().to(device)
        elif frontend_type == 'raw':
            wavs = batch['wav']  # (B,1,W)
            features = wavs.squeeze(1).float().to(device)  # (B,W)
        elif frontend_type == 'raw_fbank':
            # 這是你的融合模式：直接從 batch 拿 CPU 算好的 feat，並準備好 wav
            features = batch['feat'].float().to(device)  # (B,T,F)
        else:  # 's3prl'
            wavs = batch['wav']  # (B,1,W)
            wavs = wavs.squeeze(1).float().to(device)  # (B,W)
            wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                wavs.shape[0]).to(device)  # (B)
            with torch.amp.autocast('cuda', enabled=configs['enable_amp']):
                # Support non-DDP execution
                frontend_model = model.module if hasattr(model, 'module') else model
                features, _ = frontend_model.frontend(wavs, wavs_len)

        with torch.amp.autocast('cuda', enabled=configs['enable_amp']):
            # apply cmvn
            if configs['dataset_args'].get('cmvn', True):
                features = apply_cmvn(
                    features, **configs['dataset_args'].get('cmvn_args', {}))
            # spec augmentation
            if configs['dataset_args'].get('spec_aug', False):
                features = spec_aug(features,
                                    **configs['dataset_args']['spec_aug_args'])

                    # 只針對 FusionSpeakerModel 組 dict 傳入，其它模型維持原生 tensor 輸入
            model_class = model.module.__class__.__name__ if hasattr(model, 'module') else model.__class__.__name__
            
            # 檢查是否有語言標籤 (用於語言對抗訓練)
            use_lang_adversarial = 'lang_label' in batch and model_class in ('FusionSpeakerModel', 'FusionWavLMModel')
            lang_targets = None
            if use_lang_adversarial:
                lang_targets = batch['lang_label'].long().to(device)
            
            if model_class in ('FusionSpeakerModel', 'FusionWavLMModel'):
                # 假設 batch['key'] 是 uttid 清單
                fusion_input = {'key': batch['key'], 'feat': features}
                if 'wav' in batch:
                    features_wav = batch['wav'].squeeze(1).float().to(device)
                    fusion_input['wav'] = features_wav  # 如果有 WAV，直接傳入
                
                # 若有語言對抗訓練，要取得 lang_logits
                if use_lang_adversarial:
                    embeds, lang_logits = model(fusion_input, return_lang_logits=True)
                else:
                    embeds = model(fusion_input)
                    lang_logits = None
            else:
                outputs = model(features)
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                lang_logits = None
            
            projection_model = model.module if hasattr(model, 'module') else model
            outputs = projection_model.projection(embeds, targets)
            if isinstance(outputs, tuple):
                outputs, loss_spk = outputs
            else:
                loss_spk = criterion(outputs, targets)
            
            # ========== 語言對抗損失 ==========
            loss_lang = torch.tensor(0.0, device=device)
            if use_lang_adversarial and lang_logits is not None and lang_targets is not None:
                # 過濾掉無效標籤 (-1)
                valid_mask = lang_targets >= 0
                if valid_mask.sum() > 0:
                    loss_lang = torch.nn.functional.cross_entropy(
                        lang_logits[valid_mask], lang_targets[valid_mask]
                    )
            
            # 總損失 = 語者損失 + 語言損失
            # 注意: GRL 層內部已經包含了梯度反轉的權重 lambda (用於 Backbone 反向傳播)，
            # Language Head 則使用正常的梯度更新。
            # 如果在此處再乘上 grl_lambda，會導致 Backbone 接收到的梯度被縮放了兩次 (lambda^2)，
            # 且會影響 Language Head 的學習率。
            # 通常建議此處 Loss 直接相加，透過 GRL 層的 lambda 控制對抗強度。
            loss = loss_spk + loss_lang

        # loss, acc
        loss_meter.add(loss.item())
        acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())


        # updata the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # log
        if (i + 1) % configs['log_batch_interval'] == 0:
            logger.info(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin()) +
                       (loss_meter.value()[0], acc_meter.value()[0]),
                       width=10,
                       style='grid'))
            
            if writer is not None:
                writer.add_scalar('Train/Loss', loss_meter.value()[0], cur_iter)
                writer.add_scalar('Train/Acc', acc_meter.value()[0], cur_iter)
                writer.add_scalar('Train/LR', scheduler.get_lr(), cur_iter)
                writer.add_scalar('Train/Margin', margin_scheduler.get_margin(), cur_iter)

        if (i + 1) == epoch_iter:
            break

    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
            (loss_meter.value()[0], acc_meter.value()[0]),
            width=10,
            style='grid'))
