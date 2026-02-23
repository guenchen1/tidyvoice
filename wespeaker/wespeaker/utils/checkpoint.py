# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#               2021 Hongji Wang (jijijiang77@gmail.com)
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

import torch
import logging


def load_checkpoint(model: torch.nn.Module, path: str):
    checkpoint = torch.load(path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    missing_keys, unexpected_keys = model.load_state_dict(state_dict,
                                                          strict=False)
    for key in missing_keys:
        logging.warning('missing tensor: {}'.format(key))
    for key in unexpected_keys:
        logging.warning('unexpected tensor: {}'.format(key))
    
    return checkpoint


def save_checkpoint(model: torch.nn.Module, path: str, optimizer=None, scheduler=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
        
    data = {'model': state_dict}
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        data['scheduler'] = scheduler.state_dict()
        
    torch.save(data, path)
