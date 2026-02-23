import torch

def build_param_groups(model, base_lr, lr_groups):
    """
    根據參數名稱前綴分配不同學習率。
    
    Args:
        model (nn.Module): 訓練模型
        base_lr (float): 基礎學習率 (e.g. 1e-4)
        lr_groups (dict): 模組名稱對應學習率乘數的字典
                          e.g. {'w2v2_backbone': 0.1, 'resnet_head': 0.5}
    
    Returns:
        List[dict]: 用於 optimizer 的 parameter groups 列表
                    [{'params': [...], 'lr': ...}, ...]
    """
    
    # 定義群組儲存容器
    # 這裡的 key 對應 logic 內的判定規則
    grouped_params = {
        'w2v2_backbone': [],   # w2v2.backbone.*
        'w2v2_head': [],       # w2v2.pool.*, w2v2.linear.*
        'wavlm_backbone': [],  # wavlm.backbone.*
        'wavlm_head': [],      # wavlm.pool.*, wavlm.linear.*
        'resnet_backbone': [], # resnet.front.*
        'resnet_head': [],     # resnet.pooling.*, resnet.bottleneck.*
        'fusion': [],          # ln.*, bottleneck.*
        'language_head': [],   # language_head.*
        'projection': [],      # projection.*
        'other': []            # 未匹配到的參數 (預設 1.0x)
    }

    # 用於記錄已分配的參數，避免重複
    assigned_params = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        assigned_params.add(name)
        
        # 移除 DDP 可能產生的 module. 前綴，統一邏輯
        clean_name = name.replace('module.', '')
        
        # 分組邏輯
        if clean_name.startswith('w2v2.backbone'):
            grouped_params['w2v2_backbone'].append(param)
        elif clean_name.startswith('w2v2'):
            # w2v2 pool, linear, bn 等 (排除 backbone)
            grouped_params['w2v2_head'].append(param)
        elif clean_name.startswith('wavlm.backbone'):
            grouped_params['wavlm_backbone'].append(param)
        elif clean_name.startswith('wavlm'):
            # wavlm pool, linear, bn 等 (排除 backbone)
            grouped_params['wavlm_head'].append(param)
        elif clean_name.startswith('resnet.front'):
            grouped_params['resnet_backbone'].append(param)
        elif clean_name.startswith('resnet'):
            # resnet pooling, bottleneck (排除 front)
            grouped_params['resnet_head'].append(param)
        elif clean_name.startswith('ln'):
            grouped_params['fusion'].append(param)
        elif clean_name.startswith('language_head'):
            grouped_params['language_head'].append(param)
        elif clean_name.startswith('projection'):
            grouped_params['projection'].append(param)
        else:
            grouped_params['other'].append(param)

    # 建構 optimizer 所需的 param_groups
    param_groups = []
    
    for group_name, params in grouped_params.items():
        if not params:
            continue
            
        # 取得該群組的 multiplier，預設為 1.0
        multiplier = lr_groups.get(group_name, 1.0)
        group_lr = base_lr * multiplier
        
        param_groups.append({
            'params': params,
            'lr': group_lr,
            'initial_lr': group_lr, # 供 scheduler 使用
            'group_name': group_name # 方便 debug
        })
        
        print(f"[Discriminative LR] Group '{group_name}': {len(params)} params, lr={group_lr} (x{multiplier})")

    return param_groups
