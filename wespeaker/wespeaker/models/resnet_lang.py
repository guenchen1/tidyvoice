import torch
import torch.nn as nn
from torch.autograd import Function
# from wespeaker.models.speaker_model import get_speaker_model  <-- Moved inside class to avoid circular import


# ========== Gradient Reversal Layer (GRL) ==========
class GradientReversalFunction(Function):
    """梯度反轉函數：前向傳播時為恆等函數，反向傳播時將梯度乘以 -lambda"""
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversal(nn.Module):
    """梯度反轉層：用於對抗訓練"""
    def __init__(self, lambda_val=1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


class ResNetLanguageModel(nn.Module):
    """
    ResNet + Language Adversarial Training Model
    Removed Wav2Vec2 branch as requested.
    """
    def __init__(self, 
                 resnet_type='SimAM_ResNet34_ASP',
                 resnet_args={},
                 embed_dim=256,
                 # 語言對抗訓練參數
                 num_languages=0,
                 grl_lambda=1.0,
                 **kwargs): # Catch extra args to allow reusing config files
        super().__init__()
        
        # 取出 resnet_args 的 model_init
        resnet_args = resnet_args.copy()
        resnet_model_init = resnet_args.pop('model_init', None)
        
        # 初始化 ResNet Backbone
        from wespeaker.models.speaker_model import get_speaker_model
        self.resnet = get_speaker_model(resnet_type)(**resnet_args)
        
        if resnet_model_init:
            print(f"[ResNetLanguageModel] Loading resnet pretrained weights from: {resnet_model_init}")
            state = torch.load(resnet_model_init, map_location='cpu')
            if 'model' in state:
                self.resnet.load_state_dict(state['model'], strict=False)
            else:
                self.resnet.load_state_dict(state, strict=False)
            print(f"[ResNetLanguageModel] resnet weights loaded successfully.")
            
        self.embed_dim = embed_dim

        # ========== 語言對抗訓練 (Language Adversarial Training) ==========
        self.num_languages = num_languages
        self.use_language_adversarial = num_languages > 0
        if self.use_language_adversarial:
            self.grl = GradientReversal(lambda_val=grl_lambda)
            # 語言分類頭: MLP (embed_dim -> 256 -> num_languages)
            self.language_head = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, num_languages)
            )
            print(f"[ResNetLanguageModel] Language Adversarial Training enabled: {num_languages} languages, lambda={grl_lambda}")

    def forward(self, feat, return_lang_logits=False):
        """
        Args:
            feat: dict with 'key', 'feat' (Fbank)
            return_lang_logits: 若為 True，回傳 (embedding, lang_logits)
        Returns:
            embedding: (B, embed_dim)
            lang_logits: (B, num_languages)
        """
        # 兼容原本的 Pipeline，feat 可能是 dict
        if isinstance(feat, dict):
            feat = feat['feat']

        # ResNet Forward
        # ResNet 通常輸出 (B, embed_dim)
        embedding = self.resnet(feat)

        # ========== 語言對抗分支 ==========
        if return_lang_logits and self.use_language_adversarial:
            x_adv = self.grl(embedding)  # 通過梯度反轉層
            lang_logits = self.language_head(x_adv)  # 語言分類
            return embedding, lang_logits

        # 推論模式
        return embedding
