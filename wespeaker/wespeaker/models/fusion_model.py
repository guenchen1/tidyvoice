import torch
import torch.nn as nn
from torch.autograd import Function
from wespeaker.models.speaker_model import get_speaker_model
import torch.nn.functional as F


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

    def set_lambda(self, lambda_val):
        """動態調整 lambda 值"""
        self.lambda_val = lambda_val


class FusionSpeakerModel(nn.Module):
    def __init__(self, 
                 w2v2_type='Wav2Vec2Speaker',
                 resnet_type='SimAM_ResNet34_ASP',
                 w2v2_args={},
                 resnet_args={},
                 embed_dim=512,
                 use_layernorm=True,
                 wav_scp_path='data/tidyvoice_train/wav.scp',
                 # 語言對抗訓練參數
                 num_languages=0,
                 grl_lambda=1.0):
        super().__init__()
        # 載入 wav.scp 映射表
        self.wav_dict = {}
        try:
            with open(wav_scp_path, 'r') as f:
                for line in f:
                    utt, path = line.strip().split(None, 1)
                    self.wav_dict[utt] = path
        except Exception as e:
            print(f"[FusionSpeakerModel] Warning: wav.scp not loaded: {e}")
        # 取出 w2v2_args 的 model_init，避免傳給 __init__
        w2v2_args = w2v2_args.copy()
        w2v2_model_init = w2v2_args.pop('model_init', None)
        self.w2v2 = get_speaker_model(w2v2_type)(**w2v2_args)
        if w2v2_model_init:
            print(f"[FusionSpeakerModel] Loading wav2vec2 pretrained weights from: {w2v2_model_init}")
            state = torch.load(w2v2_model_init, map_location='cpu')
            if 'model' in state:
                self.w2v2.load_state_dict(state['model'], strict=False)
            else:
                self.w2v2.load_state_dict(state, strict=False)
            print(f"[FusionSpeakerModel] wav2vec2 weights loaded successfully.")

        # 取出 resnet_args 的 model_init，避免傳給 __init__
        resnet_args = resnet_args.copy()
        resnet_model_init = resnet_args.pop('model_init', None)
        self.resnet = get_speaker_model(resnet_type)(**resnet_args)
        if resnet_model_init:
            print(f"[FusionSpeakerModel] Loading resnet pretrained weights from: {resnet_model_init}")
            state = torch.load(resnet_model_init, map_location='cpu')
            if 'model' in state:
                self.resnet.load_state_dict(state['model'], strict=False)
            else:
                self.resnet.load_state_dict(state, strict=False)
            print(f"[FusionSpeakerModel] resnet weights loaded successfully.")
        # ========== Fusion Bottleneck (特徵壓縮) ==========
        # W2V2 (256) + ResNet (256) -> Concat (512) -> Bottleneck -> Output (256)
        # 這裡會自動計算 concat 後的維度
        w2v2_dim = w2v2_args.get('embed_dim', 256)
        resnet_dim = resnet_args.get('embed_dim', 256)
        fusion_input_dim = w2v2_dim + resnet_dim
        
        self.embed_dim = embed_dim
        self.use_layernorm = use_layernorm
        if use_layernorm:
            # 修正: LayerNorm 應作用於 Concat 後的維度 (fusion_input_dim)，而非輸出維度
            self.ln = nn.LayerNorm(fusion_input_dim)
        
        self.use_bottleneck = True
        self.bottleneck = nn.Sequential(
            nn.Linear(fusion_input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        print(f"[FusionSpeakerModel] Bottleneck initialized: {fusion_input_dim} -> {embed_dim}")

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
            print(f"[FusionSpeakerModel] Language Adversarial Training enabled: {num_languages} languages, lambda={grl_lambda}")

    def forward(self, feat, return_lang_logits=False):
        """
        Args:
            feat: dict with 'key', 'feat', and optionally 'wav'
            return_lang_logits: 若為 True，回傳 (combined, lang_logits)；否則只回傳 combined
        Returns:
            combined: (B, embed_dim) 融合後的語者特徵向量
            lang_logits: (B, num_languages) 語言分類 logits (僅當 return_lang_logits=True)
        """
        # 支援 dict: {'key': uttid, 'feat': fbank, 'wav': wav_tensor (optional)}
        if not isinstance(feat, dict) or 'key' not in feat or 'feat' not in feat:
            raise RuntimeError('FusionSpeakerModel 需要 batch dict，且含 key/feat')
        uttid = feat['key']
        feat_tensor = feat['feat']
        # 若直接傳入 wav tensor，優先使用
        if 'wav' in feat:
            wav = feat['wav']
            # 修正: 若 wav 為 (B, 1, T)，Wav2Vec2 需要 (B, T)，需 squeeze
            if wav.dim() == 3:
                wav = wav.squeeze(1)
        else:
            # 動態讀檔
            import torchaudio
            # 確保 uttid 為 list
            if not isinstance(uttid, (list, tuple)):
                batch_size = feat_tensor.shape[0]
                uttid = [uttid] * batch_size
            wav_list = []
            for u in uttid:
                wav_path = self.wav_dict.get(u, None)
                if wav_path is None:
                    raise RuntimeError(f'wav path for {u} not found in wav.scp')
                wav_data, sr = torchaudio.load(wav_path)
                if wav_data.shape[0] > 1:
                    wav_data = wav_data.mean(dim=0, keepdim=True)
                wav_list.append(wav_data)
            max_len = max([w.shape[1] for w in wav_list])
            wav_batch = torch.zeros(len(wav_list), 1, max_len)
            for i, w in enumerate(wav_list):
                wav_batch[i, 0, :w.shape[1]] = w
            wav = wav_batch.squeeze(1)
        # 確保 wav tensor 與模型權重在同一裝置與 dtype
        device = feat_tensor.device
        dtype = next(self.w2v2.parameters()).dtype
        wav = wav.to(device=device, dtype=dtype)
        w2v2_emb = self.w2v2(wav)
        resnet_emb = self.resnet(feat_tensor)
        combined = torch.cat([w2v2_emb, resnet_emb], dim=1)
        if self.use_layernorm:
            combined = self.ln(combined)

        # Apply Fusion Bottleneck
        if hasattr(self, 'bottleneck'):
            combined = self.bottleneck(combined)

        # ========== 語言對抗分支 ==========
        if return_lang_logits and self.use_language_adversarial:
            x_adv = self.grl(combined)  # 通過梯度反轉層
            lang_logits = self.language_head(x_adv)  # 語言分類
            return combined, lang_logits

        # 推論模式：僅回傳融合特徵
        return combined
