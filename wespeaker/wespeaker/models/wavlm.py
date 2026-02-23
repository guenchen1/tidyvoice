import torch
import torch.nn as nn
import wespeaker.models.pooling_layers as pooling_layers
from transformers import WavLMModel, WavLMConfig


class WavLMSpeaker(nn.Module):
    def __init__(self,
                 pretrained_path="microsoft/wavlm-base",
                 embed_dim=256,
                 freeze_feature_extractor=True,
                 freeze_all_layers=False,
                 pooling_func='ASTP',
                 layer_idx=None,
                 feat_dim=768):  # wavlm-base hidden size is 768
        super().__init__()

        # Load WavLM backbone
        try:
            config = WavLMConfig.from_pretrained(pretrained_path)
            config.layerdrop = 0.0
            config.output_hidden_states = True
            self.backbone = WavLMModel.from_pretrained(
                pretrained_path, config=config)
            print(
                f"Loaded WavLM from {pretrained_path} (LayerDrop disabled)")
        except Exception as e:
            print(f"Warning: Could not load from {pretrained_path}: {e}")
            print("Initializing random WavLM config")
            config = WavLMConfig()
            config.output_hidden_states = True
            self.backbone = WavLMModel(config)

        if freeze_feature_extractor or freeze_all_layers:
            if hasattr(self.backbone, "freeze_feature_extractor"):
                self.backbone.freeze_feature_extractor()
            else:
                self.backbone.freeze_feature_encoder()

        if freeze_all_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Freezing ALL WavLM parameters")

        self.layer_idx = layer_idx

        # Pooling layer (aggregates frame-level feats to utterance-level)
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=feat_dim)
        self.pool_out_dim = self.pool.get_out_dim()

        # Projection layer
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)

    def forward(self, x):
        # x shape: (B, T_samples) - raw waveform
        outputs = self.backbone(x)

        if self.layer_idx is not None:
            num_layers = len(outputs.hidden_states)
            if self.layer_idx >= num_layers:
                print(
                    f"Warning: layer_idx {self.layer_idx} out of bounds "
                    f"(max {num_layers-1}). Using last layer.")
                features = outputs.last_hidden_state
            else:
                features = outputs.hidden_states[self.layer_idx]
        else:
            features = outputs.last_hidden_state  # (B, T_frames, D)

        # Pooling expects (B, D, T)
        features = features.transpose(1, 2)

        stats = self.pool(features)
        embed = self.bn(stats)
        embed = self.linear(embed)

        return embed


def WavLM_Base(pretrained_path="microsoft/wavlm-base",
               embed_dim=256, feat_dim=768, **kwargs):
    return WavLMSpeaker(pretrained_path=pretrained_path,
                        embed_dim=embed_dim, feat_dim=feat_dim, **kwargs)


def WavLM_Base_Plus(pretrained_path="microsoft/wavlm-base-plus",
                    embed_dim=256, feat_dim=768, **kwargs):
    return WavLMSpeaker(pretrained_path=pretrained_path,
                        embed_dim=embed_dim, feat_dim=feat_dim, **kwargs)


def WavLM_Large(pretrained_path="microsoft/wavlm-large",
                embed_dim=256, feat_dim=1024, **kwargs):
    return WavLMSpeaker(pretrained_path=pretrained_path,
                        embed_dim=embed_dim, feat_dim=feat_dim, **kwargs)
