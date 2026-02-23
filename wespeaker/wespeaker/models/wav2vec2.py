import torch
import torch.nn as nn
import wespeaker.models.pooling_layers as pooling_layers
from transformers import Wav2Vec2Model, Wav2Vec2Config

class Wav2Vec2Speaker(nn.Module):
    def __init__(self, 
                 pretrained_path="facebook/wav2vec2-base", 
                 embed_dim=256, 
                 freeze_feature_extractor=True,
                 freeze_all_layers=False,
                 pooling_func='ASTP',
                 layer_idx=None,
                 feat_dim=768): # wav2vec2-base hidden size is 768
        super().__init__()
        
        # Load Wav2Vec2 backbone
        # We try to load local first, if not exists, load from hub
        try:
            # We must load config first to disable layerdrop, otherwise DDP will complain about unused parameters
            config = Wav2Vec2Config.from_pretrained(pretrained_path)
            config.layerdrop = 0.0
            config.output_hidden_states = True # Enable hidden states output
            self.backbone = Wav2Vec2Model.from_pretrained(pretrained_path, config=config)
            print(f"Loaded Wav2Vec2 from {pretrained_path} (LayerDrop disabled)")
        except Exception as e:
            print(f"Warning: Could not load from {pretrained_path}: {e}")
            print("Initializing random Wav2Vec2 config")
            config = Wav2Vec2Config()
            config.output_hidden_states = True
            self.backbone = Wav2Vec2Model(config)
            
        if freeze_feature_extractor or freeze_all_layers:
            if hasattr(self.backbone, "freeze_feature_extractor"):
                self.backbone.freeze_feature_extractor()
            else:
                self.backbone.freeze_feature_encoder() # New name in transformers > 5 or recent 4.x
        
        if freeze_all_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Freezing ALL Wav2Vec2 parameters")

        self.layer_idx = layer_idx

        # Pooling layer (aggregates frame-level feats to utterance-level)
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=feat_dim)
        self.pool_out_dim = self.pool.get_out_dim()
        
        # Projection layer
        # Use LayerNorm for small batch size robustness
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)

    def forward(self, x):
        # x shape: (B, T_samples) - raw waveform
        # Wav2Vec2 expects raw waveform input
        
        # Ensure input is normalized if not already (Wav2Vec2 is sensitive to amplitude)
        # Assuming wespeaker dataloader might give varying amplitudes
        
        # If accessing a specific layer, we rely on output_hidden_states=True in config
        outputs = self.backbone(x)

        if self.layer_idx is not None:
             # hidden_states is tuple of (embeddings, layer_1, layer_2, ... layer_N)
             # Index 0 is embeddings. Layer 1 is 1st transformer block output.
             # User Request: "skip the first layer, hop every 2 layers" (handled in script)
             # Here we just index into the tuple.
             # The tuple likely excludes the initial conv output unless specified?
             # Transformers doc: hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True):
             # Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
             
             # If layer_idx is passed, we assume it refers to the transformer layers (1-12)
             # But let's support flexible indexing. 
             # If layer_idx is simple integer, use it directly.
             # Note: outputs.hidden_states[0] is CNN features/Embeddings. outputs.hidden_states[1] is Layer 1 output.
             # If user passes layer_idx 12 (for base), it's the last layer.
             
             # Safeguard index
             num_layers = len(outputs.hidden_states)
             if self.layer_idx >= num_layers:
                 print(f"Warning: layer_idx {self.layer_idx} out of bounds (max {num_layers-1}). Using last layer.")
                 features = outputs.last_hidden_state
             else:
                 features = outputs.hidden_states[self.layer_idx]
        else:
            features = outputs.last_hidden_state # (B, T_frames, D)
        
        # Pooling expects (B, D, T)
        features = features.transpose(1, 2)
        
        stats = self.pool(features)
        embed = self.bn(stats)
        embed = self.linear(embed)
        
        return embed

def Wav2Vec2_Base(pretrained_path="facebook/wav2vec2-base", embed_dim=256, feat_dim=768, **kwargs):
    return Wav2Vec2Speaker(pretrained_path=pretrained_path, embed_dim=embed_dim, feat_dim=feat_dim, **kwargs)

def Wav2Vec2_Large(pretrained_path="facebook/wav2vec2-large-xlsr-53", embed_dim=256, feat_dim=1024, **kwargs):
    return Wav2Vec2Speaker(pretrained_path=pretrained_path, embed_dim=embed_dim, feat_dim=feat_dim, **kwargs)
