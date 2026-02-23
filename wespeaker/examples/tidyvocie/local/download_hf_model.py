import os
from transformers import Wav2Vec2Model, Wav2Vec2Config, AutoProcessor

model_name = "facebook/wav2vec2-base"
# Save to the experiment directory structure you requested
save_directory = "exp/wav2vec2_tidy/wav2vec2-base"

print(f"Downloading {model_name} to {save_directory}...")

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Download and save model
model = Wav2Vec2Model.from_pretrained(model_name)
model.save_pretrained(save_directory)

# Download and save config (included in model save usually, but making sure)
config = Wav2Vec2Config.from_pretrained(model_name)
config.save_pretrained(save_directory)

# Download preprocessor config just in case (useful for reference)
try:
    processor = AutoProcessor.from_pretrained(model_name)
    processor.save_pretrained(save_directory)
except:
    print("No processor found or needed.")

print(f"Successfully saved to {os.path.abspath(save_directory)}")
