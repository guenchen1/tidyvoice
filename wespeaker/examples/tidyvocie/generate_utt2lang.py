import sys
import os

def generate_utt2lang(wav_scp_path, output_path):
    print(f"Reading {wav_scp_path}...")
    with open(wav_scp_path, 'r') as f:
        lines = f.readlines()

    unique_langs = set()
    utt2lang = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        
        utt_id = parts[0]
        wav_path = parts[1]
        
        # Path format: .../id010001/cy/cy_30308905.wav
        # Extract language code (2nd to last directory)
        path_parts = wav_path.split('/')
        if len(path_parts) >= 2:
            lang_code = path_parts[-2]
            utt2lang.append((utt_id, lang_code))
            unique_langs.add(lang_code)

    # Sort languages to ensure deterministic ID mapping
    sorted_langs = sorted(list(unique_langs))
    lang2id = {lang: i for i, lang in enumerate(sorted_langs)}
    
    print(f"Found {len(sorted_langs)} unique languages.")
    print(f"Writing utt2lang to {output_path}...")
    
    with open(output_path, 'w') as f:
        for utt_id, lang_code in utt2lang:
            lang_id = lang2id[lang_code]
            f.write(f"{utt_id} {lang_id}\n")
            
    # Also save lang2id map for reference
    map_path = output_path + ".map"
    with open(map_path, 'w') as f:
        for lang, i in lang2id.items():
            f.write(f"{lang} {i}\n")
            
    print(f"Done! Map saved to {map_path}")

if __name__ == "__main__":
    wav_scp = "data/tidyvoice_train/wav.scp"
    output = "data/tidyvoice_train/utt2lang"
    generate_utt2lang(wav_scp, output)
