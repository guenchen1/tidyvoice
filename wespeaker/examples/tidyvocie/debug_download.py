#!/usr/bin/env python3
import os
import sys
from datacollective import DataCollective

API_KEY = "f7dc18d692bde66e60b6a5e450c271cb1b40cbb38f7e09e950b8f776d5f1ce08"
OUTPUT_DIR = "/home/brant/Project/tidyVoice/wespeaker/examples/tidyvocie/evadata"
DATASET_ID = "cmkv32i5e02tumg07j79d3c35"

def debug_main():
    print(f"Target Directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Try setting env vars explicitly
    os.environ["MDC_API_KEY"] = API_KEY
    os.environ["MDC_DOWNLOAD_PATH"] = OUTPUT_DIR
    
    try:
        client = DataCollective()
        print("Fetching dataset metadata...")
        dataset = client.get_dataset(DATASET_ID)
        print("Download call finished.")
        
        items = os.listdir(OUTPUT_DIR)
        print(f"Items in {OUTPUT_DIR}: {items}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_main()
