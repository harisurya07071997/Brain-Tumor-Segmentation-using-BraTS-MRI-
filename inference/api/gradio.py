import yaml
import argparse
import numpy as np
import gradio as gr
from inference import Inference




def predict(file):

    with np.load(file) as data:
        mri_volume= data["image"]

    pred_mask, metrics= pipeline.run_inference(mri_volume)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path" , type = str, default = "./Config.yaml")
    args= parser.parse_args()

    with open(args.config_path) as yaml_file: config= yaml.safe_load(yaml_file)
    pipeline= Inference(config)

    with gr.Blocks() as iface:
        file_input= gr.File(label="Upload BraTS MRI (.npz)",
                            file_types=[".npz"])
        
