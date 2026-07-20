import torch
import numpy as np
import onnxruntime as ort
import segmentation_models_pytorch as smp




def save_onnx_model(ckpt_file, onnx_save_path):
    """convert ckpt -> onnx file"""

    model = smp.UnetPlusPlus(
                encoder_name="resnet34",
                encoder_weights=None,       # No ImageNet weights since in_channels=4
                in_channels= IN_CHANNELS,    # 4 MRI modalities (FLAIR, T1, T1ce, T2)
                classes= NUM_CLASSES        
    )

    # load state_dict
    checkpoint= torch.load(ckpt_file, map_location= "cpu")["state_dict"]
    state_dict= {k.replace("model.", "") : v for k , v in checkpoint.items()}
    model.load_state_dict(state_dict)
    model.eval() 

    # export .onnx model
    _model_id= ckpt_file.split("=")[-1].replace(".ckpt", "")
    dummy_input= torch.randn(1, IN_CHANNELS, IMG_SZ, IMG_SZ)

    torch.onnx.export(model,
                    dummy_input,
                    f"{onnx_save_path}/model_{_model_id}.onnx",
                    export_params= True,
                    opset_version= 18,
                    do_constant_folding= True,
                    dynamo=False,
                    dynamic_axes= {
                            'input': {0: 'batch_size'},    # 0th dimension is dynamic
                            'output': {0: 'batch_size'}},
                    input_names= ['input'],
                    output_names= ['output'])

    print(f"Onnx model for {ckpt_file} is exported!!", end = "\n\n")


if __name__ == "__main__":

    IN_CHANNELS= 4
    NUM_CLASSES= 3
    IMG_SZ= 256
    ckpt_file= "./model/epoch=24-val_mean_dice=0.9153.ckpt"
    onnx_save_path= "./model"
    save_onnx_model(ckpt_file, onnx_save_path)
    exit(0)

    