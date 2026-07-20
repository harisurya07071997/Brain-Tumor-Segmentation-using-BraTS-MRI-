# pip install awscli gpustat s5cmd onnxruntime-gpu onnx scipy Medpy
# aws configure
# aws s3 cp s3://your-bucket-name/path/to/brats.tar - | tar -xvf - Processed/validation/


import json
import time
import numpy as np
from pathlib import Path
import onnxruntime as ort
from scipy.special import expit
from medpy.metric.binary import dc, hd95



# -----------------------------
# Load ONNX model on GPU
# -----------------------------

def load_onnx_model(model_path):

    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=[
            "CUDAExecutionProvider"
        ]
    )

    print("Providers:", session.get_providers())
    print("Provider options:")
    print(session.get_provider_options())

    return session



def normalize_volume(img):

    img= img.astype(np.float32)

    for c in range(img.shape[0]):
        channel= img[c]
        mask= channel > 0

        if np.any(mask):
            voxels= channel[mask]
            
            mean= voxels.mean()
            std= voxels.std()

            if std > 1e-6:
                channel[mask] = (voxels - mean) / std
            else:
                channel[mask] = 0.0

    return img


# -----------------------------
# GPU Inference
# -----------------------------

def run_inference(model,input_name,volume,batch_size=64):

    _, h, w, slices = volume.shape

    # preprocessing
    start_pre = time.perf_counter()
    volume = normalize_volume(volume)

    volume = np.pad(
        volume,((0,0),(8,8),(8,8),(0,0)),
        mode="constant"
    )

    # N,H,W,C -> N,C,H,W
    input_batch = np.transpose(volume,(3,0,1,2)).astype(np.float32)

    preprocessing_time = (time.perf_counter()-start_pre)

    logits = np.zeros((slices,3,h,w),dtype=np.float32)

    # -----------------------------
    # GPU inference only
    # -----------------------------
    start_inf = time.perf_counter()
    for i in range(0, slices, batch_size):

        batch = input_batch[i:i+batch_size]

        outputs = model.run(None,{input_name: batch})[0]

        outputs = outputs[:,:,8:-8,8:-8]

        logits[i:i+len(outputs)] = outputs

    inference_time = (time.perf_counter()-start_inf)

    pred = (expit(logits) > 0.5).astype(np.uint8)

    return (pred,preprocessing_time,inference_time)



def evaluate(gt_mask, pred_mask):

    pred_mask= np.transpose(pred_mask, (1,2,3,0))

    # create ground-truth binary mask for 3 sub regions
    gt_wt= np.isin(gt_mask, [1,2,3]).astype(np.uint8)  # whole tumor
    gt_tc= np.isin(gt_mask, [1,3]).astype(np.uint8)    # tumor core
    gt_et= (gt_mask == 3).astype(np.uint8)

    # extract predictions for each channel
    pred_wt= pred_mask[0, :, :, :]
    pred_tc= pred_mask[1, :, :, :]
    pred_et= pred_mask[2, :, :, :]
    
    regions= {"WT": (pred_wt , gt_wt),
              "TC":  (pred_tc, gt_tc),
              "ET": (pred_et, gt_et)}
    
    results= {}

    for name , (pred, gt) in regions.items():
        dice_score= dc(pred, gt)

        # calculate hd95, if tumor is presented in the image
        if np.sum(pred) > 0 and np.sum(gt) > 0:
            hd= hd95(pred, gt)
           
        else:
            hd= np.nan

        results[f"{name}_dice"] = dice_score
        results[f"{name}_hd95"] = float(hd)

        print(f"{name} -> Dice: {dice_score:.4f} | "
              f"HD95: {hd:.2f}mm" if not np.isnan(hd) else "N/A")
    
    return results



def evaluate_all_patient_records(npz_files):

    all_results= []

    for file in npz_files:

        with np.load(file) as data:
            mri_volume= data["image"]
            gt_mask= data["mask"]
        
        # run inference
        pred_mask, preprocessing_time, inference_time= run_inference(model, INPUT_NAME, mri_volume, batch_size=BATCH_SIZE)

        patient_results= evaluate(gt_mask=gt_mask, pred_mask=pred_mask)
        patient_results["patient_id"]= file.stem
        patient_results["model_latency_ms"]= inference_time * 1000

        all_results.append(patient_results)
    
    return all_results



if __name__ == "__main__":

    ONNX_MODEL_PATH= "./model/model_0.9153.onnx"
    VALIDATION_DIR_PATH= Path("../dataset/Processed/validation/")
    SAVE_JSON_PATH= "./pred.json"
    BATCH_SIZE= 64

    # load ONNX model
    model= load_onnx_model(ONNX_MODEL_PATH)
    INPUT_NAME= model.get_inputs()[0].name  

    # WARMUP
    dummy = np.zeros((BATCH_SIZE, 4, 256, 256), dtype=np.float32)
    for _ in range(20):
        model.run(None, {INPUT_NAME: dummy})

    # fetch all the validation files
    valid_imgs= list(VALIDATION_DIR_PATH.glob("*.npz"))

    # load .npz file
    all_results= evaluate_all_patient_records(valid_imgs)

    # Save to a file
    with open(SAVE_JSON_PATH, "w") as file:
        json.dump(all_results, file, indent=4)

    exit(0)

