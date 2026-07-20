import time
import numpy as np
import onnxruntime as ort
from scipy.special import expit


class Inference:

    def __init__(self, config):
        self.config= config

        # TODO: add model path in config file
        self.model= self.load_onnx_model(self.config["model"]["path"])
        self.input_name= self.model.get_inputs()[0].name
    
    def load_onnx_model(self, model_path):
        
        # initialize inference session
        session= ort.InferenceSession(model_path, providers= ["CUDAExecutionProvider", "CPUExecutionProvider"])

        print("Providers:", session.get_providers())
        print(session.get_provider_options())
        print("Model Loaded Successfully!")
        
        return session
    
    def normalize(self, mri_volume):

        mri_volume= mri_volume.astype(np.float32)

        # loop through each modality and normalize it separately
        for c in range(mri_volume.shape[0]):
            channel= mri_volume[c]

            # normalize only in foreground brain regions
            mask= channel > 0

            if mask.any():
                voxels= channel[mask]
                mean= voxels.mean()
                std= voxels.std()

                if std > 1e-6:
                    channel[mask]= (voxels - mean) / std
                else:
                    channel[mask]= 0.0
        
        return mri_volume
    
    def pad_volume(self, mri_volume):

        _, h, w, _ = mri_volume.shape
        target_h= self.config["model"]["height"]
        target_w= self.config["model"]["width"]

        if h > target_h or w > target_w:
            raise ValueError(f"Input size ({h}, {w}) exceeds model input size ({target_h}, {target_w})")

        pad_top= (target_h - h) // 2
        pad_bottom= target_h - h - pad_top
        pad_left= (target_w - w) // 2
        pad_right= target_w - w - pad_left

        mri_volume= np.pad(mri_volume,
                           ((0,0),
                           (pad_top, pad_bottom),
                           (pad_left, pad_right),
                           (0,0)), mode= "constant", constant_values= 0)
        
        return mri_volume, (pad_top, pad_bottom, pad_left, pad_right)


    def preprocess(self, mri_volume): 
        # normalize
        # pad the image to the model expected input 256x256

        mri_volume= self.normalize(mri_volume)
        mri_volume, pad_sizes= self.pad_volume(mri_volume)

        return mri_volume, pad_sizes
    
    def process(self, mri_volume):

        _, h, w, num_slices= mri_volume.shape
                
        num_classes= self.config["model"]["num_classes"]
        batch_size= self.config["model"]["batch_size"]

        # N,H,W,S -> N,S,H,W
        input_batch = np.transpose(mri_volume, (3,0,1,2)).astype(np.float32)
        
        # initialize logits
        logits = np.zeros((num_slices, num_classes, h, w),dtype=np.float32)

        for i in range(0, num_slices, batch_size):

            batch = input_batch[i:i+batch_size]

            outputs = self.model.run(None,{self.input_name: batch})[0]  # N,Class,H,W

            logits[i:i+len(outputs)] = outputs
        
        return logits
    

    
    def postprocess(self, logits, pad_sizes):

        pad_tp, pad_bt, pad_lft, pad_rt= pad_sizes
        logits = logits[:,:,pad_tp:logits.shape[2]-pad_bt, pad_lft:logits.shape[3]-pad_rt]
            
        pred_mask= (expit(logits) > 0.5).astype(np.uint8)

        wt_volume= np.sum(pred_mask[:, 0]) / 1000.0
        tc_volume= np.sum(pred_mask[:, 1]) / 1000.0
        et_volume= np.sum(pred_mask[:, 2]) / 1000.0

        metrics= {"wt_volume_cc": wt_volume,
                  "tc_volume_cc": tc_volume,
                  "et_volume_cc": et_volume}

        return pred_mask, metrics


    def run_inference(self, mri_volume):
        
        # preprocess
        start_pre = time.perf_counter()
        mri_volume, pad_sizes= self.preprocess(mri_volume)
        end_pre = time.perf_counter()
        
        start_inf= time.perf_counter()
        raw_logits= self.process(mri_volume)
        end_inf= time.perf_counter()
        
        start_po = time.perf_counter()
        pred_mask, metrics= self.postprocess(raw_logits, pad_sizes)
        end_po = time.perf_counter()

        result= {}

        result["latency"]= {"preprocessing_time": end_pre-start_pre,
                            "inference_time": end_inf-start_inf,
                            "postprocessing_time": end_po-start_po}
        result["metrics"]= metrics


        return pred_mask, result



# if __name__ == "__main__":

#     config= {"model":{"path": "model_0.9153.onnx",
#                       "height": 256,
#                       "width": 256,
#                       "num_classes": 3,
#                       "batch_size": 16}}
    
#     pipeline= Inference(config=config)

#     valid_image_file= "../../dataset/Processed/validation/BraTS2021_00002.npz"
#     with np.load(valid_image_file) as data:
#         mri_volume= data["image"]
    
#     pred_mask, latency_dict= pipeline.run_inference(mri_volume=mri_volume)

#     print(pred_mask.shape)
#     print(latency_dict)

#     exit(0)

    