# pip install awscli gpustat s5cmd pytorch_lightning albumentations wandb segmentation-models-pytorch 
# aws configure
# wandb login <>
# s5cmd cp ""
# rm -rf ~/.aws


import os
import yaml
import json
import torch
import wandb
import random
import argparse
import numpy as np
from pathlib import Path
import albumentations as A
import torch.optim as optim
from functools import partial
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.profilers import AdvancedProfiler
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import MultilabelAccuracy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping



def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


#  ************************** #
# TODO: Should consider cropping the mri volume in the next training phase to improve the training speed and performance

# crop the empty space in the mri volume to reduce the latency of the model
def crop_mri_volume(mri_volume):

    # merge all 4 modalities
    volume_3d= np.max(mri_volume, axis= 0)

    # create a binary mask where brain tissue exists
    binary_mask= volume_3d > 0

    if not binary_mask.any():
        raise ValueError("Empty MRI volume.")

    # fetch the indices of each axis in the binary mask
    axis_h= np.any(binary_mask, axis= (1,2))
    axis_w= np.any(binary_mask, axis= (0,2))
    axis_s= np.any(binary_mask, axis= (0,1))

    # fetch min and max index of each axis
    h_min, h_max= np.where(axis_h)[0][[0,-1]] 
    w_min, w_max= np.where(axis_w)[0][[0,-1]]
    s_min, s_max= np.where(axis_s)[0][[0,-1]]

    # add a small padding safety margin for the boundaries of brain tissue
    padding= 2
    h_min= max(0, h_min-padding)
    w_min= max(0, w_min-padding)
    s_min= max(0, s_min-padding)
    
    h_max= min(mri_volume.shape[1], h_max+padding+1)
    w_max= min(mri_volume.shape[2], w_max+padding+1)
    s_max= min(mri_volume.shape[3], s_max+padding+1)

    # crop the mri volumme based on the bounding box coordinates of foreground brain pixels
    cropped_mri_volume= mri_volume[:, h_min:h_max, w_min:w_max, s_min:s_max]
    
    return cropped_mri_volume

# *************************** #

class Normalize:

    def __call__(self, image):
        """
        image: numpy array of shape (H, W, C)
        """

        image = image.astype(np.float32)

        # Assuming HWC
        for c in range(image.shape[-1]):
            channel = image[..., c]

            mask = channel > 0

            if np.any(mask):
                mean = channel[mask].mean()
                std = channel[mask].std()

                if std > 1e-6:
                    channel[mask] = (channel[mask] - mean) / std
                else:
                    channel[mask] = 0

            image[..., c] = channel

        return image



# create a Dataset class for BraTS Segmentation
# TODO: normalize the dataset
class BrainTumorSegmentationDataset(Dataset):
    def __init__(self, root_dir, metadata, split, transform):
        
        self.root_dir= root_dir
        self.metadata= metadata
        self.transform= transform

        self.samples= self.metadata[split]

        self.crop= crop_mri_volume()
        self.preprocess = Normalize()

    def __len__(self):
        return len(self.samples)
    
    # def crop_image(self, im_arr, mask_arr):
    #     pass
    
    def __getitem__(self, idx):
        record = self.samples[idx]
        record_file = record["file"].replace("\\", "/")
        img_path = Path(self.root_dir) / record_file

        slice_id = record["slice_id"]

        # 1. Load data from the NPZ file
        with np.load(img_path) as data:
            image = data["image"][:, :, :, slice_id]  # Expected shape format: [C, H, W]
            mask = data["mask"][:, :, slice_id]       # Expected shape format: [H, W]

        # 2. Reshape (C, H, W) -> (H, W, C) for Albumentations compatibility
        image = np.transpose(image, (1, 2, 0))

        # normalize
        image= self.preprocess(image)

        # 3. Apply standard augmentations (flips, rotations, crops, etc.)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # 4. MULTI-LABEL CONVERSION: Map categorical indices to overlapping channels
        # Expected Albumentations output format for mask is [H, W]
        h, w = mask.shape
        target_mask = np.zeros((3, h, w), dtype=np.float32)

        # Class mappings: 1 = necrotic_tumor, 2 = edema, 3 = enhancing_tumor
        target_mask[0] = np.isin(mask, [1, 2, 3])  # Channel 0: Whole Tumor (WT)
        target_mask[1] = np.isin(mask, [1, 3])     # Channel 1: Tumor Core (TC)
        target_mask[2] = (mask == 3)                # Channel 2: Enhancing Tumor (ET)

        # 5. Format outputs as PyTorch tensors
        # Ensure image tensor uses PyTorch standard layout: [C, H, W]
        if isinstance(image, torch.Tensor):
            # If your Albumentations transform contains ToTensorV2():
            image_tensor = image
        else:
            # Manual fallback if transform returns raw NumPy arrays:
            image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()

        target_tensor = torch.from_numpy(target_mask).float()

        return image_tensor, target_tensor




class PLDataset(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config= config

        self.train_transform = A.Compose(
            [
                # spatial transforms (image + mask)
                A.HorizontalFlip(p= self.config["dataset"]["augmentations"]["horizontal_flip"]["prob"]),
                A.Rotate(
                    limit= self.config["dataset"]["augmentations"]["rotate"]["limit"],
                    p= self.config["dataset"]["augmentations"]["rotate"]["prob"]),
                A.Affine(
                    scale= self.config["dataset"]["augmentations"]["affine"]["scale"],
                    translate_percent= self.config["dataset"]["augmentations"]["affine"]["translate_percent"],
                    p= self.config["dataset"]["augmentations"]["affine"]["prob"]),
                A.RandomBrightnessContrast(
                    brightness_limit= self.config["dataset"]["augmentations"]["brightness"]["brightness_limit"],
                    contrast_limit= self.config["dataset"]["augmentations"]["brightness"]["contrast_limit"],
                    p= self.config["dataset"]["augmentations"]["brightness"]["prob"]),
                A.PadIfNeeded(
                    min_height= self.config["dataset"]["augmentations"]["resize"]["tgt_img_sz"], 
                    min_width= self.config["dataset"]["augmentations"]["resize"]["tgt_img_sz"], 
                    border_mode=0, # 0 means constant padding (black borders)
                    value=0,       # Value for image background padding
                    mask_value=0),   # Value for mask background padding

                ToTensorV2()
            ]
        )

        self.val_transform= A.Compose([
                A.PadIfNeeded(
                    min_height= self.config["dataset"]["augmentations"]["resize"]["tgt_img_sz"], 
                    min_width= self.config["dataset"]["augmentations"]["resize"]["tgt_img_sz"], 
                    border_mode=0, # 0 means constant padding (black borders)
                    value=0,       # Value for image background padding
                    mask_value=0   # Value for mask background padding
                ),
                
                ToTensorV2()])


    def prepare_data(self):
        pass

    def setup(self, stage= None):

        with open(self.config["dataset"]["metadata_dir"]) as f:
            slice_metadata= json.load(f)
        
        if stage in ["fit", "train"] or not stage:
            self.trn_dataset= BrainTumorSegmentationDataset(root_dir= self.config["dataset"]["inp_root_dir"], metadata= slice_metadata, split= "train", transform= self.train_transform)
            print(f"Train dataset has been created : Size - {len(self.trn_dataset)}")

            self.val_dataset= BrainTumorSegmentationDataset(root_dir= self.config["dataset"]["inp_root_dir"], metadata= slice_metadata, split= "validation", transform= self.val_transform)
            print(f"Validation dataset has been created : Size - {len(self.val_dataset)}")
        
        elif stage == "test":
            self.tst_dataset= BrainTumorSegmentationDataset(root_dir= self.config["dataset"]["inp_root_dir"], metadata= slice_metadata,split= "validation", transform= self.val_transform)
            print(f"Test dataset has been created : Size - {len(self.tst_dataset)}")

    def train_dataloader(self):

        dataloader= DataLoader(dataset= self.trn_dataset,
                               shuffle= True,
                               drop_last=True,
                               batch_size= self.config["dataset"]["dataloader"]["trn_batch_sz"],
                               num_workers= self.config["dataset"]["dataloader"]["num_workers"],
                               pin_memory= self.config["dataset"]["dataloader"]["pin_memory"],
                               prefetch_factor= self.config["dataset"]["dataloader"]["prefetch_factor"],
                               persistent_workers= self.config["dataset"]["dataloader"]["persistent_workers"])
        
        return dataloader
    
    def val_dataloader(self):

        dataloader= DataLoader(dataset= self.val_dataset,
                               batch_size= self.config["dataset"]["dataloader"]["val_batch_sz"],
                               num_workers= self.config["dataset"]["dataloader"]["num_workers"],
                               pin_memory= self.config["dataset"]["dataloader"]["pin_memory"],
                               prefetch_factor= self.config["dataset"]["dataloader"]["prefetch_factor"],
                               persistent_workers= self.config["dataset"]["dataloader"]["persistent_workers"]) 
        
        return dataloader
    
    def test_dataloader(self):

        dataloader= DataLoader(dataset= self.val_dataset,
                               batch_size= self.config["dataset"]["dataloader"]["val_batch_sz"],
                               num_workers= self.config["dataset"]["dataloader"]["num_workers"],
                               pin_memory= self.config["dataset"]["dataloader"]["pin_memory"],
                               prefetch_factor= self.config["dataset"]["dataloader"]["prefetch_factor"],
                               persistent_workers= self.config["dataset"]["dataloader"]["persistent_workers"]) 
        
        return dataloader
        


class PLModel(pl.LightningModule):
    
    def __init__(self,
                 config: dict,
                 args: argparse.ArgumentParser):
        super().__init__()
        self.args = args
        self.config = config
        
        self.metric_name= self.config["model"]["training"]["val_metric"]
        self.label2id = self.config["dataset"]["dataloader"]["label2id"]
        self.id2label = self.config["dataset"]["dataloader"]["id2label"]
        num_classes= len(self.label2id) - 1   # exclude bg and include 3 Hierarchical regions (WT, TC, ET)  

        # Unet++ configuration for 2D BraTS
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,       # No ImageNet weights since in_channels=4
            in_channels=self.config["model"]["training"]["in_channels"],    # 4 MRI modalities (FLAIR, T1, T1ce, T2)
            classes=  num_classes        
        )

        self.print_trainable_parameters()

        # Multi-label loss (allows overlapping channels for WT, TC, ET)
        self.dice_loss_fn = smp.losses.DiceLoss(mode=smp.losses.MULTILABEL_MODE, from_logits=True)
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss()

        # TODO: make these inputs configurable
        self.val_dice= DiceScore(num_classes= num_classes, 
                                 include_background= True,
                                 average= None,
                                 aggregation_level= "global",
                                 input_format= "one-hot")
        
        self.val_iou= MeanIoU(num_classes= num_classes, 
                              include_background= True,
                              per_class= True,
                              input_format= "one-hot")

        self.val_accuracy= MultilabelAccuracy(num_labels= num_classes,
                                              average= None,
                                              multidim_average= "global") 

    
    def print_trainable_parameters(self):
        
        all_params = 0
        trainable_params = 0
        for _ , param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad: trainable_params += param.numel()
                
        print(f"Trainable params: {trainable_params} || all params: {all_params}")
        print(f"Percentage of trainable params: {round(100 * trainable_params / all_params, 2)}")
        
        return
    
    def configure_optimizers(self):

        lr = self.config["model"]["optimizer"]["lr"]
        factor = self.config["model"]["optimizer"]["factor"]
        patience = self.config["model"]["optimizer"]["patience"]
        threshold = self.config["model"]["optimizer"]["threshold"]
        weight_decay = self.config["model"]["optimizer"]["weight_decay"]
        optimizer = optim.AdamW(self.model.parameters(), lr = lr, weight_decay = weight_decay)

        lr_scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = "min", factor = factor, patience = patience,
                                             threshold = threshold)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler,
                                 "monitor": "val_loss",
                                 "interval": "epoch",
                                 "frequency": 1}}
    
    def forward(self, x): 
        return self.model(x)
    
    def _shared_step(self, batch):
        images, labels= batch
        logits= self(images)

        # ensures BCE stabilizes training on empty/healthy tissue, while Dice prevents the tumor boundaries from being erased by the background bias.
        dice_weight= self.config["model"]["optimizer"]["dice_weight"]
        loss= dice_weight * self.dice_loss_fn(logits, labels.float()) + (1 - dice_weight) * self.bce_loss_fn(logits, labels.float())

        return loss, logits, labels
    
    

    def training_step(self, batch, batch_idx, dataloader_idx= 0):
        loss, _, _= self._shared_step(batch)
        logging_metrics= {"loss": loss}

        self.log_dict(logging_metrics, on_step = True, on_epoch = True, logger = True, prog_bar = True)
        return logging_metrics
    


    def validation_step(self, batch, batch_idx, dataloader_idx= 0):
        loss, logits, labels = self._shared_step(batch)

        # Dynamic thresholding to create binary mask inputs for the metric
        preds = (torch.sigmoid(logits) > 0.5).long()
        
        # Accumulate raw inputs (Shape: [B, 3, H, W])
        self.val_dice(preds, labels.long())
        self.val_iou(preds, labels.long())
        self.val_accuracy(preds, labels.long())

        logging_dict = {"val_loss": loss}
        self.log_dict(logging_dict, on_epoch = True,  prog_bar=True, sync_dist = True)
        
        return logging_dict
    


    def on_validation_epoch_end(self):

        dice_metric = self.val_dice.compute()
        iou_metric= self.val_iou.compute()
        acc_metric= self.val_accuracy.compute()
        
        logging_metrics = {f"val_{self.metric_name}": dice_metric.mean(),
                           f"val_mean_iou": iou_metric.mean()}
        logging_metrics.update({f"val_dice_{self.id2label[i+1]}": v for i, v in enumerate(dice_metric)})
        logging_metrics.update({f"val_iou_{self.id2label[i+1]}": v for i, v in enumerate(iou_metric)})
        logging_metrics.update({f"val_acc_{self.id2label[i+1]}": v for i, v in enumerate(acc_metric)})

        self.log_dict(logging_metrics, on_epoch = True,  prog_bar=True, sync_dist = True)
        # reset engine states for the upcoming epoch
        self.val_dice.reset()
        self.val_iou.reset()
        self.val_accuracy.reset()
        
        return




def main(args: argparse.ArgumentParser, config: dict):
    
    if args.do_tuning:
        wandb_run = wandb.init(project = config["model"]["wandb"]["project"])
        # TODO: update all the required parameters for tuning using wandb sweep
        config["model"]["optimizer"]["lr"] = wandb_run.config.get("lr", config["model"]["optimizer"]["lr"])
        config["model"]["optimizer"]["weight_decay"] = wandb_run.config.get("weight_decay", config["model"]["optimizer"]["weight_decay"])
        config["model"]["optimizer"]["dice_weight"] = wandb_run.config.get("dice_weight", config["model"]["optimizer"]["dice_weight"])
        config["model"]["optimizer"]["patience"] = wandb_run.config.get("lr_patience", config["model"]["optimizer"]["patience"])
        config["model"]["optimizer"]["gradient_clip_val"] = wandb_run.config.get("gradient_clip_val", config["model"]["optimizer"]["gradient_clip_val"])
        config["dataset"]["augmentations"]["affine"]["scale"] = wandb_run.config.get("scale", config["dataset"]["augmentations"]["affine"]["scale"])
        config["dataset"]["augmentations"]["rotate"]["limit"] = wandb_run.config.get("rotate", config["dataset"]["augmentations"]["rotate"]["limit"])
        print(f"Doing tuning for this configuration: {wandb_run.config}")
    
    pl_dataset = PLDataset(config = config)
    pl_dataset.prepare_data()
    pl_dataset.setup("train")

    wandb_logger = WandbLogger(project = config["model"]["wandb"]["project"], log_model = False)
    wandb_logger.experiment.config.update(config)
    
    profiler = AdvancedProfiler(dirpath  = config["model"]["profiler"]["dirpath" ],
                                filename = config["model"]["profiler"]["filename"],
                                line_count_restriction = config["model"]["profiler"]["line_count_restriction"])
    
    train_params = dict(logger = wandb_logger,
                        profiler = profiler,
                        devices = config["model"]["training"]["devices"],
                        strategy = config["model"]["training"]["strategy"],
                        precision = config["model"]["training"]["precision"],
                        accelerator = config["model"]["training"]["accelerator"],
                        deterministic = config["model"]["training"]["deterministic"],
                        max_epochs = config["model"]["optimizer"]["num_tuning_epochs"],
                        gradient_clip_val = config["model"]["optimizer"]["gradient_clip_val"],
                        limit_val_batches = config["model"]["training"]["limit_val_batches" ],
                        limit_train_batches = config["model"]["training"]["limit_train_batches"],
                        accumulate_grad_batches = config["model"]["optimizer"]["gradient_accumulation_steps"])
    
    train_params["callbacks"] = [LearningRateMonitor(logging_interval = 'epoch')]

    if not args.do_tuning: 
        metric_to_monitor = config["model"]["checkpoint_strategy"]["monitor"]
        checkpoint_callback = ModelCheckpoint(dirpath = config["dataset"]["out_root_dir"],
                                              monitor = metric_to_monitor,
                                              filename = f'{{epoch}}-{{{metric_to_monitor}:.4f}}',
                                              mode = config["model"]["checkpoint_strategy"]["mode"],
                                              save_last = config["model"]["checkpoint_strategy"]["save_last"],
                                              save_top_k = config["model"]["checkpoint_strategy"]["save_top_k"])
        
        early_stop_callback = EarlyStopping(monitor=config["model"]["checkpoint_strategy"]["monitor"],
                                            mode=config["model"]["checkpoint_strategy"]["mode"],
                                            patience=config["model"]["checkpoint_strategy"]["early_stopping"]["patience"]) # Number of validation checks to wait before stopping

        train_params["callbacks"] += [checkpoint_callback, early_stop_callback]
        train_params["max_epochs"] = config["model"]["optimizer"]["num_train_epochs"]
    
    pl_model = PLModel(config, args)
    trainer = pl.Trainer(**train_params)

    if args.do_resume_from_checkpoint:
        ckpt_path = config["model"]["training"]["ckpt_path"]
        if not ckpt_path or not os.path.exists(ckpt_path): print("Checkpoint path not specified or wrong path for checkpoint specified!")
        trainer.fit(pl_model, pl_dataset, ckpt_path = ckpt_path)
    else:
        trainer.fit(pl_model, pl_dataset)
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path" , type = str, default = "./Config.yaml")
    parser.add_argument("--do_training" , action = 'store_true', help = "Flag to trigger training")
    parser.add_argument("--do_resume_from_checkpoint" , action = 'store_true', help = "Flag to trigger start \
                        training from the checkpoint specified!")
    parser.add_argument("--do_tuning"   , action = 'store_true', help = "Flag to trigger hyperparameter tuning")
    
    args = parser.parse_args()
    if not args.do_tuning: args.do_training = True
    with open(args.config_path) as yaml_file: config = yaml.safe_load(yaml_file)
    set_seed(config["model"]["training"]["seed"])
    
    if   args.do_tuning:
        sweep_config = config["model"]["wandb"]["sweep_config"]
        sweep_id = wandb.sweep(sweep_config, project = config["model"]["wandb"]["project"])
        partial_main = partial(main, args, config)
        wandb.agent(sweep_id, partial_main)
    
    else:
        main(args, config)
