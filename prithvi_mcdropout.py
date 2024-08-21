"""
this python script applies montecarlo dropout to the prithvi foundation model's inference on the sen1floods11 flood segmentation dataset. 

it writes results of uncertainty quantification, along with ground truth/original prediction arrays into a directory called inference_results.

ARGS: 
--gpu (include this flag for inference with gpu. currently, multiple gpu is not supported.)
--stop (int) (include the flag with an integer n, to stop inference after image n)
--mc (int) (specify number of montecarlo dropout trials for certainty estimation. default is 3.)

e.g.
python prithvi_mcdropout.py --gpu 1 --stop 2 --mc 2
"""


from skimage import filters, morphology, segmentation, exposure
import argparse
import os
import socket
import torch
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from mmcv import Config
from mmseg.apis import init_segmentor
from model_inference import inference_segmentor, process_test_pipeline
from huggingface_hub import hf_hub_download
import random
from scipy import stats
import matplotlib
from PIL import Image
import sys
import imageio

def stretch_rgb(rgb):
    ls_pct=1
    pLow, pHigh = np.percentile(rgb[~np.isnan(rgb)], (ls_pct,100-ls_pct))
    img_rescale = exposure.rescale_intensity(rgb, in_range=(pLow,pHigh))
    
    return img_rescale

def open_tiff(fname):
    with rasterio.open(fname, "r") as src:
        
        data = src.read()
        
    return data
def get_meta(fname):
    with rasterio.open(fname, "r") as src:
        
        meta = src.meta
        
    return meta


def load_raster(path, crop=None):
    with rasterio.open(path) as src:
        img = src.read()
        # load first 6 bands
        img = img[:6]
        img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
        if crop:
            img = img[:, -crop[0]:, -crop[1]:]
    return img

def enhance_raster_for_visualization(raster, ref_img=None):
    if ref_img is None:
        ref_img = raster
    channels = []
    for channel in range(raster.shape[0]):
        valid_mask = np.ones_like(ref_img[channel], dtype=bool)
        valid_mask[ref_img[channel] == NO_DATA_FLOAT] = False
        mins, maxs = np.percentile(ref_img[channel][valid_mask], PERCENTILES)
        normalized_raster = (raster[channel] - mins) / (maxs - mins)
        normalized_raster[~valid_mask] = 0
        clipped = np.clip(normalized_raster, 0, 1)
        channels.append(clipped)
    clipped = np.stack(channels)
    channels_last = np.moveaxis(clipped, 0, -1)[..., :3]
    rgb = channels_last[..., ::-1]
    return rgb

def reg_inference(image_path, gpu):
    """function to run inference on the prithvi model with any given images. make sure orignal_model/model.pth exists. this is an original copy of the finetuned model"""
    
    device="cpu"
    if gpu:
        device="cuda"

    copy_model = torch.load('original_model/model.pth', map_location=torch.device(device))
    copy_model.eval()
    copy_test_pipeline = process_test_pipeline(copy_model.cfg.data.test.pipeline)

    
    result = inference_segmentor(copy_model, image_path, custom_test_pipeline=copy_test_pipeline)

    # remove image artifacts given in image metadata
    mask = open_tiff(image_path)
    rgb = stretch_rgb((mask[[3, 2, 1], :, :].transpose((1,2,0))/10000*255).astype(np.uint8))
    meta = get_meta(image_path)
    mask = np.where(mask == meta['nodata'], 1, 0)
    mask = np.max(mask, axis=0)[None]

    result = np.where(mask == 1, 0, result*255)
    return result



def enable_dropout(model, drop):
    """ function to enable the dropout layers during test-time. """

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
                m.train()
                m.p = drop
        

def get_monte_carlo_predictions(forward_passes, image_path, gpu):
    """ function to perform inference with monte carlo dropout. calls enable_dropout function and returns monte carlo dropout predictions."""

    device = "cpu"
    if gpu:
        device="cuda"
    # load the model from disk
    copy_model = torch.load('original_model/model.pth', map_location=torch.device(device))
    predictions = []

    # inference
    for _ in range(forward_passes):
         
        enable_dropout(copy_model, drop=0.5) # adjust the dropout rate for all layers here
        copy_test_pipeline = process_test_pipeline(copy_model.cfg.data.test.pipeline)
        result = inference_segmentor(copy_model, image_path, custom_test_pipeline=copy_test_pipeline)
      
        mask = open_tiff(image_path)
        rgb = stretch_rgb((mask[[3, 2, 1], :, :].transpose((1,2,0))/10000*255).astype(np.uint8))
        meta = get_meta(image_path)
        mask = np.where(mask == meta['nodata'], 1, 0)
        mask = np.max(mask, axis=0)[None]

        result = np.where(mask == 1, 0, result*255)
  
        predictions.append(result)

    return predictions
  


def heatmap(preds):
    """given n-arrays, this function returns the stacked variance of each position in the array, which is the uncertainty estimate. will also return the mean prediction array, and a            mode prediction array """

    stacked_arrays = np.stack([arr[0] for arr in preds], axis=0)
    variance_array = np.var(stacked_arrays, axis=0)
    certainty_estimate = np.mean(stacked_arrays == 1, axis=0, dtype=np.float64)
    mode_array = stats.mode(stacked_arrays, axis=0)[0]


    
    return certainty_estimate, variance_array, mode_array




if __name__ == "__main__":
    device="cpu"
    NO_DATA = -9999
    NO_DATA_FLOAT = 0.0001
    PERCENTILES = (0.1, 99.9)
    MULTIPLE_GPU = False
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='Use GPU or not')
    parser.add_argument('--stop', type=int, default=-1, help='Early stopping mechanism during inference')
    parser.add_argument('--mc', type=int, default=3, help="Certainty simulation count")

    args = parser.parse_args()


    # TODO: implement multiple gpu inference
    if args.gpu == 1:
        print("Using GPU")
        device="cuda"
    elif args.gpu>1:
        MULTIPLE_GPU = True
        print("Using multiple GPU")
        device="cuda"
    else:
        print("Not using GPU")

    if args.stop== -1:
        print("No early stopping.")
    elif args.stop>0:
        print(f"Stopping inference on image {args.stop}")
    else:
        print("Invalid stopping count. --stop must be >=1")
        sys.exit(1)

    if args.mc < 2:
        print("Invalid simulation count. --mc must be >=2")
        sys.exit(1)
    else:
        print(f"Simulation count == {args.mc}")



    directory = "test_labels"
    if os.path.exists(directory) == False:
        print(f"Error: The directory '{directory}' does not exist.", file=sys.stderr)
        print(f"Please create {directory} with labeled flood segmentation images inside (..._LabelHand.tif).", file=sys.stderr)
        sys.exit(1)


    # TO DO: add try and except for model download
    # download finetuned model weights and config from huggingface. comment this out after the first run.
    config_path=hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M-sen1floods11", filename="sen1floods11_Prithvi_100M.py")
    ckpt=hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M-sen1floods11", filename='sen1floods11_Prithvi_100M.pth')
    finetuned_model = init_segmentor(Config.fromfile(config_path), ckpt, device)
    
    if not os.path.exists("original_model/model.pth"):
        os.makedirs("original_model", exist_ok=True) 
        torch.save(finetuned_model, "original_model/model.pth")


    images={}
    
    # early stopping for testing
    stop_count = 0


    # iterate through a directory and use its images as inference, specify ground truth path inside the loop
    for filename in os.listdir(directory):
        if args.stop > 0 and stop_count == args.stop: 
            print("Early stop.")
            break
            
        if filename == ".ipynb_checkpoints":
            continue

        imgname = filename.split("_")
        imgname=f"{imgname[0]}_{imgname[1]}_S2Hand.tif"
        filepath = os.path.join(directory, filename)

        
        ground_truth_path = filepath
        image_path = f"test_images/{imgname}"
        
        if os.path.exists(image_path) == False:
            print(f"Error: The path '{image_path}' does not exist.", file=sys.stderr)
            print(f"Please a directory called test_images exists, with images for model inference. The images can be found in the the Sen1Floods11 dataset.", file=sys.stderr)
            print("It can be installed using: gsutil -m rsync -r gs://sen1floods11 .")
            sys.exit(1)

        
        print(f"Image: {image_path} \nLabel: {filepath}")

        # inference with Prithvi and certainty estimation + evaluation
        orig = reg_inference(image_path, args.gpu)
        mc_preds = get_monte_carlo_predictions(args.mc, image_path, args.gpu)
        certainty_arr, arr, mode_arr=heatmap(mc_preds)
        stop_count+=1


        # image saving
 
        base_dir = "inference_results" # name of the directory in which uncertainty estimates, predictions, ground truth will be saved to.

        os.makedirs(base_dir, exist_ok=True)
        image_folder = os.path.join(base_dir, os.path.splitext(os.path.basename(image_path))[0])
        os.makedirs(image_folder, exist_ok=True)

        # saving formats
        np.save(f'{image_folder}/original_pred.npy', orig[0])
        np.save(f'{image_folder}/groundtruth_arr.npy', load_raster(ground_truth_path)[0])
        np.save(f'{image_folder}/variance_arr{args.mc}.npy', arr)
        np.save(f'{image_folder}/certainty_arr{args.mc}.npy', certainty_arr)
  