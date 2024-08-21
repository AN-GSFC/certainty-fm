import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import rasterio
from skimage import exposure
from scipy import stats
import io
from skimage import filters, morphology, segmentation
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
import matplotlib.cm as cm

# Set page configuration
st.set_page_config(layout="wide")

NO_DATA = -9999
NO_DATA_FLOAT = float(NO_DATA)
PERCENTILES = (1, 99)
model = None  # Global model variable
strict_threshold = 0.15  # Global variable for strict threshold
free_threshold = 0.01  # Global variable for free threshold

def load_model(device):
    """Load the model and set it as a global variable."""
    global model
    model = torch.load('original_model/model.pth', map_location=torch.device(device))
    model.eval()

def stretch_rgb(rgb):
    ls_pct = 1
    pLow, pHigh = np.percentile(rgb[~np.isnan(rgb)], (ls_pct, 100 - ls_pct))
    img_rescale = exposure.rescale_intensity(rgb, in_range=(pLow, pHigh))
    return img_rescale

def open_tiff(file):
    with rasterio.open(file) as src:
        data = src.read()
    return data

def get_meta(file):
    with rasterio.open(file) as src:
        meta = src.meta
    return meta

def load_raster(path, crop=None):
    with rasterio.open(path) as src:
        img = src.read()
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
    """Run inference on the Prithvi model with any given images."""
    device = "cuda" if gpu else "cpu"

    # Ensure the model is loaded

    model = torch.load('original_model/model.pth', map_location=torch.device(device))
    copy_test_pipeline = process_test_pipeline(model.cfg.data.test.pipeline)

    result = inference_segmentor(model, image_path, custom_test_pipeline=copy_test_pipeline)

    # Remove image artifacts
    mask = open_tiff(image_path)
    rgb = stretch_rgb((mask[[3, 2, 1], :, :].transpose((1, 2, 0)) / 10000 * 255).astype(np.uint8))
    meta = get_meta(image_path)
    mask = np.where(mask == meta['nodata'], 1, 0)
    mask = np.max(mask, axis=0)[None]

    result = np.where(mask == 1, 0, result * 255)
    return result

def enable_dropout(model, drop):
    """Enable dropout layers during test-time."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            m.p = drop

def get_monte_carlo_predictions(forward_passes, image_path, gpu):
    """Perform inference with Monte Carlo dropout."""
    device = "cuda" if gpu else "cpu"
    

    predictions = []

    for _ in range(forward_passes):
        model = torch.load('original_model/model.pth', map_location=torch.device(device))
        enable_dropout(model, drop=0.5)
        copy_test_pipeline = process_test_pipeline(model.cfg.data.test.pipeline)
        result = inference_segmentor(model, image_path, custom_test_pipeline=copy_test_pipeline)

        mask = open_tiff(image_path)
        rgb = stretch_rgb((mask[[3, 2, 1], :, :].transpose((1, 2, 0)) / 10000 * 255).astype(np.uint8))
        meta = get_meta(image_path)
        mask = np.where(mask == meta['nodata'], 1, 0)
        mask = np.max(mask, axis=0)[None]

        result = np.where(mask == 1, 0, result * 255)
        predictions.append(result)

    return predictions

def heatmap(preds):
    """Return stacked variance, mean prediction, and mode prediction arrays."""
    stacked_arrays = np.stack([arr[0] for arr in preds], axis=0)
    variance_array = np.var(stacked_arrays, axis=0)
    certainty_estimate = np.mean(stacked_arrays == 1, axis=0, dtype=np.float64)
    mode_array = stats.mode(stacked_arrays, axis=0)[0]
    return certainty_estimate, variance_array, mode_array

def run_inference(image_file, gpu=False, mc=3):
    """Run inference and return results."""
    temp_image_path = 'temp_image.tif'
    with open(temp_image_path, 'wb') as f:
        f.write(image_file.read())

    # Run inference for original image
    original_image = enhance_raster_for_visualization(load_raster(temp_image_path))
    preds = get_monte_carlo_predictions(mc, temp_image_path, gpu=gpu)
    certainty_arr, variance_arr, mode_arr = heatmap(preds)
    orig = reg_inference(temp_image_path, gpu=gpu)
    
    return original_image, orig[0], certainty_arr, variance_arr, mode_arr

def colormap_image(image, cmap_name='viridis'):
    """Apply a colormap to an image and return it."""
    normed_img = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize image
    colormap = cm.get_cmap(cmap_name)
    colored_img = colormap(normed_img)  # Apply colormap
    return (colored_img * 255).astype(np.uint8)  # Convert to 8-bit image

def display_results(original_image, orig, certainty_arr, variance_arr, mode_arr):
    """Display the results using Streamlit."""
    orig_colored = colormap_image(orig)
    certainty_colored = colormap_image(certainty_arr)
    variance_colored = colormap_image(variance_arr)

    # Create columns for the other results
    col1, col2, col3, col4 = st.columns(4)
    with col1:
         st.image(original_image, caption='Original Image', use_column_width=True)
    with col2:
        st.image(orig_colored, caption='Flooding Prediction (bright pixels are flooding)', use_column_width=True)
    with col3:
        st.image(certainty_colored, caption=f'Average Prediction ({mc_trials} trials, brighter is more likely to be flooding)', use_column_width=True)
    with col4:
        st.image(variance_colored, caption='Uncertainty Estimate (brighter pixels have higher uncertainty)', use_column_width=True)


def display_special_results(orig, variance_arr):
    global strict_threshold, free_threshold
    
    # Compute the strict and free predictions
    strict_prediction = np.where(variance_arr > strict_threshold, 0, 1).reshape(512, 512) * orig.reshape(512, 512)
    free_prediction = np.where(variance_arr > free_threshold, 1, 0).reshape(512, 512)
    free_prediction = orig.reshape(512, 512) + free_prediction
    free_prediction[free_prediction > 1] = 1

    # Apply colormap
    strict_colored = colormap_image(strict_prediction)
    free_colored = colormap_image(free_prediction)

    # Create columns for the other results
    col1, col2 = st.columns(2)
    with col1:
        st.image(strict_colored, caption='Strict Prediction (a more precise prediction for flood pixels, creating by assigning pixels to be a non-flood pixels if they do not meet a specified threshold)', use_column_width=True)
    with col2:
        st.image(free_colored, caption='Free Prediction (a prediction that is more likely to cover the entire range of flood pixels, created by assigning pixels to be a floods pixel if they meet a specified threshold)', use_column_width=True)

# Display the title
st.title("Prithvi-100M Flood Segmentation and Uncertainty Quantification")

# Sidebar for model settings
st.sidebar.title("Model Settings")

gpu_available = torch.cuda.is_available()

# Set the checkbox state based on GPU availability
gpu = st.sidebar.checkbox("Use GPU", value=gpu_available)

mc_trials = st.sidebar.slider(
    'Number of Monte Carlo Trials',
    min_value=1,
    max_value=50,
    value=10,
    help='Higher values converge to better uncertainty estimates but take longer to run.'
)
strict_threshold = st.sidebar.slider(
    'Strict Threshold',
    min_value=0.0,
    max_value=1.0,
    value=0.15,
    help='Pixels with variance above this threshold will not be considered flooding in strict prediction.'
)
free_threshold = st.sidebar.slider(
    'Free Threshold',
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    help='Pixels with variance above this threshold will be considered flooding in free prediction.'
)

# File uploader (not in the sidebar)
uploaded_file = st.file_uploader("Choose a TIFF image", type=["tif"])

if gpu_available:
    device='cuda'
else:
    device='cpu'

if not os.path.exists("original_model/model.pth"):
    with st.spinner("Please wait... downloading model..."):
        config_path = hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M-sen1floods11", filename="sen1floods11_Prithvi_100M.py")
        ckpt = hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M-sen1floods11", filename='sen1floods11_Prithvi_100M.pth')
        finetuned_model = init_segmentor(Config.fromfile(config_path), ckpt, device=device)

        if not os.path.exists("original_model/model.pth"):
            os.makedirs("original_model", exist_ok=True) 
            torch.save(finetuned_model, "original_model/model.pth")
            st.success('Prithvi-100M found and initialized for inference', icon="✅")
else:
    st.success('Prithvi-100M found and initialized for inference', icon="✅")

# Run inference when the button is clicked
if uploaded_file and st.button("Run Inference"):
    with st.spinner("Running inference..."):
        original_image, orig, certainty_arr, variance_arr, mode_arr = run_inference(uploaded_file, gpu=gpu, mc=mc_trials)
    display_results(original_image, orig, certainty_arr, variance_arr, mode_arr)
    display_special_results(orig, variance_arr)