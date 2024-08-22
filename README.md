# certainty-fm

**Empowering Trustworthy Image Segmentation with Uncertainty Quantification**

This repository accompanies the paper **"Towards Trustworthy and Reliable Image Segmentation Foundation Models"** (An Ngo, Michael Little) and provides an implementation of MC dropout uncertainty quantification for the Prithvi-100M foundation model's image segmentation task. Uncertainty estimates are then applied to binary flood predictions, enhancing the model's ability to provide nuanced and reliable results.

## Examples from the paper
![ex2](https://github.com/user-attachments/assets/f919091a-46be-4e3a-9797-8a463de361de)
![ex1](https://github.com/user-attachments/assets/68dcf1af-f60c-4a30-822e-e89257c9232a)
![ex3](https://github.com/user-attachments/assets/15bae8c0-8610-45c9-8bc1-c13925f3dadf)

## Inference Dashboard

We have also integrated our implementation into an interactive web dashboard. The dashboard can be used to perform flood segmentation on uploaded satellite imagery with Prithvi-100M, while also visualizing uncertainty quantification and supporting predictions.

**Dashboard UI:**
![image](https://github.com/user-attachments/assets/533ed5f4-d708-4ad4-9f3e-707ca377e9b0)

## Instructions 

### Environment Setup

1. **Create a conda environment:**
   ```bash
   conda create -n prithvi_uq python==3.9

2. **Activate the conda environment:**
   ```bash
    conda activate prithvi_uq

3. **Run the setup script to install dependencies:**
   ```bash
    bash setup.sh


### Dashboard Setup 

**Prerequisites:**

* **Environment Setup:** Complete the environment setup and ensure the conda environment is created and activated with all dependencies installed.

**Launch the Dashboard:**

1. **run Streamlit:**

   ```bash
   streamlit run prithvi_mcdropout_ui.py

2. **Navigate to the Dashboard:**

   * Open your web browser and head to the address where Streamlit is running (usually `localhost:8501`).

**Get Flood Predictions:**

1. **Upload an Image:**

   * Use the dashboard interface to upload an image you want to analyze.
   * You can find sample images in the `test_images` directory of this repository.

2. **Run Inference:**

   * Click the "Run Inference" button.

3. **View Results:**

   * The dashboard will display the flood prediction for your image, along with uncertainty quantification.

### CLI for Uncertainty Quantification 

**Prerequisites:**

* **Environment Setup:** Complete the environment setup and activate your conda environment with all dependencies installed.

```bash
python prithvi_mcdropout.py [ARGS]

ARGS: 
--gpu (number of gpus used. specify 0 for no gpus. multi-gpu training currently not working.)
--stop (int) (include the flag with an integer n, to stop inference after image n. for inference of all images in /test_images, specify this as -1)
--mc (int) (specify number of montecarlo dropout trials for certainty estimation. default is 3.)

e.g.
python prithvi_mcdropout.py --gpu 1 --stop 5 --mc 5
```
