# LST-GAN: Land Surface Temperature Spatiotemporal Trend Prediction using GANs

## Project Overview

This project implements a Generative Adversarial Network (GAN) to predict spatiotemporal trends in Land Surface Temperature (LST). The goal is to generate future LST maps based on historical data, capturing complex patterns and variations.

## Project Structure

- `config/`: Contains `config.py` for all model hyperparameters, paths, and training settings.
- `data/`: Placeholder for your LST dataset. Expected to contain image files (e.g., `.png`, `.jpg`, `.npy`) representing LST maps.
- `models/`: Stores trained Generator and Discriminator model checkpoints.
- `results/`: Stores generated LST images and other output visualizations during training and prediction.
- `src/`: Contains the core Python source code:
    - `data_loader.py`: Handles data loading and preprocessing (e.g., `LSTDataset`, `get_dataloader`).
    - `models.py`: Defines the Generator and Discriminator network architectures (DCGAN-based).
    - `trainer.py`: Encapsulates the GAN training loop and logging logic.
- `train.py`: The main script to start the training process.
- `predict.py`: A script to use a trained Generator model for making predictions.
- `requirements.txt`: Lists all necessary Python dependencies.
- `README.md`: This file, providing project information.

## Getting Started

### 1. Environment Setup

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data

Place your Land Surface Temperature (LST) data files (e.g., `.png`, `.jpg`, `.npy` images) into the `data/` directory.

**Important:** You might need to modify `src/data_loader.py` to correctly parse your specific LST data format and potentially adapt the `__getitem__` method if you are loading sequences of images for spatiotemporal prediction (e.g., loading `t` and `t+1` images). The current `data_loader.py` is set up to load single grayscale images.

### 4. Configure Training

Adjust the hyperparameters and paths in `config/config.py` as needed, e.g., `BATCH_SIZE`, `NUM_EPOCHS`, `IMAGE_SIZE`, `DEVICE`.

### 5. Train the Model

Run the main training script:

```bash
python train.py
```

During training, generated samples and model checkpoints will be saved in the `results/` and `models/` directories, respectively.

### 6. Make Predictions

After training, you can use the `predict.py` script to generate new LST maps using a trained Generator model. Make sure to update `config/config.py` to point `INITIAL_MODEL_G` to your desired trained generator model.

```bash
python predict.py
```

