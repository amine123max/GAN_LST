import torch

DATA_DIR = r"C:\Users\Amine\Desktop\GAN_LST\data\land_surface_temperature"
MODELS_DIR = r"C:\Users\Amine\Desktop\GAN_LST\models"
RESULTS_DIR = r"C:\Users\Amine\Desktop\GAN_LST\results"

BATCH_SIZE = 4  # Reduced for 10-channel input (uses more memory)
IMAGE_SIZE = 64
NC = 1
NUM_INPUT_YEARS = 10
NGF = 64
NDF = 64
NUM_EPOCHS = 150  # Increased for better convergence

# Separate learning rates to balance G and D
LR_G = 0.0002  # Generator learning rate
LR_D = 0.00005  # Lower discriminator LR to prevent it from being too strong
BETA1 = 0.5
BETA2 = 0.999

# L1 loss weight - higher value emphasizes reconstruction quality
LAMBDA_L1 = 200  # Increased from 100 to emphasize pixel-wise accuracy

# Label smoothing to stabilize training
LABEL_SMOOTH_REAL = 0.9  # Real labels = 0.9 instead of 1.0
LABEL_SMOOTH_FAKE = 0.1  # Fake labels = 0.1 instead of 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Learning rate scheduler
USE_LR_SCHEDULER = True
LR_DECAY_EPOCH = 75  # Start decay after this epoch
LR_DECAY_RATE = 0.5  # Multiply LR by this factor

SAVE_MODEL_EPOCHS = 10
SAVE_IMAGE_EPOCHS = 5

# Path to pre-trained models for prediction or resuming training
INITIAL_MODEL_G = ""
INITIAL_MODEL_D = ""

# Data split: use first 49 years for training, last year (2020) for testing
TRAIN_YEARS = 49
TEST_YEAR = 2020