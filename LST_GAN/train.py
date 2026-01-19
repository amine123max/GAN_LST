import torch
import torch.optim as optim
from config import config
from src.data_loader import get_dataloaders
from src.models import Generator, Discriminator, weights_init
from src.trainer import Trainer

def main():
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Dataloaders
    print("Loading data...")
    train_dataloader, test_dataloader = get_dataloaders(config.DATA_DIR, config.BATCH_SIZE, config.IMAGE_SIZE)
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Testing samples: {len(test_dataloader.dataset)}")

    # Initialize models
    print("Initializing models...")
    netG = Generator(config.NC, config.NGF, config.NUM_INPUT_YEARS).to(device)
    netD = Discriminator(config.NC, config.NDF, config.NUM_INPUT_YEARS).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Load pre-trained models if specified
    if config.INITIAL_MODEL_G:
        netG.load_state_dict(torch.load(config.INITIAL_MODEL_G, map_location=device))
        print(f"Loaded generator from {config.INITIAL_MODEL_G}")
    if config.INITIAL_MODEL_D:
        netD.load_state_dict(torch.load(config.INITIAL_MODEL_D, map_location=device))
        print(f"Loaded discriminator from {config.INITIAL_MODEL_D}")

    # Optimizers with separate learning rates
    optimizerD = optim.Adam(netD.parameters(), lr=config.LR_D, betas=(config.BETA1, config.BETA2))
    optimizerG = optim.Adam(netG.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))

    # Learning rate schedulers
    schedulerD = None
    schedulerG = None
    if config.USE_LR_SCHEDULER:
        schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=config.LR_DECAY_EPOCH, gamma=config.LR_DECAY_RATE)
        schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=config.LR_DECAY_EPOCH, gamma=config.LR_DECAY_RATE)
        print(f"Using LR scheduler: decay by {config.LR_DECAY_RATE} every {config.LR_DECAY_EPOCH} epochs")

    # Trainer
    trainer = Trainer(netG, netD, optimizerG, optimizerD, train_dataloader, device, config, schedulerG, schedulerD)

    # Train
    print("Starting training...")
    trainer.train()
    print("Training completed!")

if __name__ == '__main__':
    main()
