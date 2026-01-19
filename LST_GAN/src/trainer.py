import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, generator, discriminator, optimizer_g, optimizer_d, dataloader, device, config, scheduler_g=None, scheduler_d=None):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.dataloader = dataloader
        self.device = device
        self.config = config
        self.criterion_gan = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()

        # Initialize lists to store losses
        self.G_losses = []
        self.D_losses = []

    def train(self):
        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.config.NUM_EPOCHS):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            num_batches = 0

            for i, (input_images, target_images) in enumerate(self.dataloader):
                input_images = input_images.to(self.device)
                target_images = target_images.to(self.device)
                batch_size = input_images.size(0)
                
                # Train Discriminator
                self.optimizer_d.zero_grad()
                
                # Real pair (input, target) with label smoothing
                real_labels = torch.ones(batch_size, 1, 1, 1, device=self.device) * self.config.LABEL_SMOOTH_REAL
                output_real = self.discriminator(input_images, target_images)
                loss_d_real = self.criterion_gan(output_real, real_labels)
                
                # Fake pair (input, generated) with label smoothing
                fake_images = self.generator(input_images)
                fake_labels = torch.zeros(batch_size, 1, 1, 1, device=self.device) + self.config.LABEL_SMOOTH_FAKE
                output_fake = self.discriminator(input_images, fake_images.detach())
                loss_d_fake = self.criterion_gan(output_fake, fake_labels)
                
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                loss_d.backward()
                self.optimizer_d.step()

                # Train Generator
                self.optimizer_g.zero_grad()
                fake_images = self.generator(input_images)
                output = self.discriminator(input_images, fake_images)
                
                # GAN loss + L1 loss (use non-smoothed labels for generator)
                real_labels_g = torch.ones(batch_size, 1, 1, 1, device=self.device)
                loss_g_gan = self.criterion_gan(output, real_labels_g)
                loss_g_l1 = self.criterion_l1(fake_images, target_images) * self.config.LAMBDA_L1
                loss_g = loss_g_gan + loss_g_l1
                
                loss_g.backward()
                self.optimizer_g.step()

                epoch_d_loss += loss_d.item()
                epoch_g_loss += loss_g.item()
                num_batches += 1

                if i % 10 == 0:
                    print(f'[{epoch}/{self.config.NUM_EPOCHS}][{i}/{len(self.dataloader)}] '
                          f'Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f} '
                          f'Loss_G_GAN: {loss_g_gan.item():.4f} Loss_G_L1: {loss_g_l1.item():.4f}')

            # Calculate average epoch losses
            if num_batches > 0:
                avg_d_loss = epoch_d_loss / num_batches
                avg_g_loss = epoch_g_loss / num_batches
                self.D_losses.append(avg_d_loss)
                self.G_losses.append(avg_g_loss)
            else:
                self.D_losses.append(0.0)
                self.G_losses.append(0.0)
            
            # Update learning rates
            if self.scheduler_g is not None:
                self.scheduler_g.step()
            if self.scheduler_d is not None:
                self.scheduler_d.step()
            
            # Print epoch summary
            current_lr_g = self.optimizer_g.param_groups[0]['lr']
            current_lr_d = self.optimizer_d.param_groups[0]['lr']
            print(f'Epoch [{epoch}/{self.config.NUM_EPOCHS}] completed - '
                  f'Avg Loss_D: {avg_d_loss:.4f}, Avg Loss_G: {avg_g_loss:.4f}, '
                  f'LR_G: {current_lr_g:.6f}, LR_D: {current_lr_d:.6f}')

            # Save generated images
            if epoch % self.config.SAVE_IMAGE_EPOCHS == 0:
                self.save_sample_images(epoch, input_images[:4], target_images[:4], fake_images[:4])
            
            # Save model checkpoints
            if (epoch % self.config.SAVE_MODEL_EPOCHS == 0) or (epoch == self.config.NUM_EPOCHS - 1):
                self.save_checkpoint(epoch)
        
        self.plot_losses()

    def save_sample_images(self, epoch, input_imgs, target_imgs, generated_imgs):
        if not os.path.exists(self.config.RESULTS_DIR):
            os.makedirs(self.config.RESULTS_DIR)
        
        # input_imgs shape: (batch, 10, H, W) - 10 years of data
        # For visualization, show last year of input, target, and generated
        last_year_input = input_imgs[:, -1:, :, :]  # Take last channel (most recent year)
        
        comparison = torch.cat([last_year_input, target_imgs, generated_imgs], dim=0)
        save_image(comparison, f"{self.config.RESULTS_DIR}/epoch_{epoch:03d}.png", 
                   nrow=4, normalize=True)

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="Generator")
        plt.plot(self.D_losses, label="Discriminator")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, "gan_losses.png"))
        plt.close()

    def save_checkpoint(self, epoch):
        if not os.path.exists(self.config.MODELS_DIR):
            os.makedirs(self.config.MODELS_DIR)
        torch.save(self.generator.state_dict(), f"{self.config.MODELS_DIR}/generator_epoch_{epoch}.pth")
        torch.save(self.discriminator.state_dict(), f"{self.config.MODELS_DIR}/discriminator_epoch_{epoch}.pth")
        print(f"Saved model checkpoint at epoch {epoch}")


