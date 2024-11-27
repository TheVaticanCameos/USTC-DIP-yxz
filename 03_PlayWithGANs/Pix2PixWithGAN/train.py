import torch
import torch.nn as nn
from facades_dataset import FacadesDataset
from utils import save_images
from model import Discriminator, Generator


print("Loading data...")

train_list_file = "train_list.txt"
valid_list_file = "val_list.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = FacadesDataset(train_list_file)
valid_dataset = FacadesDataset(valid_list_file)

BATCH_SIZE = 32
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.999), weight_decay=1e-5)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999), weight_decay=1e-5)

scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=100, gamma=0.5)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=100, gamma=0.5)

criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()

NUM_EPOCHS = 500
print("Start training...")
for epoch in range(NUM_EPOCHS):
    for i, (real_A, real_B) in enumerate(train_dataloader):
        real_A, real_B = real_A.to(device), real_B.to(device)

        optimizer_D.zero_grad()
        fake_B = generator(real_A)
        real_labels = torch.ones(real_A.size(0), 1).to(device)
        fake_labels = torch.zeros(real_A.size(0), 1).to(device)    
        real_loss = criterion_GAN(discriminator(real_A, real_B), real_labels)
        fake_loss = criterion_GAN(discriminator(real_A, fake_B.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()
        scheduler_D.step()

        optimizer_G.zero_grad()
        g_loss_GAN = criterion_GAN(discriminator(real_A, fake_B), real_labels)
        g_loss_L1 = criterion_L1(fake_B, real_B)
        g_loss = g_loss_GAN + 20 * g_loss_L1
        g_loss.backward()
        optimizer_G.step()
        scheduler_G.step()

        print(f"Epoch [{epoch}/{NUM_EPOCHS}], Step [{i}/{len(train_dataloader)}], "
              f"D Loss: {d_loss.item()}, G Loss: {g_loss.item()}, G Loss GAN: {g_loss_GAN.item()}, G Loss L1: {g_loss_L1.item()}")
        
        if epoch % 5 == 0 and i == 0:
            save_images(real_A, real_B, fake_B, 'train_results', epoch)
        
    if epoch % 50 == 0:
        with torch.no_grad():
            for i, (real_A, real_B) in enumerate(valid_dataloader):
                real_A, real_B = real_A.to(device), real_B.to(device)
                fake_B = generator(real_A)
                save_images(real_A, real_B, fake_B, 'val_results', epoch, num_images=4)
