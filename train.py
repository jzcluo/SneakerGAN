import torch
import torch.nn as nn
import torch.optim as optim

from dcgan import Generator as dcgan_generator
from dcgan import Discriminator as dcgan_discriminator

from config import *
import utils

# train
if model == "DCGAN":
    generator = dcgan_generator().to(device)
    discriminator = dcgan_discriminator().to(device)
    generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

generator.apply(utils.weights_init)
discriminator.apply(utils.weights_init)

generator_losses = []
discriminator_losses = []

loss = nn.BCELoss()

fixed_noise = torch.rand(batch_size, latent_feature_size, device=device)

dataloader = utils.get_dataloader()

for epoch in range(1, epochs):
    for i, real_data in enumerate(dataloader):
        if epoch < add_noise_until:
            real_data = (real_data[0] + torch.normal(torch.zeros(real_data[0].shape),std=0.01)).to(device)
        else:
            real_data = real_data[0].to(device)
            
        current_batch_size = real_data.shape[0]

        for j in range(discriminator_iterations):
            discriminator.zero_grad()

            real_image_score = discriminator(real_data)

            label = torch.full((current_batch_size, ), real_label, device=device)
            discriminator_real_loss = loss(real_image_score, label)
            discriminator_real_loss.backward()

            fake_image_input = torch.rand(current_batch_size, latent_feature_size, device=device)

            fake_image_data = generator(fake_image_input)

            fake_image_score = discriminator(fake_image_data.detach())

            label = torch.full((current_batch_size, ), fake_label, device=device)
            discriminator_fake_loss = loss(fake_image_score, label)
            discriminator_fake_loss.backward()

            discriminator_optimizer.step()

        for k in range(generator_iterations):

            generator.zero_grad()
            fake_image_input = torch.rand(current_batch_size, latent_feature_size, device=device)

            fake_image_data = generator(fake_image_input)

            fake_image_score = discriminator(fake_image_data)

            label = torch.full((current_batch_size, ), real_label, device=device)
            generator_loss = loss(fake_image_score, label)
            generator_loss.backward()

            generator_optimizer.step()

    if epoch % test_every == 0:
        test_image_input = torch.rand(num_test_images, latent_feature_size, device=device)
        test_image_data = generator(test_image_input).detach().cpu()
        utils.save_test_images(test_image_data, model, epoch)
        
    if epoch % save_every == 0:
        utils.save_model(generator, discriminator, generator__optimizer, discriminator_optimizer, epoch)
