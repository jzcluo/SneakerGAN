import torch

latent_feature_size = 100

lr = 0.0002
beta1 = 0.5

real_label = 1
fake_label = 0

epochs = 200
discriminator_iterations = 1
generator_iterations = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = "DCGAN"

epochs = 500
add_noise_until = 200
discriminator_iterations = 1
generator_iterations = 1

save_every = 50
test_every = 50

img_size = 128
img_folder = "datasets"
batch_size = 64

num_test_images = 25