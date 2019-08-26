from config import *
import utils
import torch
import matplotlib.pyplot as plt
from dcgan import Generator as dcgan_generator

if model == "DCGAN":
    generator = dcgan_generator().to(device)
    
state = utils.load_model()
generator.load_state_dict(state["generator"])

image_input = torch.rand(1, latent_feature_size, device=device)
image_data = generator(image_input)
transposed_image = image_data.detach().cpu().numpy()[0].transpose([1,2,0])

plt.imshow(transposed_image)
plt.show()