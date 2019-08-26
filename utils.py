from config import *
import matplotlib.pyplot as plt
import os
import glob
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)
        
def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.ImageFolder(root=img_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def save_test_images(test_images, model_name, epoch):
    filename = model_name + "-epoch" + str(epoch) + ".png"
    nrows = math.floor(math.sqrt(test_images.shape[0]))
    plt.axis("off")
    plt.imsave(filename, np.transpose(vutils.make_grid(test_image_data, padding=5, normalize=True, nrow=nrows), (1,2,0)))
    
def save_model(generator, discriminator, generator_optimizer, discriminator_optimizer, epoch):
    torch.save({
        'generator' : generator.state_dict(),
        'discriminator' : discriminator.state_dict(),
        'optimizerG' : generator_optimizer.state_dict(),
        'optimizerD' : discriminator_optimizer.state_dict(),
    }, "model/sneaker-gan-" + str(epoch) + ".model")
    
def load_model(filename):
    if filename is None:
        filename = max(glob.glob("model"), key=os.path.getctime)
    if torch.cuda.is_available():
        state = torch.load(filename)
    else:
        state = torch.load(filename, map_location="cpu")
    return state