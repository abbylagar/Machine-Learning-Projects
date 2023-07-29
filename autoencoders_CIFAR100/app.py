"""
hugging face app.py file
"""


#import functions

import matplotlib.pylab as plt
import PIL.Image as Image
import gradio as gr
import torch
import torchvision
from torch import nn
from einops import rearrange, reduce
from pytorch_lightning import LightningModule, Trainer, Callback
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from denoiseCIFAR100 import DenoiseCIFAR100Model
from torchvision import transforms
from PIL import Image
import numpy as np



modelcheck = DenoiseCIFAR100Model.load_from_checkpoint("./autoencoder_model.ckpt")
modelcheck=modelcheck.eval()

def denoise(image):
  in_im=tinp = transforms.ToTensor()(image).unsqueeze(0)
  with torch.no_grad():
      out_im = modelcheck(in_im)

  out = out_im[0].permute(1, 2, 0)
  out = out.numpy()
  im = Image.fromarray((out * 255).astype(np.uint8))
  im.save("./output.jpeg")
  return "./output.jpeg"
  


iface = gr.Interface(denoise, inputs=gr.inputs.Image(shape=(32,32), image_mode="RGB", type="numpy",label="Input"),
             outputs=gr.outputs.Image(type="file",label="Output"),
            examples=["panda.jpeg","outside.jpeg"],
            live=False,layout="horizontal", interpretation=None,
            title="CIFAR100 Denoising",
            #description=description,
            #article=article
            )
iface.launch()    