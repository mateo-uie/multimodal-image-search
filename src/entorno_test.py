import platform

import numpy
import PIL
import sentence_transformers
import torch
import torchvision
import transformers

print("Python", platform.python_version(), platform.machine())
for m in (torch, torchvision, transformers, sentence_transformers, numpy, PIL):
    print(m.__name__, m.__version__)
print("CUDA:", torch.cuda.is_available())
