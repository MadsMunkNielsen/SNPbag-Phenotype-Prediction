

fn = 'code/SNPbag/emb_example.pt'
import torch
emb_tensor = torch.load(fn)
emb_tensor.shape


import sys
sys.path.append('code/SNPbag')
from nature import *
import numpy as np

if emb_tensor.dtype == torch.bfloat16:
    emb_tensor = emb_tensor.to(torch.float32)
emb_array = emb_tensor.cpu().numpy().transpose()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.imshow(emb_array, aspect='auto', cmap='jet')
plt.colorbar(label='')
# plt.title('Embedding Matrix for emb_dict[0]')
plt.xlabel('Contig')
plt.ylabel('Embedding dimension')
plt.tight_layout()
plt.show()
plt.close()