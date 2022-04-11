import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from perceptual_advex.distances import normalize_flatten_features
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
from torchvision.models import resnet50
import os
import numpy as np
import gc

torch.manual_seed(0)
device = torch.device('cuda')

class ResNet_orig_LPIPS(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        pretrained = bool(pretrained)
        print("Use pytorch pretrained weights: [{}]".format(pretrained))
        self.back = resnet50(pretrained=pretrained)
        self.back.fc = nn.Linear(2048, 
                                 num_classes)
        # ===== Truncate the back and append the model to enable attack models
        model_list = list(self.back.children())
        self.head = nn.Sequential(
            *model_list[0:4]
        )
        self.layer1 = model_list[4]
        self.layer2 = model_list[5]
        self.layer3 = model_list[6]
        self.layer4 = model_list[7]
        self.tail = nn.Sequential(
            *[model_list[8],
              nn.Flatten(),
              model_list[9]]
            )    
        # print()    

    def features(self, x):
        """
            This function is called to produce perceptual features.
            Output ==> has to be a tuple of conv features.
        """
        x = x.type(self.back.fc.weight.dtype)
        x = self.head(x)
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        return x_layer1, x_layer2, x_layer3, x_layer4
    
    def classifier(self, last_layer):
        last_layer = self.tail(last_layer)
        return last_layer
    
    def forward(self, x):
        return self.classifier(self.features(x)[-1])
    
    def features_logits(self, x):
        features = self.features(x)
        logits = self.classifier(features[-1])
        return features, logits

base_model = ResNet_orig_LPIPS(num_classes=100,pretrained=False).to(device)

# please download the checkpoint.pth from our Google Drive
pretrained_path = os.path.join("/home/buyun/Documents/GitHub/PyGRANSO/examples/data/checkpoints/","checkpoint.pth")
state_dict = torch.load(pretrained_path)["model_state_dict"]
base_model.load_state_dict(state_dict)

# The ImageNet dataset is no longer publicly accessible. 
# You need to download the archives externally and place them in the root directory
valset = datasets.ImageNet('/home/buyun/Documents/datasets/ImageNet/', split='val', transform=transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor()]), download=False)
val_loader = torch.utils.data.DataLoader(valset, batch_size=1,shuffle=False, num_workers=0, collate_fn=None, pin_memory=False,)

# for batch_idx, samples in enumerate(val_loader):
#       print(batch_idx, samples)

inputs, labels = next(iter(val_loader))
i=0
for inputs, labels in val_loader:
    i+=1
    if i > 200:
        break

# All the user-provided data (vector/matrix/tensor) must be in torch tensor format.
# As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
# Also, please make sure the device of provided torch tensor is the same as opts.torch_device.
inputs = inputs.to(device=device, dtype=torch.double)
labels = labels.to(device=device)

# variables and corresponding dimensions.
var_in = {"x_tilde": list(inputs.shape)}

# def neg_ce_loss(logits, labels):
#     return -torch.nn.functional.cross_entropy(logits, labels)

# def MarginLoss(logits,labels):
#     correct_logits = torch.gather(logits, 1, labels.view(-1, 1))
#     max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1)
#     top_max, second_max = max_2_logits.chunk(2, dim=1)
#     top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
#     labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
#     labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
#     max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max
#     loss = -(max_incorrect_logits - correct_logits).clamp(max=1).squeeze().sum()
#     return loss

def user_fn(X_struct, inputs, labels, lpips_model, model, attack_type, eps=1.5):
    adv_inputs = X_struct.x_tilde
    epsilon = eps
    logits_outputs = model(adv_inputs)
    f = -torch.nn.functional.cross_entropy(logits_outputs,labels)

    # inequality constraint
    ci = pygransoStruct()
    if attack_type == 'L_2':
        ci.c1 = torch.norm((inputs - adv_inputs).reshape(inputs.size()[0], -1)) - epsilon
    elif attack_type == 'L_inf':
        ci.c1 = torch.norm((inputs - adv_inputs).reshape(inputs.size()[0], -1), float('inf')) - epsilon
    else:
        input_features = normalize_flatten_features( lpips_model.features(inputs)).detach()
        adv_features = lpips_model.features(adv_inputs)
        adv_features = normalize_flatten_features(adv_features)
        lpips_dists = (adv_features - input_features).norm(dim=1)
        ci.c1 = lpips_dists - epsilon

    # equality constraint
    ce = None
    return [f,ci,ce]

attack_type = "Perceptual"
var_in = {"x_tilde": list(inputs.shape)}

comb_fn = lambda X_struct : user_fn(X_struct, inputs, labels, lpips_model=base_model, model=base_model, attack_type=attack_type, eps=0.25)

opts = pygransoStruct()
opts.torch_device = device
# opts.maxit = 1000
opts.opt_tol = 1e-4*np.sqrt(torch.numel(inputs))
# opts.viol_ineq_tol = 5e-5



opts.print_frequency = 1
# opts.limited_mem_size = 100
opts.limited_mem_size = 10

# opts.mu0 = 0.1
opts.viol_ineq_tol = 1e-5
opts.maxit = 50
opts.opt_tol = 1e-6

# opts.is_backtrack_linesearch = True
# opts.search_direction_rescaling = True

opts.x0 = torch.reshape(inputs,(torch.numel(inputs),1))

start = time.time()
soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))






import matplotlib.pyplot as plt

def rescale_array(array):
    ele_min, ele_max = np.amin(array), np.amax(array)
    array = (array - ele_min) / (ele_max - ele_min)
    return array

def tensor2img(tensor):
    tensor = torch.nn.functional.interpolate(
        tensor,
        scale_factor=3,
        mode="bilinear"
    )
    array = tensor.detach().cpu().numpy()[0, :, :, :]
    array = np.transpose(array, [1, 2, 0])
    return array

final_adv_input = torch.reshape(soln.final.x,inputs.shape)

ori_image = rescale_array(tensor2img(inputs))
adv_image = rescale_array(tensor2img(final_adv_input))

f = plt.figure()
f.add_subplot(1,2,1)
plt.imshow(ori_image)
plt.title('Original Image')
plt.axis('off')
f.add_subplot(1,2,2)
plt.imshow(adv_image)
plt.title('Adversarial Attacked Image')
plt.axis('off')
plt.show()