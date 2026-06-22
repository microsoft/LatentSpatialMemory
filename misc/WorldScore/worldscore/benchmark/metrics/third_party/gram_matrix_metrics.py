from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

from worldscore.benchmark.metrics.base_metrics import BaseMetric

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU
normalization_mean = torch.tensor([0.485, 0.456, 0.406])
normalization_std = torch.tensor([0.229, 0.224, 0.225])

# desired depth layers to compute style/content losses :
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name, device):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               style_layers):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of style
    # losses
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses

class GramMatrixMetric(BaseMetric):
    """
    TODO: add description
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._model = vgg19(weights=VGG19_Weights.DEFAULT).features.to(self._device).eval()
        self._normalization_mean = normalization_mean.to(self._device)
        self._normalization_std = normalization_std.to(self._device)
        self._style_layers_default = style_layers_default

    def _compute_scores(
        self, 
        reference_image: str,
        rendered_images: List[str],
    ) -> float:

        img1 = image_loader(reference_image, self._device)
        style_scores = []
        for rendered_image in rendered_images:
            img2 = image_loader(rendered_image, self._device)

            model, style_losses = get_style_model_and_losses(self._model, self._normalization_mean, self._normalization_std, img1, img2, self._style_layers_default)
            input_img = img2.clone()
            model(input_img)
            style_score = 0
            for sl in style_losses:
                style_score += sl.loss
            style_scores.append(style_score.item())
        
        return sum(style_scores) / len(style_scores)
