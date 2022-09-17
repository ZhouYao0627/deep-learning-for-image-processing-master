import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from pytorch_classification.grad_cam.utils import GradCAM, show_cam_on_image
from model_resnet import resnext101_32x8d


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnext101_32x8d(num_classes=45)
    target_layers = [model.layer4]

    # model = models.resnet50(pretrained=True)
    # target_layers = [model.layer4[-1]]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    # img_path = "both.png"
    data_root = os.path.abspath(os.path.join(os.getcwd()), "../")
    img_path = os.path.join(data_root, "B", "harbor", "0001.tif")
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)

    # [N, C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    # target_category = 281  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    target_category = None  # harbor
    # target_category = 3  # "river", "0020.tif"的3不错

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
