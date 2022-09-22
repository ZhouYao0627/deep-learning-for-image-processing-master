import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image
from model_resnet import resnext101_32x8d, resnet50


# best_acc 0.7975793650793651
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create model
    model = resnext101_32x8d(num_classes=45).to(device)
    target_layers = [model.layer4[-1]]

    # load model weights
    weights_path = "../Test8_densenet/save_weights/train_NWPU45_2080_bad.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
    img_path = os.path.join(data_root, "data_set", "NWPU45", "80_20", "train", "airplane", "airplane_061.jpg")
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)

    # [N, C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = None  # 类别真实数减1

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.colorbar()  # 显示colorbar
    plt.savefig("./plot/GradCAM_NWPU45_airplane_061_bad.png", dpi=500, format="png")
    plt.show()


if __name__ == '__main__':
    main()
