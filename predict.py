import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "向日葵.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    #torch.unsqueeze():在指定维度上增加一个新的维度，通常用于改变张量的形状
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = AlexNet(num_classes=5).to(device)

    # load model weights
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        #torch.squeeze():用于去除张量中的尺寸为1的维度。这通常用于处理批量推断的情况，以便获得形状适当的输出。
        #如果模型的输出是形状 (1, C, H, W)，squeeze 操作将移除第一个维度，使输出形状变为 (C, H, W)。
        #.cpu():将张量移回到 CPU 设备
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        print(predict)#每种类别的可能性 tensor([5.7935e-06, 8.1234e-03, 1.3366e-05, 9.9129e-01, 5.6708e-04])
        print(predict_cla)#预测类别 3


    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    print(print_res)
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()