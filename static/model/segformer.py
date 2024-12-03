import colorsys
import copy
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from nets.segformer import SegFormer
from utils.utils import cvtColor, preprocess_input, resize_image, show_config

class SegFormer_Segmentation(object):
    _defaults = {
        "model_path"        : 'static/model/best_epoch_weights.pth',
        "num_classes"       : 3,
        "phi"               : 'b0',
        "input_shape"       : [512, 512],  # 输入图片的大小
        "device"            : 'cuda',      # 使用 'cuda' 或 'cpu'
    }

    # 初始化SegFormer
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.colors = [
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), 
            (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), 
            (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), 
            (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), 
            (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
            (128, 64, 12)
        ]
        
        self.generate()

        show_config(**self._defaults)

    # 获取所有的分类
    def generate(self, onnx=False):
        # 加载模型与权重
        self.net = SegFormer(num_classes=self.num_classes, phi=self.phi, pretrained=False)

        # 根据设备选择（'cuda' 或 'cpu'）初始化
        device = torch.device(self.device if torch.cuda.is_available() and self.device == 'cuda' else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        
        if not onnx:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.to(device)

    # 检测图片
    def detect_image(self, image, count=False, name_classes=None):
        # 代码仅支持RGB图像的预测，其他类型的图像转为RGB
        image = cvtColor(image)
        # 对输入图像进行备份
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        # 传入网络进行预测
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.to(self.net.device)  # 使用与网络相同的设备
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        # 计数
        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image = Image.fromarray(np.uint8(seg_img))
            image = Image.blend(old_img, image, 0.7)

        return image

    # 计算FPS
    def get_FPS(self, image, test_interval):
        image = cvtColor(image)
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.to(self.net.device)  # 使用与网络相同的设备
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        # 计时
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                        int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    # 获取mIoU的PNG图像
    def get_miou_png(self, image):
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.to(self.net.device)  # 使用与网络相同的设备
            pr = self.net(images)[0]  # 传入网络进行预测
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
