import torch.nn as nn
import torch.nn.functional as F


class PIPNet(nn.Module):
    def __init__(self, params, backbone, depth='18'):
        super(PIPNet, self).__init__()
        self.params = params
        self.stride = params['stride']
        self.num_nb = params['num_nb']
        self.num_lms = params['num_lms']

        width_map = {'18': 512, '50': 2048, '101': 2048}
        width = width_map.get(depth)

        self.initialize_backbone(backbone)
        self.create_output(width)

    def initialize_backbone(self, backbone):
        self.conv1, self.bn1, self.maxpool = backbone.conv1, backbone.bn1, backbone.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4

    def initialize_layers(self, layers):
        for layer in layers:
            nn.init.normal_(layer.weight, std=0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def create_output(self, width):
        layers = self.initialize_output(width, self.num_lms, self.num_nb)
        self.cls_layer, self.x_layer, self.y_layer, self.nb_x_layer, self.nb_y_layer = layers
        self.initialize_layers(layers)

    @staticmethod
    def initialize_output(width, num_lms, num_nb):
        num_lms = [num_lms, num_lms, num_lms, num_lms * num_nb, num_lms * num_nb]
        return [nn.Conv2d(width, lms_points, kernel_size=1) for lms_points in num_lms]

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)

        return {'cls_layer': x1, 'x_layer': x2, 'y_layer': x3, 'nb_x_layer': x4, 'nb_y_layer': x5}