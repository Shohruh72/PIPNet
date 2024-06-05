import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PIPNet(nn.Module):
    def __init__(self, params, resnet_type, mean_indices,
                 reverse_index1, reverse_index2, max_len):
        super(PIPNet, self).__init__()

        resnet = self.get_resnet(resnet_type)
        if resnet_type == 'resnet18':
            feature_size = 512
        else:
            feature_size = 2048

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.sigmoid = nn.Sigmoid()
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.params = params
        self.resnet_type = resnet_type
        self.max_len = max_len
        self.mean_indices = mean_indices
        self.reverse_index1 = reverse_index1
        self.reverse_index2 = reverse_index2

        self.cls_layer = nn.Conv2d(feature_size, params['num_lms'], 1, 1, 0)
        self.x_layer = nn.Conv2d(feature_size, params['num_lms'], 1, 1, 0)
        self.y_layer = nn.Conv2d(feature_size, params['num_lms'], 1, 1, 0)
        self.nb_x_layer = nn.Conv2d(feature_size, params['num_nb'] * params['num_lms'], 1, 1, 0)
        self.nb_y_layer = nn.Conv2d(feature_size, params['num_nb'] * params['num_lms'], 1, 1, 0)

        nn.init.normal_(self.cls_layer.weight, std=0.001)
        if self.cls_layer.bias is not None:
            nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.nb_x_layer.weight, std=0.001)
        if self.nb_x_layer.bias is not None:
            nn.init.constant_(self.nb_x_layer.bias, 0)

        nn.init.normal_(self.nb_y_layer.weight, std=0.001)
        if self.nb_y_layer.bias is not None:
            nn.init.constant_(self.nb_y_layer.bias, 0)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        out_cls = self.cls_layer(x)
        out_x = self.x_layer(x)
        out_y = self.y_layer(x)
        out_nb_x = self.nb_x_layer(x)
        out_nb_y = self.nb_y_layer(x)
        if self.training:
            return out_cls, out_x, out_y, out_nb_x, out_nb_y

        b, ch, h, w = out_cls.size()
        assert b == 1

        out_cls = out_cls.view(b * ch, -1)
        max_ids = torch.argmax(out_cls, 1)
        max_cls = torch.max(out_cls, 1)[0]
        max_ids = max_ids.view(-1, 1)
        max_ids_nb = max_ids.repeat(1, self.params['num_nb']).view(-1, 1)

        out_x = out_x.view(b * ch, -1)
        out_x_select = torch.gather(out_x, 1, max_ids)
        out_x_select = out_x_select.squeeze(1)
        out_y = out_y.view(b * ch, -1)
        out_y_select = torch.gather(out_y, 1, max_ids)
        out_y_select = out_y_select.squeeze(1)

        out_nb_x = out_nb_x.view(b * self.params['num_nb'] * ch, -1)
        out_nb_x_select = torch.gather(out_nb_x, 1, max_ids_nb)
        out_nb_x_select = out_nb_x_select.squeeze(1).view(-1, self.params['num_nb'])
        out_nb_y = out_nb_y.view(b * self.params['num_nb'] * ch, -1)
        out_nb_y_select = torch.gather(out_nb_y, 1, max_ids_nb)
        out_nb_y_select = out_nb_y_select.squeeze(1).view(-1, self.params['num_nb'])

        tmp_x = (max_ids % w).view(-1, 1).float() + out_x_select.view(-1, 1)
        tmp_y = (max_ids // w).view(-1, 1).float() + out_y_select.view(-1, 1)
        tmp_x /= 1.0 * self.params['input_size'] / self.params['stride']
        tmp_y /= 1.0 * self.params['input_size'] / self.params['stride']

        tmp_nb_x = (max_ids % w).view(-1, 1).float() + out_nb_x_select
        tmp_nb_y = (max_ids // w).view(-1, 1).float() + out_nb_y_select
        tmp_nb_x = tmp_nb_x.view(-1, self.params['num_nb'])
        tmp_nb_y = tmp_nb_y.view(-1, self.params['num_nb'])
        tmp_nb_x /= 1.0 * self.params['input_size'] / self.params['stride']
        tmp_nb_y /= 1.0 * self.params['input_size'] / self.params['stride']

        lms_pred = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        tmp_nb_x = tmp_nb_x[self.reverse_index1, self.reverse_index2].view(self.params['num_lms'], self.max_len)
        tmp_nb_y = tmp_nb_y[self.reverse_index1, self.reverse_index2].view(self.params['num_lms'], self.max_len)
        tmp_x = torch.mean(torch.cat((tmp_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
        tmp_y = torch.mean(torch.cat((tmp_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1)
        return torch.flatten(lms_pred_merge)

    @staticmethod
    def get_resnet(resnet_type):
        if resnet_type == 'resnet18':
            return models.resnet18(pretrained=True)
        elif resnet_type == 'resnet50':
            return models.resnet50(pretrained=True)
        elif resnet_type == 'resnet101':
            return models.resnet101(pretrained=True)
        else:
            raise ValueError("Unsupported ResNet type. Choose from 'resnet18', 'resnet50', 'resnet101'.")
