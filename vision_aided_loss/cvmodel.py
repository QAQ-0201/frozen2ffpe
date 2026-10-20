import importlib
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import timm
import antialiased_cnns
import gdown

import sys

from open_clip_custom import create_model_from_pretrained
from vision_aided_loss.DiffAugment_pytorch import DiffAugment
from vision_aided_loss.CLIP.clip import *

class Vgg(nn.Module):
    def __init__(self, cv_type='adv'):
        super().__init__()
        self.cv_type = cv_type
        self.model = antialiased_cnns.vgg16(pretrained=True, filter_size=4).features
        self.model.eval()
        self.model.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_mean = torch.tensor([0.485, 0.456, 0.406])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])

    def forward(self, x):
        x = F.interpolate(x*0.5+0.5, size=(224, 224), mode='area')
        x = x - self.image_mean[:, None, None].to(x.device)
        x /= self.image_std[:, None, None].to(x.device)
        out = self.model(x)

        if 'pool' in self.cv_type:
            out = self.pool(out)
            out = out.view(out.shape[0], -1)

        return out


class Swin(torch.nn.Module):

    def __init__(self, cv_type='adv'):
        super().__init__(
        )

        self.cv_type = cv_type
        self.model = timm.create_model('swin_tiny_patch4_window7_224')
        if not os.path.exists(os.path.join(os.environ['HOME'], '.cache', 'moby_swin_t_300ep_pretrained.pth')):
            st = gdown.download("https://drive.google.com/u/0/uc?id=1PS1Q0tAnUfBWLRPxh9iUrinAxeq7Y--u&export=download", os.path.join(os.environ['HOME'], '.cache', 'moby_swin_t_300ep_pretrained.pth'))

        st = torch.load(os.path.join(os.environ['HOME'], '.cache', 'moby_swin_t_300ep_pretrained.pth'), map_location='cpu')
        new_st = {}
        for each in st['model'].keys():
            if 'encoder.' in each:
                newk = each.replace('encoder.', '')
                new_st[newk] = st['model'][each]
        self.model.load_state_dict(new_st, strict=False)

        self.model.eval()
        self.model.requires_grad = False

        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_custom(self, x, return_intermediate=False):
        x = self.model.patch_embed(x)
        if self.model.ape:
            x = x + self.model.absolute_pos_embed
        x = self.model.pos_drop(x)
        x = self.model.layers(x)     
        x = self.model.norm(x)
        if return_intermediate:
            return x.transpose(1, 2)

        x = self.model.avgpool(x.transpose(1, 2)) 
        x = torch.flatten(x, 1)
        return x

    def __call__(self, x):
        x = F.interpolate(x*0.5+0.5, size=(224, 224), mode='area')
        x = x - self.image_mean[:, None, None].to(x.device)
        x /= self.image_std[:, None, None].to(x.device)

        if 'conv' in self.cv_type:
            x = self.forward_custom(x, return_intermediate=True)
            x = x.reshape(-1, 768, 7, 7)
            return x

        return self.model.forward_features(x)


class CLIP(torch.nn.Module):

    def __init__(self, cv_type='adv'):
        super().__init__(
        )

        self.cv_type = cv_type
        self.model, _ = clip.load("ViT-B/32", jit=False, device='cpu')
        self.model = self.model.visual
        self.model.eval()
        self.model.requires_grad = False

        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    def forward_custom(self, x):
        x = self.model.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)

        x = x.permute(1, 0, 2)

        x1 = []
        feat_points = [0, 4, 8, len(self.model.transformer.resblocks)]
        for i in range(len(feat_points)-1):
            x = self.model.transformer.resblocks[feat_points[i]:feat_points[i+1]](x)
            x1.append(x.permute(1, 0, 2))

        x = self.model.ln_post(x1[-1][:, 0, :])
        if self.model.proj is not None:
            x = x @ self.model.proj
        x1[-1] = x
        return x1

    def __call__(self, x):
        x = F.interpolate(x*0.5+0.5, size=(224, 224), mode='area')
        x = x - self.image_mean[:, None, None].to(x.device)
        x /= self.image_std[:, None, None].to(x.device)

        if 'conv_multi_level' in self.cv_type:
            x = self.forward_custom(x.type(self.model.conv1.weight.dtype))
            x[0] = x[0][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 7, 7).float()
            x[1] = x[1][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 7, 7).float()
            x[2] = x[2].float()
        else:
            x = self.model(x.type(self.model.conv1.weight.dtype)).float()

        return x


class DINO(torch.nn.Module):

    def __init__(self, cv_type='adv'):
        super().__init__(
        )

        self.cv_type = cv_type
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.model.eval()
        self.model.requires_grad = False
        self.input_resolution = 224
        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])

    def __call__(self, x):
        x = F.interpolate(x*0.5+0.5, size=(224, 224), mode='area')
        x = x - self.image_mean[:, None, None].to(x.device)
        x /= self.image_std[:, None, None].to(x.device)

        if 'conv_multi_level' in self.cv_type:
            x = self.model.get_intermediate_layers(x, n=8)
            x = [x[i] for i in [0, 4, -1]]
            x[0] = x[0][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 14, 14)
            x[1] = x[1][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 14, 14)
            x[2] = x[2][:, 0, :]
        else:
            x = self.model(x)
        return x


class Conch(torch.nn.Module):

    def __init__(self, cv_type='adv'):
        super().__init__()

        self.cv_type = cv_type
        self.model = timm.create_model(
            "vit_large_patch16_224",
            img_size=448,
            patch_size=16,
            init_values=1.0,
            num_classes=0,
            dynamic_img_size=True,
        )
        ckpt = torch.load("./checkpoints/conchv1_5.bin", map_location="cpu")
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            ckpt = ckpt['model']

        strip_prefixes = ("module.", "model.", "visual.trunk.", "visual.", "trunk.")
        new_ckpt = {}
        for k, v in ckpt.items():
            new_k = k
            changed = True
            while changed:
                changed = False
                for prefix in strip_prefixes:
                    if new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]
                        changed = True
            new_ckpt[new_k] = v

        info = self.model.load_state_dict(new_ckpt, strict=False)
        if info.missing_keys:
            core_terms = ['blocks.', 'patch_embed.', 'norm.', 'pos_embed', 'cls_token', 'fc_norm.']
            core_missing = [k for k in info.missing_keys
                           if any(t in k for t in core_terms)]
            if core_missing:
                raise RuntimeError(
                    f"CONCH v1.5: {len(core_missing)} core weights not loaded - "
                    f"checkpoint key mismatch. First 5: {core_missing[:5]}")
            print(f"[CONCH v1.5] {len(info.missing_keys)} non-core keys missing (head etc.), OK")
        if info.unexpected_keys:
            print(f"[CONCH v1.5] {len(info.unexpected_keys)} unexpected keys ignored")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.input_resolution = 448
        self.image_mean = torch.tensor([0.485, 0.456, 0.406])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])

    def __call__(self, x):
        x = F.interpolate(x * 0.5 + 0.5, size=(448, 448), mode='area')
        x = x - self.image_mean[:, None, None].to(x.device)
        x /= self.image_std[:, None, None].to(x.device)

        if 'conv_multi_level' in self.cv_type:
            x = self.model.get_intermediate_layers(x, n=8)
            x = [x[i] for i in [0, 4, -1]]
            patches_0 = x[0][:, 1:, :] if x[0].shape[1] == 28 * 28 + 1 else x[0]
            patches_1 = x[1][:, 1:, :] if x[1].shape[1] == 28 * 28 + 1 else x[1]
            if x[2].shape[1] == 28 * 28 + 1:
                global_feat = x[2][:, 0, :]
            else:
                global_feat = x[2].mean(dim=1)
            x[0] = patches_0.permute(0, 2, 1).reshape(-1, 1024, 28, 28)
            x[1] = patches_1.permute(0, 2, 1).reshape(-1, 1024, 28, 28)
            x[2] = global_feat
        else:
            x = self.model(x)
        return x


class CVBackbone(torch.nn.Module):

    def __init__(self, cv_type, output_type, diffaug=False, device='cpu'):
        super().__init__(
        )
        cv_type = cv_type.split('+')
        output_type = output_type.split('+')
        self.class_name_dict = {
                'clip': 'vision_aided_loss.cvmodel.CLIP',
                'dino': 'vision_aided_loss.cvmodel.DINO',
                'vgg': 'vision_aided_loss.cvmodel.Vgg',
                'swin': 'vision_aided_loss.cvmodel.Swin',
                'conch': 'vision_aided_loss.cvmodel.Conch',
            }

        self.cv_type = cv_type
        self.policy = ''
        if diffaug:
            self.policy = 'color,translation,cutout'

        self.models = []
        for cv_type_, output_type_ in zip(cv_type, output_type):
            modellib = importlib.import_module('.'.join(self.class_name_dict[cv_type_].split('.')[:-1]))
            model = None
            target_model_name = self.class_name_dict[cv_type_].split('.')[-1]
            for name, cls in modellib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    model = cls

            cv_type_ = cv_type_ + '_' + output_type_
            model = model(cv_type=cv_type_).requires_grad_(False).to(device)
            self.models.append(model)

    def __call__(self, images):
        image_features = []
        for i, each in enumerate(self.models):
            feat = [x.to(dtype=images.dtype) for x in each(DiffAugment(images, policy=self.policy))]
            image_features.append(feat)
        return image_features
