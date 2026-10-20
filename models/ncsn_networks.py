import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
from torch.nn import init
import math 
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2

    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, num_D=1):
        super(D_NLayersMulti, self).__init__()

        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.model = nn.Sequential(*layers)
        else:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.add_module("model_0", nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[
                                     1, 1], count_include_pad=False)
            for i in range(1, num_D):
                ndf_i = int(round(ndf / (2**i)))
                layers = self.get_layers(input_nc, ndf_i, n_layers, norm_layer)
                self.add_module("model_%d" % i, nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        return sequence

    def forward(self, input):
        if self.num_D == 1:
            return self.model(input)
        result = []
        down = input
        for i in range(self.num_D):
            model = getattr(self, "model_%d" % i)
            result.append(model(down))
            if i != self.num_D - 1:
                down = self.down(down)
        return result


class ConvBlock_cond(nn.Module):
    def __init__(self, in_channel, out_channel,t_emb_dim, kernel_size=4,stride=1,padding=1,norm_layer=None,downsample=True,use_bias=None):
        super().__init__()
        self.downsample=downsample
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)

        if norm_layer is not None:
            self.use_norm =True
            self.norm = norm_layer(out_channel)
        else:
            self.use_norm = False
        self.act = nn.LeakyReLU(0.2, True)
        self.down = Downsample(out_channel)

        self.dense= nn.Linear(t_emb_dim, out_channel)
    def forward(self, input,t_emb):
        out = self.conv1(input)
        out += self.dense(t_emb)[..., None, None]
        if self.use_norm:
            out = self.norm(out)
        out = self.act(out)
        if self.downsample:
            out = self.down(out)

        return out

class NLayerDiscriminator_ncsn(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        super(NLayerDiscriminator_ncsn, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.model_main = nn.ModuleList()
        kw = 4
        padw = 1
        if no_antialias:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            self.model_main.append(ConvBlock_cond(input_nc, ndf, 4*ndf,kernel_size=kw, stride=1, padding=padw,use_bias=use_bias))

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if no_antialias:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)]
            else:
                self.model_main.append(
                    ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult, 4*ndf,kernel_size=kw, stride=1, padding=padw,use_bias=use_bias,norm_layer=norm_layer)

                )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.model_main.append(
            ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult,4*ndf, kernel_size=kw, stride=1, padding=padw,use_bias=use_bias,norm_layer=norm_layer,downsample=False)

        )
        self.final_conv =nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        self.t_embed = TimestepEmbedding(
            embedding_dim=4*ndf,
            hidden_dim=4*ndf,
            output_dim=4*ndf,
            act=nn.LeakyReLU(0.2),
        )

    def forward(self, input,t_emb,input2=None):
        t_emb = self.t_embed(t_emb)
        if input2 is not None:
            out = torch.cat([input,input2],dim=1)
        else:

            out = input
        for layer in self.model_main:
            out = layer(out,t_emb)

        return self.final_conv(out)

class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2),

        )

    def forward(self, temp):
        temb = get_timestep_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2

    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class AdaptiveLayer(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.style_net = nn.Linear(style_dim, in_channel * 2)

        self.style_net.bias.data[:in_channel] = 1
        self.style_net.bias.data[in_channel:] = 0

    def forward(self, input, style):

        style = self.style_net(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = gamma * input + beta

        return out

class SpatialAttention(nn.Module):

    def __init__(self, channels=64, kernel_size=1):
        super(SpatialAttention, self).__init__()
        kernel_size = 1
        padding = 3 if kernel_size == 7 else 0

        self.conv1 = nn.Conv2d(channels, 4, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(4, 4, kernel_size, padding=padding, bias=False)
        self.conv3 = nn.Conv2d(4, channels, kernel_size, padding=padding, bias=False)
        self.maxPooling = nn.MaxPool2d(4, stride=4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.upsample = nn.Upsample(scale_factor=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.maxPooling(x1)
        reshaped1 = torch.reshape(x2, (x2.shape[0], x2.shape[1], -1, x2.shape[2]))
        y = torch.matmul(reshaped1, x2)
        z = self.relu(y)
        z = self.conv2(z)
        t = self.softmax(z)
        out1 = torch.matmul(t, reshaped1)
        conv3_out = self.conv3(out1)
        upsample_out = self.upsample(conv3_out)
        k = torch.reshape(upsample_out, (upsample_out.shape[0], upsample_out.shape[1], -1, upsample_out.shape[2]))
        output = k + x
        return output


class ResnetGenerator_ncsn(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect',  no_antialias=False, no_antialias_up=False, opt=None):
        assert(n_blocks >= 0)
        super(ResnetGenerator_ncsn, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        self.ngf = ngf
        self.use_sab = bool(getattr(opt, 'use_sab', False)) if opt is not None else False
        self.sab = SpatialAttention(ngf) if self.use_sab else None
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            if no_antialias:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)

                        ]
        self.model_res = nn.ModuleList()
        mult = 2 ** n_downsampling
        for i in range(n_blocks):

            self.model_res += [ResnetBlock_cond(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,temb_dim=4*ngf,z_dim=4*ngf)]

        model_upsample = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model_upsample += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model_upsample += [
                    Upsample(ngf * mult),

                    nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
        model_upsample += [nn.ReflectionPad2d(3)]
        model_upsample += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_upsample += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.model_upsample = nn.Sequential(*model_upsample)
        mapping_layers = [PixelNorm(),
                      nn.Linear(self.ngf*4, self.ngf*4),
                      nn.LeakyReLU(0.2)]
        for _ in range(opt.n_mlp):
            mapping_layers.append(nn.Linear(self.ngf*4, self.ngf*4))
            mapping_layers.append(nn.LeakyReLU(0.2))
        self.z_transform = nn.Sequential(*mapping_layers)
        modules_emb = []
        modules_emb += [nn.Linear(self.ngf,self.ngf*4)]

        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        modules_emb += [nn.Linear(self.ngf*4,self.ngf*4)]

        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        self.time_embed = nn.Sequential(*modules_emb)

        layers_per_down = 3 if no_antialias else 4
        self._d1_layer_id = 4 + layers_per_down - 1
        self._d2_layer_id = 4 + 2 * layers_per_down - 1
        self._up_split = 3 if no_antialias_up else 4

        self.skip_conv_1_fwd = nn.Conv2d(ngf * 2, ngf * 2, 1, bias=False)
        self.skip_conv_2_fwd = nn.Conv2d(ngf * 4, ngf * 4, 1, bias=False)
        self.skip_conv_1_rev = nn.Conv2d(ngf * 2, ngf * 2, 1, bias=False)
        self.skip_conv_2_rev = nn.Conv2d(ngf * 4, ngf * 4, 1, bias=False)
        nn.init.constant_(self.skip_conv_1_fwd.weight, 1e-5)
        nn.init.constant_(self.skip_conv_2_fwd.weight, 1e-5)
        nn.init.constant_(self.skip_conv_1_rev.weight, 1e-5)
        nn.init.constant_(self.skip_conv_2_rev.weight, 1e-5)

        self.dir_embed = nn.Embedding(2, ngf)
        with torch.no_grad():
            self.dir_embed.weight[0].uniform_(-0.5, 0.5)
            self.dir_embed.weight[1].copy_(-self.dir_embed.weight[0])
        self.dir_mlp = nn.Sequential(
            nn.Linear(ngf, ngf * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(ngf * 4, ngf * 4),
        )

    def forward(self, x, time_cond, z, layers=[], encode_only=False, direction=None):
        z_embed = self.z_transform(z)
        temb = get_timestep_embedding(time_cond, self.ngf)
        time_embed = self.time_embed(temb)
        dir_val = 0
        if direction is not None:
            if not isinstance(direction, torch.Tensor):
                dir_val = direction
                direction = torch.full((x.shape[0],), direction, dtype=torch.long, device=x.device)
            else:
                dir_val = direction[0].item()
            dir_code = self.dir_mlp(self.dir_embed(direction))
            time_embed = time_embed + dir_code
            z_embed = z_embed + dir_code
        if len(layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if self.use_sab and layer_id == 3:
                    feat = self.sab(feat)
                if layer_id in layers:
                    feats.append(feat)

            for layer_id, layer in enumerate(self.model_res):
                feat = layer(feat,time_embed,z_embed)
                if layer_id+len(self.model) in layers:
                    feats.append(feat)
                if layer_id+len(self.model) == layers[-1] and encode_only:
                    return feats
            return feat, feats
        else:
            out = x
            d1, d2 = None, None
            for layer_id, layer in enumerate(self.model):
                out = layer(out)
                if self.use_sab and layer_id == 3:
                    out = self.sab(out)
                if layer_id == self._d1_layer_id:
                    d1 = out
                elif layer_id == self._d2_layer_id:
                    d2 = out
            for layer in self.model_res:
                out = layer(out, time_embed, z_embed)
            if dir_val == 1:
                out = out + self.skip_conv_2_rev(d2)
            else:
                out = out + self.skip_conv_2_fwd(d2)
            for lid in range(self._up_split):
                out = self.model_upsample[lid](out)
            if dir_val == 1:
                out = out + self.skip_conv_1_rev(d1)
            else:
                out = out + self.skip_conv_1_fwd(d1)
            for lid in range(self._up_split, len(self.model_upsample)):
                out = self.model_upsample[lid](out)
            return out


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetBlock_cond(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,temb_dim,z_dim):
        super(ResnetBlock_cond, self).__init__()
        self.conv_block,self.adaptive,self.conv_fin = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,temb_dim,z_dim)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias,temb_dim,z_dim):

        self.conv_block = nn.ModuleList()
        self.conv_fin = nn.ModuleList()
        p = 0
        if padding_type == 'reflect':
            self.conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            self.conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        self.adaptive = AdaptiveLayer(dim,z_dim) 
        self.conv_fin += [nn.ReLU(True)]
        if use_dropout:
            self.conv_fin += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            self.conv_fin += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            self.conv_fin += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        self.conv_fin += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        self.Dense_time = nn.Linear(temb_dim, dim)

        nn.init.zeros_(self.Dense_time.bias)

        self.style = nn.Linear(z_dim, dim * 2)

        self.style.bias.data[:dim] = 1
        self.style.bias.data[dim:] = 0

        return self.conv_block,self.adaptive,self.conv_fin

    def forward(self, x,time_cond,z):

        time_input = self.Dense_time(time_cond)
        out = x
        for layer in self.conv_block:
            out = layer(out)
        out = out + time_input[:, :, None, None]
        out = self.adaptive(out,z)
        for layer in self.conv_fin:
            out = layer(out)
        out = x + out
        return out


def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net
