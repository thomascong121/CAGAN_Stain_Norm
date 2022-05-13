import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import Conv2d
import torchvision.models as models
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

'''
Some useful functions
'''
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('LayerNorm') != -1:
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('LayerNorm') != -1:
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('LayerNorm') != -1:
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('LayerNorm') != -1:
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt['lr_decay_iters'], gamma=0.1)
    elif opt['lr_policy'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=1e-5, patience=5)
    elif opt['lr_policy'] == 'linear':
        def lambda_rule(epoch):
            if epoch > opt['lr_decay_iters']:
                times = (epoch % opt['lr_decay_iters']) // 10 + 1
                return 0.65 ** times
            return
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt['lr_policy'])
    return scheduler


'''
function for define generator
'''
def define_G(opt=None):
    netG = None
    use_gpu = len(opt.run.opt_run['gpu_ids']) > 0

    if opt.model.opt_G['which_model_netG'] == 'twodecoder_unet':
        print('==> Unet+ResUnet')
        netG = TwoDecoderUnetGenerator(opt)
    elif opt.model.opt_G['which_model_netG'] == 'twodecoder_attentunet':
        print('==> Unet+ATTUnet')
        netG = TwoDecoderAttenUnetGenerator(opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' %
                                  opt.model.opt_G['which_model_netD'])
    if use_gpu:
        assert (torch.cuda.is_available())
        netG.cuda(opt.run.opt_run['gpu_ids'][0])
    init_weights(netG, init_type=opt.model.opt_G['init_type'])

    return netG


'''
function for define discriminator
'''
def define_D(opt=None):
    netD = None
    use_gpu = len(opt.run.opt_run['gpu_ids']) > 0

    if opt.model.opt_D['which_model_netD'] in ['basic', 'n_layers']:
        netD = NLayerDiscriminator(opt)
    elif opt.model.opt_D['which_model_netD'] == 'pixel':
        netD = PixelDiscriminator(opt)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  opt.model.opt_D['which_model_netD'])
    if use_gpu:
        assert (torch.cuda.is_available())
        netD.cuda(opt.run.opt_run['gpu_ids'][0])
    init_weights(netD, init_type=opt.model.opt_D['init_type'])
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)



'''
Define useful generator networks
'''
# Resnet
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc,
                 output_nc,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 n_blocks=6,
                 gpu_ids=[],
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim,
                 padding_type,
                 norm_layer,
                 use_dropout,
                 use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim,
                         padding_type,
                         norm_layer,
                         use_dropout,
                         use_bias):
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc,
                 output_nc,
                 num_downs,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc,
                 inner_nc,
                 input_nc=None,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetEncoderBlock(nn.Module):
    def __init__(self, input_nc,
                 output_nc,
                 kernel_size=4,
                 stride=2,
                 norm_layer=nn.BatchNorm2d,
                 num_layer=1,
                 use_dropout=False,
                 gpu_ids=[]):
        super(UnetEncoderBlock, self).__init__()
        # print('in ',input_nc, output_nc)
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        for _ in range(num_layer):
            model.append(Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias))
        model += [norm_layer(output_nc), nn.LeakyReLU(negative_slope=0.2)]
        if use_dropout:
            model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class UnetDecoderBlock(nn.Module):
    def __init__(self, input_nc,
                 output_nc,
                 norm_layer=nn.BatchNorm2d,
                 num_layer=1,
                 use_dropout=True,
                 gpu_ids=[]):
        super(UnetDecoderBlock, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # self.up = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(output_nc))
        self.model = None
        model = []
        for i in range(num_layer):
            if i == 0:
                model += [nn.Conv2d(output_nc * 2, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(output_nc),
                          nn.LeakyReLU(negative_slope=0.2)]
            else:
                model += [nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(output_nc),
                          nn.LeakyReLU(negative_slope=0.2)]
            self.model = nn.Sequential(*model)
        self.Dropout = use_dropout

    def forward(self, input, skip):
        input = self.up(input)
        if self.Dropout:
            input = nn.Dropout(0.5)(input)
        input = torch.cat([input, skip], 1)
        input = nn.ReLU()(input)
        if self.model:
            return self.model(input)
        return input


class UnetDecoderBlockRes(nn.Module):
    def __init__(self, input_nc,
                 output_nc,
                 norm_layer=nn.BatchNorm2d,
                 num_layer=2,
                 use_dropout=True,
                 gpu_ids=[]):
        super(UnetDecoderBlockRes, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # self.up = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(output_nc))
        model = []
        for i in range(num_layer):
            if i == 0:
                model.append(nn.Conv2d(output_nc * 2, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias))
            else:
                model.append(nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias))
            model.append(norm_layer(output_nc))
            if use_dropout:
                model.append(nn.Dropout(0.5))
            model.append(nn.LeakyReLU(negative_slope=0.2))
        self.model = nn.Sequential(*model)
        self.res = nn.Conv2d(output_nc * 2, output_nc, kernel_size=1, bias=False)

    def forward(self, input, skip):
        input = self.up(input)
        input_cat = torch.cat([skip, input], 1)
        decoded = self.model(input_cat)
        res = self.res(input_cat)
        res_con = res + decoded

        return nn.ReLU()(res_con)


class TwoDecoderUnetGenerator(nn.Module):
    def __init__(self, opt):
        super(TwoDecoderUnetGenerator, self).__init__()
        gpu_ids = opt.run.opt_run['gpu_ids']
        in_nc = opt.model.opt_G['input_nc']
        out_nc = opt.model.opt_G['output_nc']
        # construct Encoder
        self.input_encode = nn.Sequential(
            nn.Conv2d(in_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU())
        self.e1 = UnetEncoderBlock(64, 128, gpu_ids=gpu_ids, num_layer=0)
        self.e2 = UnetEncoderBlock(128, 256, gpu_ids=gpu_ids, num_layer=0)
        self.e3 = UnetEncoderBlock(256, 512, gpu_ids=gpu_ids, num_layer=0)
        # self.e4 = UnetEncoderBlock(512, 512, gpu_ids=gpu_ids, num_layer=0)
        # self.e5 = UnetEncoderBlock(512, 512, gpu_ids=gpu_ids, num_layer=0)
        # self.e6 = UnetEncoderBlock(512, 512, gpu_ids=gpu_ids, num_layer=0)
        self.bn = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        # construct Decoder1
        self.d1_1 = UnetDecoderBlock(1024, 512, gpu_ids=gpu_ids, num_layer=1)
        # self.d1_2 = UnetDecoderBlock(512, 512, gpu_ids=gpu_ids, num_layer=1)
        # self.d1_3 = UnetDecoderBlock(512, 512, gpu_ids=gpu_ids, num_layer=1)
        # self.d1_4 = UnetDecoderBlock(512, 512, gpu_ids=gpu_ids, num_layer=1, use_dropout=False)
        self.d1_5 = UnetDecoderBlock(512, 256, gpu_ids=gpu_ids, num_layer=1, use_dropout=False)
        self.d1_6 = UnetDecoderBlock(256, 128, gpu_ids=gpu_ids, num_layer=1, use_dropout=False)
        self.d1_7 = UnetDecoderBlock(128, 64, gpu_ids=gpu_ids, num_layer=1, use_dropout=False)
        self.d1_out = nn.ConvTranspose2d(64, out_nc, kernel_size=4, stride=2, padding=1)
        self.d1_out_img = nn.Sigmoid()
        # construct Decoder2
        self.d2_1 = UnetDecoderBlockRes(1024, 512, gpu_ids=gpu_ids)
        # self.d2_2 = UnetDecoderBlockRes(512, 512, gpu_ids=gpu_ids)
        # self.d2_3 = UnetDecoderBlockRes(512, 512, gpu_ids=gpu_ids)
        # self.d2_4 = UnetDecoderBlockRes(512, 512, gpu_ids=gpu_ids, use_dropout=False)
        self.d2_5 = UnetDecoderBlockRes(512, 256, gpu_ids=gpu_ids, use_dropout=False)
        self.d2_6 = UnetDecoderBlockRes(256, 128, gpu_ids=gpu_ids, use_dropout=False)
        self.d2_7 = UnetDecoderBlockRes(128, 64, gpu_ids=gpu_ids, use_dropout=False)
        self.d2_out = nn.ConvTranspose2d(64, out_nc, kernel_size=4, stride=2, padding=1)
        self.d2_out_img = nn.Sigmoid()

    def forward(self, input):
        # encoder branch
        x1 = self.input_encode(input)
        x2 = self.e1(x1)
        x3 = self.e2(x2)
        x6 = self.e3(x3)
        # x5 = self.e4(x4)
        # x6 = self.e5(x5)
        # x7 = self.e6(x6)
        # bottlenect
        bn = self.bn(x6)

        # decoder output 1
        d1_1 = self.d1_1(bn, x6)
        # d1_2 = self.d1_2(d1_1, x6)
        # d1_3 = self.d1_3(d1_2, x5)
        # d1_4 = self.d1_4(d1_3, x4)
        d1_5 = self.d1_5(d1_1, x3)
        d1_6 = self.d1_6(d1_5, x2)
        d1_7 = self.d1_7(d1_6, x1)
        d1_out = self.d1_out(d1_7)
        d1_out = self.d1_out_img(d1_out)

        # decoder output 1
        d2_1 = self.d2_1(bn, x6)
        # d2_2 = self.d2_2(d2_1, x6)
        # d2_3 = self.d2_3(d2_2, x5)
        # d2_4 = self.d2_4(d2_3, x4)
        d2_5 = self.d2_5(d2_1, x3)
        d2_6 = self.d2_6(d2_5, x2)
        d2_7 = self.d2_7(d2_6, x1)
        d2_out = self.d2_out(d2_7)
        d2_out = self.d2_out_img(d2_out)
        return d1_out, d2_out


class AttentionBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, int_channels):
        super(AttentionBlock, self).__init__()
        self.Wx = nn.Sequential(nn.Conv2d(in_channels_x, int_channels, kernel_size=1),
                                nn.BatchNorm2d(int_channels))
        self.Wg = nn.Sequential(nn.Conv2d(in_channels_g, int_channels, kernel_size=1),
                                nn.BatchNorm2d(int_channels))
        self.psi = nn.Sequential(nn.Conv2d(int_channels, 1, kernel_size=1),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())

    def forward(self, x, g):
        # apply the Wx to the skip connection
        x1 = self.Wx(x)
        # after applying Wg to the input, upsample to the size of the skip connection
        g1 = nn.functional.interpolate(self.Wg(g), x1.shape[2:], mode='bilinear', align_corners=False)
        out = self.psi(nn.ReLU()(x1 + g1))
        out = nn.Sigmoid()(out)
        return out * x


class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionBlock(out_channels, in_channels, int(out_channels / 2))
        self.conv_bn1 = nn.Sequential(nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1),
                                      nn.BatchNorm2d(out_channels))
        self.conv_bn2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1),
                                      nn.BatchNorm2d(out_channels))

    def forward(self, x, x_skip):
        # note : x_skip is the skip connection and x is the input from the previous block
        # apply the attention block to the skip connection, using x as context
        x_attention = self.attention(x_skip, x)
        # upsample x to have th same size as the attention map
        x = nn.functional.interpolate(x, x_skip.shape[2:], mode='bilinear', align_corners=False)
        # stack their channels to feed to both convolution blocks
        x = torch.cat((x_attention, x), dim=1)
        x = self.conv_bn1(x)
        return self.conv_bn2(x)


class TwoDecoderAttenUnetGenerator(nn.Module):
    def __init__(self, opt):
        super(TwoDecoderAttenUnetGenerator, self).__init__()
        gpu_ids = opt.run.opt_run['gpu_ids']
        in_nc = opt.model.opt_G['input_nc']
        out_nc = opt.model.opt_G['output_nc']
        # construct Encoder
        self.input_encode = nn.Sequential(
            nn.Conv2d(in_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU())
        self.e1 = UnetEncoderBlock(64, 128, gpu_ids=gpu_ids, num_layer=0)
        self.e2 = UnetEncoderBlock(128, 256, gpu_ids=gpu_ids, num_layer=0)
        self.e3 = UnetEncoderBlock(256, 512, gpu_ids=gpu_ids, num_layer=0)
        # self.e4 = UnetEncoderBlock(512, 512, gpu_ids=gpu_ids, num_layer=0)
        # self.e5 = UnetEncoderBlock(512, 512, gpu_ids=gpu_ids, num_layer=0)
        # self.e6 = UnetEncoderBlock(512, 512, gpu_ids=gpu_ids, num_layer=0)
        self.bn = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        # construct Decoder1
        # self.d1_1 = UnetDecoderBlock(1024, 512, gpu_ids=gpu_ids, num_layer=1)
        # self.d1_2 = UnetDecoderBlock(512, 512, gpu_ids=gpu_ids, num_layer=1)
        self.d1_3 = UnetDecoderBlock(1024, 512, gpu_ids=gpu_ids, num_layer=1)
        # self.d1_4 = UnetDecoderBlock(512, 512, gpu_ids=gpu_ids, num_layer=1, use_dropout=False)
        self.d1_5 = UnetDecoderBlock(512, 256, gpu_ids=gpu_ids, num_layer=1, use_dropout=False)
        self.d1_6 = UnetDecoderBlock(256, 128, gpu_ids=gpu_ids, num_layer=1, use_dropout=False)
        self.d1_7 = UnetDecoderBlock(128, 64, gpu_ids=gpu_ids, num_layer=1, use_dropout=False)
        self.d1_out = nn.ConvTranspose2d(64, out_nc, kernel_size=4, stride=2, padding=1)
        self.d1_out_img = nn.Sigmoid()
        # construct Decoder2
        # self.d2_1 = AttentionUpBlock(1024, 512)
        # self.d2_2 = AttentionUpBlock(512, 512)
        self.d2_3 = AttentionUpBlock(1024, 512)
        # self.d2_4 = AttentionUpBlock(512, 512)
        self.d2_5 = AttentionUpBlock(512, 256)
        self.d2_6 = AttentionUpBlock(256, 128)
        self.d2_7 = AttentionUpBlock(128, 64)
        self.d2_out = nn.ConvTranspose2d(64, out_nc, kernel_size=4, stride=2, padding=1)
        self.d2_out_img = nn.Sigmoid()

    def forward(self, input):
        # encoder branch
        x1 = self.input_encode(input)
        x2 = self.e1(x1)
        x3 = self.e2(x2)
        x4 = self.e3(x3)
        # x5 = self.e4(x4)
        # x6 = self.e5(x5)
        # x7 = self.e6(x6)
        # bottlenect
        bn = self.bn(x4)

        # decoder output 1
        # d1_1 = self.d1_1(bn, x7)
        # d1_2 = self.d1_2(bn, x6)
        d1_3 = self.d1_3(bn, x4)
        # d1_4 = self.d1_4(d1_3, x4)
        d1_5 = self.d1_5(d1_3, x3)
        d1_6 = self.d1_6(d1_5, x2)
        d1_7 = self.d1_7(d1_6, x1)
        d1_out = self.d1_out(d1_7)
        d1_out = self.d1_out_img(d1_out)
        # decoder output 1
        # d2_1 = self.d2_1(bn, x7)
        # d2_2 = self.d2_2(bn, x6)
        d2_3 = self.d2_3(bn, x4)
        # d2_4 = self.d2_4(d2_3, x4)
        d2_5 = self.d2_5(d2_3, x3)
        d2_6 = self.d2_6(d2_5, x2)
        d2_7 = self.d2_7(d2_6, x1)
        d2_out = self.d2_out(d2_7)
        d2_out = self.d2_out_img(d2_out)
        return d1_out, d2_out

'''
Define useful discriminator networks
'''
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, opt):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = opt.run.opt_run['gpu_ids']
        norm_layer = nn.BatchNorm2d
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(opt.model.opt_D['input_nc'], opt.model.opt_D['ndf'], kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, opt.model.opt_D['n_layers']):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(opt.model.opt_D['ndf'] * nf_mult_prev, opt.model.opt_D['ndf'] * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(opt.model.opt_D['ndf'] * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** opt.model.opt_D['n_layers'], 8)
        sequence += [
            nn.Conv2d(opt.model.opt_D['ndf'] * nf_mult_prev, opt.model.opt_D['ndf'] * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(opt.model.opt_D['ndf'] * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(opt.model.opt_D['ndf'] * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if opt.model.opt_D['use_sigmoid']:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = opt.run.opt_run['gpu_ids']
        norm_layer=nn.BatchNorm2d
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(opt.model.opt_D['input_nc'], opt.model.opt_D['ndf'], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(opt.model.opt_D['ndf'], opt.model.opt_D['ndf'] * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            opt.norm_layer(opt.model.opt_D['ndf'] * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(opt.model.opt_D['ndf'] * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if opt.model.opt_D['use_sigmoid']:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.net, input, self.gpu_ids)
        else:
            return self.net(input)


'''
Classifiers
'''
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class pretrained_classifier:
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0
        if model_name == "resnet":
            """ Resnet152
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 1024
        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 1024

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg19_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 1024

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 1024

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 1024

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        params_to_update = model_ft.parameters()
        # # print("Params to learn:")
        # if feature_extract:
        #     params_to_update = []
        #     for name, param in model_ft.named_parameters():
        #         if param.requires_grad == True:
        #             params_to_update.append(param)
        #             # print("\t",name)
        # else:
        #     for name, param in model_ft.named_parameters():
        #         if param.requires_grad == True:
        #             print("\t", name)

        return model_ft
