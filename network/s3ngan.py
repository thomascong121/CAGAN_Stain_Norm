import math
from piq import ssim, psnr
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from network.network import *
from utils.loss import *
from utils.util import *

def adjust_learning_rate(optimizer, epoch, args, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args['warmup'] and epoch < args['warmup_epochs']:
        lr = args.lr / args.warmup_epochs * (epoch + 1)
    elif args['lr_policy'] == 'cosine':
        eta_min = lr * (args['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args['n_epoch'])) / 2
    elif args['lr_policy'] == 'step':
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)
    else:
        print('===> Use Other Learning adjust')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class twodecoderGANModel(nn.Module):
    '''
    Basic Idea:
        source to target = strong augmented to original
        consistency regularization = f(source) similar with f(target) in colours
        1. use transformer layer to replace content loss
        2. use colour histogram to replace reference target image
    '''

    def name(self):
        return 'transformerGANModel'

    def __init__(self, cfg):
        super(twodecoderGANModel, self).__init__()
        self.opt = cfg
        self.gpu_ids = self.opt.run.opt_run['gpu_ids']
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir_name = os.path.join(self.opt.run.opt_run['checkpoints_dir'], self.opt.dataset.opt_dataset['name'])
        if not os.path.exists(self.save_dir_name):
            os.makedirs(self.save_dir_name)
        log_dir = os.path.join(self.save_dir_name, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)
        if not os.path.exists(self.save_dir_name):
            os.mkdir(self.save_dir_name)
        self.netG = define_G(self.opt)
        self.netD = define_D(self.opt)
        print_network(self.netD)
        print_network(self.netG)

        if self.opt.run.opt_run['continue_train']:
            which_epoch = self.opt.run.opt_run['which_epoch']
            self.load_network(self.netG, 'G', which_epoch)
            self.load_network(self.netD, 'D', which_epoch)
            print('Both G and D of epoch [%d] is loaded ' % which_epoch)

        # define loss functions
        self.criterionMSE = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(use_lsgan=self.opt.model.opt_D['use_sigmoid'], target_real_label=0.8, tensor=self.Tensor)
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionContent = content_loss(self.opt.run.opt_run['gpu_ids'])
        self.criterionHist = histogram_loss(self.opt)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.opt.run.opt_run['lr_G'],
                                            betas=(0.9, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.opt.run.opt_run['lr_D'],
                                            betas=(0.9, 0.999))

        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.schedulers = [get_scheduler(optimizer, self.opt.run.opt_run) for optimizer in self.optimizers]

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt['lr_policy'] == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

    def set_input(self, input, batch_index=None):
        target_gray, target_rgb = input['target']
        source_gray, source_rgb = input['source']

        self.target_gray = target_gray.to(self.gpu_ids[0], dtype=torch.float) if len(self.gpu_ids) > 0 else target_gray
        self.target_rgb = target_rgb.to(self.gpu_ids[0], dtype=torch.float) if len(self.gpu_ids) > 0 else target_rgb
        self.source_gray = source_gray.to(self.gpu_ids[0], dtype=torch.float) if len(self.gpu_ids) > 0 else source_gray
        self.source_rgb = source_rgb.to(self.gpu_ids[0], dtype=torch.float) if len(self.gpu_ids) > 0 else source_rgb

        self.inputs = torch.cat([self.target_gray, self.source_gray], dim=0)
        self.half = len(target_gray)

    def forward(self):
        self.fake_RGB_1, self.fake_RGB_2 = self.netG(self.inputs)
        self.fake_target_rgb_1, self.fake_source_rgb_1 = torch.split(self.fake_RGB_1, [self.half, self.half], dim=0)
        self.fake_target_rgb_2, self.fake_source_rgb_2 = torch.split(self.fake_RGB_2, [self.half, self.half], dim=0)

    def backward_D(self):
        """
        D update strategy:
            target (labelled): real + fake
            source (unlabelled): fake
        """
        # Fake; stop backdrop to the generator by detaching fake_B;
        fake_pair_1 = torch.cat((self.target_gray, self.fake_target_rgb_1), 1)
        fake_pair_2 = torch.cat((self.target_gray, self.fake_target_rgb_2), 1)
        real_pair = torch.cat((self.target_gray, self.target_rgb), 1)

        pred_fake_1 = self.netD(fake_pair_1.detach())
        pred_fake_2 = self.netD(fake_pair_2.detach())
        pred_real = self.netD(real_pair.detach())

        self.loss_D_fake_1 = self.criterionGAN(pred_fake_1, False, self.opt)
        self.loss_D_fake_2 = self.criterionGAN(pred_fake_2, False, self.opt)
        self.loss_D_real = self.criterionGAN(pred_real, True, self.opt)

        self.loss_D = self.loss_D_real + self.loss_D_fake_1 + self.loss_D_fake_2
        self.loss_D.backward()

    def backward_G(self):
        '''
        Applying adversarial loss
        '''
        fake_pair_1 = torch.cat((self.inputs, self.fake_RGB_1), 1)
        fake_pair_2 = torch.cat((self.inputs, self.fake_RGB_2), 1)

        pred_fake_1 = self.netD(fake_pair_1)
        pred_fake_2 = self.netD(fake_pair_2)
        loss_G_GAN_1 = self.criterionGAN(pred_fake_1, True, self.opt)
        loss_G_GAN_2 = self.criterionGAN(pred_fake_2, True, self.opt)
        self.loss_gan = loss_G_GAN_1 + loss_G_GAN_2

        '''
        Applying content loss
        '''
        loss_Content_1 = self.criterionContent(self.fake_target_rgb_1, self.target_rgb) * self.opt.run.opt_run['lambda_content']
        loss_Content_2 = self.criterionContent(self.fake_target_rgb_2, self.target_rgb) * self.opt.run.opt_run['lambda_content']
        self.loss_content = loss_Content_1 + loss_Content_2
        '''
        Applying supervised loss
        '''
        loss_l1_1 = self.criterionIdt(self.fake_target_rgb_1, self.target_rgb) * self.opt.run.opt_run['lambda_l1']
        loss_l1_2 = self.criterionIdt(self.fake_target_rgb_2, self.target_rgb) * self.opt.run.opt_run['lambda_l1']
        self.loss_l1 = loss_l1_1 + loss_l1_2
        '''
        Applying Consistency loss
        '''
        self.loss_consist = self.criterionIdt(self.fake_source_rgb_1, self.fake_source_rgb_2) * self.opt.run.opt_run['lambda_l1']
        '''
        Applying Histogram loss 
        '''
        loss_histo_1 = self.criterionHist(self.fake_source_rgb_1, self.target_rgb)
        loss_histo_2 = self.criterionHist(self.fake_source_rgb_2, self.target_rgb)
        self.loss_histo = loss_histo_1 + loss_histo_2

        self.loss_G = self.loss_gan + self.loss_content + self.loss_l1 + self.loss_consist + self.loss_histo
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights


    def test(self, test_loader, stage='test', save=False):
        """
        call to generate ssim psnr
        """
        self.netG.eval()
        avg_psnr = 0
        avg_ssim = 0
        steps = 0
        test_root = '%s/%s/normalised'%(str(self.opt.run.opt_run['checkpoints_dir']),  self.opt.dataset.opt_dataset['name'])
        if not os.path.exists(test_root):
            os.makedirs(test_root)
        out_img_pth = os.path.join(test_root, str(self.opt.run.opt_run['which_epoch']), stage)
        if not os.path.exists(out_img_pth):
            os.makedirs(out_img_pth)
        with torch.no_grad():
            count = 0
            for i, (input, target, label) in tqdm(enumerate(test_loader)):
                label = label.to(self.gpu_ids[0])
                if len(self.gpu_ids) > 0:
                    input, target = input.to(self.gpu_ids[0]), target.to(self.gpu_ids[0])
                prediction1, prediction2 = self.netG(input.float())
                prediction = (prediction1 + prediction2) / 2
                if save:
                    for j in range(len(prediction)):
                        prediction_save = tensor2im(prediction[j])
                        prediction_save = Image.fromarray(prediction_save)
                        save_pth = out_img_pth + '/%d_%d.png' % (count, label[j])
                        prediction_save.save(save_pth)
                        count += 1

                avg_psnr += psnr(prediction, target, convert_to_greyscale=True)
                avg_ssim += ssim(transforms.Grayscale(1)(prediction), transforms.Grayscale(1)(target))
                steps += 1

        self.netG.train()
        epoch_psnr = avg_psnr / steps
        epoch_ssim = avg_ssim / steps
        print("===> Avg. PSNR: {:.4f} dB, Avg. SSIM: {:.4f}".format(epoch_psnr, epoch_ssim))
        return epoch_psnr, epoch_ssim

    def get_current_errors(self):
        ret_errors = OrderedDict([('D', self.loss_D.item()),
                                  ('G', self.loss_G.item()),
                                  ('G_gan', self.loss_gan.item()),
                                  ('G_cont', self.loss_content.item()),
                                  ('G_l1', self.loss_l1.item()),
                                  ('G_consis', self.loss_consist.item()),
                                  ('G_histo', self.loss_histo.item()), ])
        return ret_errors

    def get_current_visuals(self, epoch):
        target_gray = tensor2im(self.target_gray[0])
        target_rgb = tensor2im(self.target_rgb[0])
        source_gray = tensor2im(self.source_gray[0])
        source_rgb = tensor2im(self.source_rgb[0])
        fake_target_1 = tensor2im(self.fake_target_rgb_1[0])
        fake_target_2 = tensor2im(self.fake_target_rgb_2[0])
        fake_source_1 = tensor2im(self.fake_source_rgb_1[0])
        fake_source_2 = tensor2im(self.fake_source_rgb_2[0])
        ret_visuals = OrderedDict([('target_gray', target_gray), ('target_rgb', target_rgb),
                                   ('source_gray', source_gray), ('source_rgb', source_rgb),
                                   ('fake_target_1', fake_target_1), ('fake_target_2', fake_target_2),
                                   ('fake_source_1', fake_source_1), ('fake_source_2', fake_source_2)])
        image_save_path = os.path.join(self.save_dir_name, 'images')
        if not os.path.exists(image_save_path):
            os.mkdir(image_save_path)
        image_save_path = os.path.join(image_save_path, str(epoch))
        if not os.path.exists(image_save_path):
            os.mkdir(image_save_path)
        for img_name in ret_visuals:
            img_np = ret_visuals[img_name]
            img_pil = Image.fromarray(img_np)
            img_pil.save(os.path.join(image_save_path, img_name + '.png'))

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        model_save_path = os.path.join(self.save_dir_name, 'models')
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(model_save_path, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def save(self, label):
        self.save_network(self.netG, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD, 'D_A', label, self.gpu_ids)
        print('models saved as [%s] G_A[%d], D_A[%s]' % (self.save_dir_name, label, label))
        self.writer.add_images('Enoder1_target', (self.fake_target_rgb_1 + 1) * 127.5, label)
        self.writer.add_images('Enoder2_target', (self.fake_target_rgb_2 + 1) * 127.5, label)
        self.writer.add_images('Enoder1_source', (self.fake_source_rgb_1 + 1) * 127.5, label)
        self.writer.add_images('Enoder2_source', (self.fake_source_rgb_2 + 1) * 127.5, label)
        self.writer.add_scalar('G_loss', self.loss_D.item(), label)
        self.writer.add_scalar('D_loss', self.loss_G.item(), label)

    def load_network(self, network, network_label, epoch_label):
        save_filename = 'models/%s_net_%s_A.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir_name, save_filename)
        network.load_state_dict(torch.load(save_path))

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad