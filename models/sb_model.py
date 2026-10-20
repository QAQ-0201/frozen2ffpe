import numpy as np
import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
from .ssim import SSIMLoss
import util.util as util

class SBModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--mode', type=str, default="sb", choices='(FastCUT, fastcut, sb)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=0.35, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_SB', type=float, default=0.1, help='weight for SB loss')
        parser.add_argument('--lambda_idt', type=float, default=0.5, help='weight for identity L1 loss at t=0, z=0')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use identity loss: L1(G(Y,t=0,z=0), Y)')
        parser.add_argument('--nce_layers', type=str, default='4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--lmda', type=float, default=0.1)
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--lambda_GAN_aux', type=float, default=0.1, help='weight for CONCH auxiliary GAN loss')
        parser.add_argument('--lambda_GAN_B', type=float, default=0.2, help='weight for reverse B->A GAN loss')
        parser.add_argument('--sb_clamp', type=float, default=-10.0, help='clamp floor for SB loss (prevent divergence)')
        parser.add_argument('--lr_D_aux', type=float, default=0.0001, help='learning rate for CONCH discriminator decoder')
        parser.add_argument('--max_grad_norm', type=float, default=10.0, help='gradient clipping max norm (0=disable)')
        parser.add_argument('--lambda_idt_lpips', type=float, default=1.0, help='weight for identity LPIPS loss (0=disable)')
        parser.add_argument('--lambda_cycle', type=float, default=1.0, help='weight for cycle consistency L1 loss')
        parser.add_argument('--lambda_cycle_lpips', type=float, default=2.0, help='weight for cycle consistency LPIPS loss')
        parser.add_argument('--lambda_cycle_ssim', type=float, default=0.0, help='weight for cycle-reconstruction SSIM loss (structural anti-hallucination; 0=off)')
        parser.add_argument('--use_sab', action='store_true')
        parser.add_argument('--self_regularization', type=float, default=0.0)


        parser.add_argument('--use_ema', action='store_true', help='maintain EMA of netG for eval/best')
        parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay for netG')

        parser.set_defaults(pool_size=0)

        opt, _ = parser.parse_known_args()


        if opt.mode.lower() == "sb":
            parser.set_defaults(nce_idt=True, lambda_NCE=0.35,
                                ngf=128, n_blocks=18, n_layers_D=6)
        elif opt.mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)


        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE','SB', 'SB_raw', 'E_self', 'E_cross', 'ET_XY', 'SR', 'ssim']
        self.visual_names = ['real_A','real_A_noisy', 'fake_B', 'real_B']
        if self.opt.phase == 'test':
            self.visual_names = ['real']
            for NFE in range(self.opt.num_timesteps):
                fake_name = 'fake_' + str(NFE+1)
                self.visual_names.append(fake_name)
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['idt', 'idt_A', 'NCE_Y']
            self.visual_names += ['idt_B', 'idt_A']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D','E']
        else:
            self.model_names = ['G']


        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        net_g = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
        nn.init.constant_(net_g.skip_conv_1_fwd.weight, 1e-5)
        nn.init.constant_(net_g.skip_conv_2_fwd.weight, 1e-5)
        nn.init.constant_(net_g.skip_conv_1_rev.weight, 1e-5)
        nn.init.constant_(net_g.skip_conv_2_rev.weight, 1e-5)
        if hasattr(net_g, 'dir_mlp'):
            nn.init.xavier_normal_(net_g.dir_mlp[0].weight, gain=0.2)
            nn.init.zeros_(net_g.dir_mlp[0].bias)
            nn.init.xavier_normal_(net_g.dir_mlp[2].weight, gain=0.2)
            nn.init.zeros_(net_g.dir_mlp[2].bias)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netE = networks.define_D(opt.output_nc*4, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            for module in self.netE.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.utils.spectral_norm(module)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            _needs_lpips = (opt.lambda_idt_lpips > 0.0 or
                            (opt.lambda_cycle > 0.0 and opt.lambda_cycle_lpips > 0.0))
            if _needs_lpips:
                import lpips
                self.criterionLPIPS = lpips.LPIPS(net='vgg').to(self.device)
                self.criterionLPIPS.requires_grad_(False)
            if opt.lambda_cycle_ssim > 0.0:
                self.criterionSSIM = SSIMLoss(channels=opt.output_nc).to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)

            if opt.lambda_GAN_aux > 0.0:
                self.model_names.append('D_aux')
                self.loss_names += ['G_GAN_aux', 'D_aux', 'D_aux_real', 'D_aux_fake']
                import vision_aided_loss
                self.netD_aux = vision_aided_loss.Discriminator(
                    cv_type='conch',
                    output_type='conv_multi_level',
                    loss_type='multilevel_sigmoid',
                    diffaug=True,
                    device=self.device
                ).to(self.device)
                self.netD_aux.cv_ensemble.requires_grad_(False)
                self.optimizer_D_aux = torch.optim.Adam(
                    self.netD_aux.decoder.parameters(),
                    lr=opt.lr_D_aux,
                    betas=(opt.beta1, opt.beta2)
                )
                self.optimizers.append(self.optimizer_D_aux)

            if opt.lambda_cycle > 0.0:
                self.model_names += ['D_A']
                self.loss_names += ['cycle_A', 'cycle_B', 'G_GAN_B', 'D_A', 'D_A_real', 'D_A_fake']
                self.visual_names += ['fake_A', 'rec_A']
                self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                                opt.normD, opt.init_type, opt.init_gain,
                                                opt.no_antialias, self.gpu_ids, opt)
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr,
                                                      betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_D_A)


        self.use_ema = getattr(opt, 'use_ema', False)
        if self.use_ema and self.isTrain:
            import copy
            net_g_src = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
            self.netG_ema = copy.deepcopy(net_g_src).eval()
            for p in self.netG_ema.parameters():
                p.requires_grad_(False)
            self.ema_decay = opt.ema_decay

    def data_dependent_initialize(self, data,data2):
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data,data2)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()
        if self.opt.isTrain:
            if self.opt.lambda_GAN_aux > 0.0:
                net_aux = self.netD_aux.module if isinstance(self.netD_aux, nn.DataParallel) else self.netD_aux
                self.set_requires_grad(net_aux.decoder, False)
            self.compute_G_loss().backward()
            if self.opt.lambda_GAN_aux > 0.0:
                self.set_requires_grad(net_aux.decoder, True)
            self.compute_D_loss().backward()
            self.compute_E_loss().backward()
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def _compute_sb_trajectory(self):
        tau = self.opt.tau
        T = self.opt.num_timesteps
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        times = torch.tensor(times).float().to(self.device)
        self.times = times
        time_idx = (torch.randint(T, size=[1]).to(self.device) * torch.ones(size=[1]).to(self.device)).long()
        self.time_idx = time_idx

        with torch.no_grad():
            self.netG.eval()
            for t in range(self.time_idx.int().item() + 1):
                if t > 0:
                    delta = times[t] - times[t-1]
                    denom = times[-1] - times[t-1]
                    inter = (delta / denom).reshape(-1, 1, 1, 1)
                    scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                Xt = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.device)
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.device)).long()
                z = torch.randn(size=[self.real_A.shape[0], 4*self.opt.ngf]).to(self.device)
                Xt_1 = self.netG(Xt, time_idx, z, direction=0)

                Xt2 = self.real_A2 if (t == 0) else (1-inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2).to(self.device)
                z2 = torch.randn(size=[self.real_A.shape[0], 4*self.opt.ngf]).to(self.device)
                Xt_12 = self.netG(Xt2, time_idx, z2, direction=0)

            self.real_A_noisy = Xt.detach()
            self.real_A_noisy2 = Xt2.detach()

    def _freeze_discriminators(self):
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        if self.opt.lambda_GAN_aux > 0.0:
            net_aux = self.netD_aux.module if isinstance(self.netD_aux, nn.DataParallel) else self.netD_aux
            self.set_requires_grad(net_aux.decoder, False)
        if self.opt.lambda_cycle > 0.0:
            self.set_requires_grad(self.netD_A, False)

    def optimize_parameters(self):
        self.flipped_for_equivariance = False
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        self._compute_sb_trajectory()
        self.netG.train()

        self.loss_idt = 0.0
        self.loss_idt_A = 0.0
        self.loss_NCE_Y = 0.0
        self.loss_G_GAN_B = 0.0
        self.loss_ssim = 0.0
        self.loss_SB_raw = 0.0
        self.loss_E_self = 0.0
        self.loss_E_cross = 0.0
        self.loss_ET_XY = 0.0
        self.loss_SR = 0.0

        bs = self.real_A.size(0)


        if self.opt.lambda_cycle > 0.0:
            self._freeze_discriminators()
            self.optimizer_G.zero_grad()

            z1 = torch.randn(size=[bs, 4*self.opt.ngf]).to(self.device)
            fake_B_cyc = self.netG(self.real_A_noisy, self.time_idx, z1, direction=0)

            z2 = torch.randn(size=[bs, 4*self.opt.ngf]).to(self.device)
            rec_A = self.netG(fake_B_cyc, self.time_idx, z2, direction=1)

            z3 = torch.randn(size=[bs, 4*self.opt.ngf]).to(self.device)
            fake_A = self.netG(self.real_B, self.time_idx, z3, direction=1)
            z4 = torch.randn(size=[bs, 4*self.opt.ngf]).to(self.device)
            rec_B = self.netG(fake_A, self.time_idx, z4, direction=0)

            self.loss_cycle_A = self.criterionIdt(rec_A, self.real_A) * self.opt.lambda_cycle
            self.loss_cycle_B = self.criterionIdt(rec_B, self.real_B) * self.opt.lambda_cycle
            if self.opt.lambda_cycle_lpips > 0.0:
                self.loss_cycle_A += self.criterionLPIPS(rec_A, self.real_A).mean() * self.opt.lambda_cycle_lpips
                self.loss_cycle_B += self.criterionLPIPS(rec_B, self.real_B).mean() * self.opt.lambda_cycle_lpips
            if self.opt.lambda_cycle_ssim > 0.0:

                ssim_a = self.criterionSSIM(rec_A, self.real_A)
                ssim_b = self.criterionSSIM(rec_B, self.real_B)
                self.loss_cycle_A += ssim_a * self.opt.lambda_cycle_ssim
                self.loss_cycle_B += ssim_b * self.opt.lambda_cycle_ssim
                self.loss_ssim = 0.5 * (ssim_a.item() + ssim_b.item())

            loss_step1 = self.loss_cycle_A + self.loss_cycle_B
            loss_step1.backward()
            if self.opt.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt.max_grad_norm)
            self.optimizer_G.step()

            self.fake_A = fake_A.detach()
            self.rec_A = rec_A.detach()


        z_in = torch.randn(size=[bs, 4*self.opt.ngf]).to(self.device)
        self.fake_B = self.netG(self.real_A_noisy, self.time_idx, z_in, direction=0)
        z_in2 = torch.randn(size=[bs, 4*self.opt.ngf]).to(self.device)
        self.fake_B2 = self.netG(self.real_A_noisy2, self.time_idx, z_in2, direction=0)


        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()


        if self.opt.lambda_GAN_aux > 0.0:
            net_aux = self.netD_aux.module if isinstance(self.netD_aux, nn.DataParallel) else self.netD_aux
            self.set_requires_grad(net_aux.decoder, True)
            self.optimizer_D_aux.zero_grad()
            self.loss_D_aux = self.compute_D_aux_loss()
            self.loss_D_aux.backward()
            self.optimizer_D_aux.step()


        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        if self.opt.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.netE.parameters(), self.opt.max_grad_norm)
        self.optimizer_E.step()


        self._freeze_discriminators()
        self.optimizer_G.zero_grad()
        if self.opt.lambda_NCE > 0.0 and self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()

        self.loss_G = self.compute_GAN_NCE_SB_loss()
        self.loss_G.backward()
        if self.opt.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt.max_grad_norm)
        self.optimizer_G.step()
        if self.opt.lambda_NCE > 0.0 and self.opt.netF == 'mlp_sample':
            if self.opt.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.netF.parameters(), self.opt.max_grad_norm)
            self.optimizer_F.step()


        if self.opt.nce_idt and self.opt.lambda_idt > 0.0:
            self._freeze_discriminators()
            self.optimizer_G.zero_grad()
            if self.opt.lambda_NCE > 0.0 and self.opt.netF == 'mlp_sample':
                self.optimizer_F.zero_grad()

            z_zero = torch.zeros(size=[bs, 4*self.opt.ngf]).to(self.device)
            t_zero = torch.zeros(size=[bs]).long().to(self.device)
            self.idt_B = self.netG(self.real_B, t_zero, z_zero, direction=0)
            self.idt_A = self.netG(self.real_A, t_zero, z_zero, direction=1)

            self.loss_idt = self.criterionIdt(self.idt_B, self.real_B) * self.opt.lambda_idt
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_A) * self.opt.lambda_idt
            if self.opt.lambda_idt_lpips > 0.0:
                self.loss_idt += self.criterionLPIPS(self.idt_B, self.real_B).mean() * self.opt.lambda_idt_lpips
                self.loss_idt_A += self.criterionLPIPS(self.idt_A, self.real_A).mean() * self.opt.lambda_idt_lpips

            if self.opt.lambda_NCE > 0.0 and self.opt.nce_idt:
                self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            else:
                self.loss_NCE_Y = 0.0

            loss_step3 = self.loss_idt + self.loss_idt_A
            if self.opt.lambda_NCE > 0.0 and self.opt.nce_idt:
                loss_step3 = loss_step3 + self.opt.lambda_NCE * self.loss_NCE_Y
            loss_step3.backward()
            if self.opt.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt.max_grad_norm)
            self.optimizer_G.step()
            if self.opt.lambda_NCE > 0.0 and self.opt.netF == 'mlp_sample':
                if self.opt.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.netF.parameters(), self.opt.max_grad_norm)
                self.optimizer_F.step()


        if self.opt.lambda_cycle > 0.0:
            self._freeze_discriminators()
            self.optimizer_G.zero_grad()

            z_b = torch.randn(size=[bs, 4*self.opt.ngf]).to(self.device)
            fake_A_gan = self.netG(self.real_B, self.time_idx, z_b, direction=1)
            pred_fake_A = self.netD_A(fake_A_gan, self.time_idx)
            self.loss_G_GAN_B = self.criterionGAN(pred_fake_A, True).mean() * self.opt.lambda_GAN_B

            self.loss_G_GAN_B.backward()
            if self.opt.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt.max_grad_norm)
            self.optimizer_G.step()
            self.fake_A = fake_A_gan.detach()

            self.set_requires_grad(self.netD_A, True)
            self.optimizer_D_A.zero_grad()
            self.loss_D_A_total = self.compute_D_A_loss()
            self.loss_D_A_total.backward()
            self.optimizer_D_A.step()


        self.update_ema()

    def set_input(self, input,input2=None):

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if input2 is not None:
            self.real_A2 = input2['A' if AtoB else 'B'].to(self.device)
            self.real_B2 = input2['B' if AtoB else 'A'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.flipped_for_equivariance = False
        tau = self.opt.tau
        T = self.opt.num_timesteps
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1),times])
        times = torch.tensor(times).float().to(self.device)
        self.times = times
        bs =  self.real_A.size(0)
        time_idx = (torch.randint(T, size=[1]).to(self.device) * torch.ones(size=[1]).to(self.device)).long()
        self.time_idx = time_idx
        self.timestep     = times[time_idx]

        with torch.no_grad():
            self.netG.eval()
            for t in range(self.time_idx.int().item()+1):

                if t > 0:
                    delta = times[t] - times[t-1]
                    denom = times[-1] - times[t-1]
                    inter = (delta / denom).reshape(-1,1,1,1)
                    scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
                Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                Xt_1     = self.netG(Xt, time_idx, z, direction=0)

                Xt2       = self.real_A2 if (t == 0) else (1-inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2).to(self.real_A.device)
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                Xt_12    = self.netG(Xt2, time_idx, z, direction=0)

            self.real_A_noisy = Xt.detach()
            self.real_A_noisy2 = Xt2.detach()

        z_in = torch.randn(size=[bs, 4*self.opt.ngf]).to(self.real_A.device)
        self.fake_B = self.netG(self.real_A_noisy, self.time_idx, z_in, direction=0)

        z_in2 = torch.randn(size=[bs, 4*self.opt.ngf]).to(self.real_A.device)
        self.fake_B2 = self.netG(self.real_A_noisy2, self.time_idx, z_in2, direction=0)

        if self.opt.nce_idt and self.opt.isTrain:
            z_zero = torch.zeros(size=[bs, 4*self.opt.ngf]).to(self.real_A.device)
            t_zero = torch.zeros(size=[bs]).long().to(self.real_A.device)
            self.idt_B = self.netG(self.real_B, t_zero, z_zero, direction=0)
            self.idt_A = self.netG(self.real_A, t_zero, z_zero, direction=1)

        if getattr(self.opt, 'lambda_cycle', 0) > 0.0 and self.opt.isTrain:
            z_rev = torch.randn(size=[bs, 4*self.opt.ngf]).to(self.real_A.device)
            self.rec_A = self.netG(self.fake_B, self.time_idx, z_rev, direction=1)
            z_b = torch.randn(size=[bs, 4*self.opt.ngf]).to(self.real_A.device)
            self.fake_A = self.netG(self.real_B, self.time_idx, z_b, direction=1)
            z_rev_b = torch.randn(size=[bs, 4*self.opt.ngf]).to(self.real_A.device)
            self.rec_B = self.netG(self.fake_A, self.time_idx, z_rev_b, direction=0)

        if self.opt.phase == 'test':
            self.real = self.real_A
            tau = self.opt.tau
            T = self.opt.num_timesteps
            incs = np.array([0] + [1/(i+1) for i in range(T-1)])
            times = np.cumsum(incs)
            times = times / times[-1]
            times = 0.5 * times[-1] + 0.5 * times
            times = np.concatenate([np.zeros(1),times])
            times = torch.tensor(times).float().to(self.device)
            self.times = times
            bs =  self.real.size(0)
            time_idx = (torch.randint(T, size=[1]).to(self.device) * torch.ones(size=[1]).to(self.device)).long()
            self.time_idx = time_idx
            self.timestep     = times[time_idx]
            visuals = []
            with torch.no_grad():
                self.netG.eval()
                for t in range(self.opt.num_timesteps):

                    if t > 0:
                        delta = times[t] - times[t-1]
                        denom = times[-1] - times[t-1]
                        inter = (delta / denom).reshape(-1,1,1,1)
                        scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
                    Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    Xt_1     = self.netG(Xt, time_idx, z, direction=0)

                    setattr(self, "fake_"+str(t+1), Xt_1)

    @staticmethod
    def _seeded_randn(shape, sample_seeds, stream_id, device, dtype):
        chunks = []
        modulus = (1 << 63) - 1
        stream_stride = 0x1E3779B97F4A7C15
        for seed in sample_seeds:
            mixed_seed = (
                int(seed)
                + (int(stream_id) + 1) * stream_stride
            ) % modulus
            generator = torch.Generator(device=device)
            generator.manual_seed(mixed_seed)
            chunks.append(
                torch.randn(
                    (1, *tuple(shape)[1:]),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            )
        return torch.cat(chunks, dim=0)

    def inference(self, sample_seeds=None):
        tau = self.opt.tau
        T = self.opt.num_timesteps
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        times = torch.tensor(times).float().to(self.real_A.device)
        bs = self.real_A.size(0)
        if sample_seeds is not None:
            sample_seeds = [int(seed) for seed in sample_seeds]
            if len(sample_seeds) != bs:
                raise ValueError(
                    f"sample_seeds length ({len(sample_seeds)}) must match batch size ({bs})"
                )
        random_stream = 0
        Xt_1 = None
        with torch.no_grad():
            self.netG.eval()
            for t in range(T):
                if t > 0:
                    delta = times[t] - times[t-1]
                    denom = times[-1] - times[t-1]
                    inter = (delta / denom).reshape(-1, 1, 1, 1)
                    scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                    if sample_seeds is None:
                        bridge_noise = torch.randn_like(Xt).to(self.real_A.device)
                    else:
                        bridge_noise = self._seeded_randn(
                            Xt.shape,
                            sample_seeds,
                            random_stream,
                            self.real_A.device,
                            Xt.dtype,
                        )
                        random_stream += 1
                Xt = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * bridge_noise
                time_idx = (t * torch.ones(size=[bs]).to(self.real_A.device)).long()
                if sample_seeds is None:
                    z = torch.randn(size=[bs, 4*self.opt.ngf]).to(self.real_A.device)
                else:
                    z = self._seeded_randn(
                        (bs, 4 * self.opt.ngf),
                        sample_seeds,
                        random_stream,
                        self.real_A.device,
                        self.real_A.dtype,
                    )
                    random_stream += 1
                Xt_1 = self.netG(Xt, time_idx, z, direction=0)
            self.time_idx = ((T - 1) * torch.ones(size=[1]).to(self.real_A.device)).long()
            self.real_A_noisy = Xt.detach()
            self.fake_B = Xt_1.detach()

    def compute_D_loss(self):
        bs =  self.real_A.size(0)

        fake = self.fake_B.detach()
        std = torch.rand(size=[1]).item() * self.opt.std

        pred_fake = self.netD(fake,self.time_idx)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.pred_real = self.netD(self.real_B,self.time_idx)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_E_loss(self):
        XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B.detach()], dim=1)
        XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2.detach()], dim=1)
        self_energy = self.netE(XtXt_1, self.time_idx, XtXt_1).mean()
        cross_energy = torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0).mean()
        self.loss_E = -self_energy + cross_energy + cross_energy**2
        return self.loss_E

    @torch.no_grad()
    def update_ema(self):
        if not getattr(self, 'use_ema', False) or not hasattr(self, 'netG_ema'):
            return
        src = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
        for pe, p in zip(self.netG_ema.parameters(), src.parameters()):
            pe.mul_(self.ema_decay).add_(p.detach(), alpha=1.0 - self.ema_decay)
        for be, b in zip(self.netG_ema.buffers(), src.buffers()):
            be.copy_(b)

    def ema_swap_in(self):
        if getattr(self, 'use_ema', False) and hasattr(self, 'netG_ema'):
            self._netG_backup = self.netG
            self.netG = self.netG_ema

    def ema_swap_out(self):
        if hasattr(self, '_netG_backup'):
            self.netG = self._netG_backup
            del self._netG_backup

    def compute_D_aux_loss(self):
        loss_fake = self.netD_aux(self.fake_B.detach(), for_real=False).mean()
        loss_real = self.netD_aux(self.real_B, for_real=True).mean()
        self.loss_D_aux_fake = loss_fake
        self.loss_D_aux_real = loss_real
        self.loss_D_aux = 0.5 * (loss_fake + loss_real)
        return self.loss_D_aux

    def compute_D_A_loss(self):
        fake = self.fake_A.detach()
        pred_fake = self.netD_A(fake, self.time_idx)
        self.loss_D_A_fake = self.criterionGAN(pred_fake, False).mean()
        pred_real = self.netD_A(self.real_A, self.time_idx)
        self.loss_D_A_real = self.criterionGAN(pred_real, True).mean()
        self.loss_D_A = (self.loss_D_A_fake + self.loss_D_A_real) * 0.5
        return self.loss_D_A

    def compute_GAN_NCE_SB_loss(self):
        fake = self.fake_B

        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake, self.time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        self.loss_SB = 0
        if self.opt.lambda_SB > 0.0:
            XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
            XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)
            e_self = self.netE(XtXt_1, self.time_idx, XtXt_1).mean()
            e_cross = torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0)
            ET_XY = e_self - e_cross
            self.loss_E_self = e_self.item()
            self.loss_E_cross = e_cross.item()
            self.loss_ET_XY = ET_XY.item()
            self.loss_SB = -(self.opt.num_timesteps - self.time_idx[0]) / self.opt.num_timesteps * self.opt.tau * ET_XY
            self.loss_SB += self.opt.tau * torch.mean((self.real_A_noisy - self.fake_B) ** 2)
            self.loss_SB_raw = self.loss_SB.item()
            self.loss_SB = torch.clamp(self.loss_SB, min=self.opt.sb_clamp)

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
        else:
            self.loss_NCE = 0.0

        self.loss_G_GAN_aux = 0.0
        if self.opt.lambda_GAN_aux > 0.0:
            self.loss_G_GAN_aux = self.netD_aux(fake, for_G=True).mean() * self.opt.lambda_GAN_aux

        self.loss_SR = 0.0
        if self.opt.self_regularization > 0.0:
            self.loss_SR = self.opt.self_regularization * self.calculate_SR_loss(self.real_A_noisy, fake)

        self.loss_G = (self.loss_G_GAN + self.loss_G_GAN_aux +
                       self.opt.lambda_SB * self.loss_SB +
                       self.opt.lambda_NCE * self.loss_NCE +
                       self.loss_SR)
        return self.loss_G

    def compute_G_loss(self):
        fake = self.fake_B

        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake, self.time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        self.loss_SB = 0
        if self.opt.lambda_SB > 0.0:
            XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
            XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)
            e_self = self.netE(XtXt_1, self.time_idx, XtXt_1).mean()
            e_cross = torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0)
            ET_XY = e_self - e_cross
            self.loss_E_self = e_self.item()
            self.loss_E_cross = e_cross.item()
            self.loss_ET_XY = ET_XY.item()
            self.loss_SB = -(self.opt.num_timesteps - self.time_idx[0]) / self.opt.num_timesteps * self.opt.tau * ET_XY
            self.loss_SB += self.opt.tau * torch.mean((self.real_A_noisy - self.fake_B) ** 2)
            self.loss_SB_raw = self.loss_SB.item()
            self.loss_SB = torch.clamp(self.loss_SB, min=self.opt.sb_clamp)

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
        else:
            self.loss_NCE = 0.0

        self.loss_G_GAN_aux = 0.0
        if self.opt.lambda_GAN_aux > 0.0:
            self.loss_G_GAN_aux = self.netD_aux(fake, for_G=True).mean() * self.opt.lambda_GAN_aux

        self.loss_SR = 0.0
        if self.opt.self_regularization > 0.0:
            self.loss_SR = self.opt.self_regularization * self.calculate_SR_loss(self.real_A_noisy, fake)

        self.loss_idt = 0.0
        self.loss_idt_A = 0.0
        self.loss_NCE_Y = 0.0

        self.loss_G = (self.loss_G_GAN + self.loss_G_GAN_aux +
                       self.opt.lambda_SB * self.loss_SB +
                       self.opt.lambda_NCE * self.loss_NCE +
                       self.loss_SR)
        return self.loss_G

    def calculate_SR_loss(self, src, tgt):
        diff_chan = src - tgt
        batch_mean = torch.mean(diff_chan, dim=0)
        rgb_sum = torch.sum(batch_mean, 0)
        batch_mean2 = torch.mean(torch.mean(rgb_sum, 0), 0)
        return batch_mean2


    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        z    = torch.randn(size=[self.real_A.size(0),4*self.opt.ngf]).to(self.real_A.device)
        feat_q = self.netG(tgt, self.time_idx*0, z, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.time_idx*0,z,self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
