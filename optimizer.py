import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.nn_complex import CGRU, CFC, CRA, CRAN, MCRA

class CGRU_Optimizer(nn.Module):
    def __init__(self,
                input_dim = 10,
                output_dim = 1,
                rnn_units = 32,
                rnn_layers = 2,
                frame_size = 1,
                 ):

        super(CGRU_Optimizer, self).__init__()
        self.frame_size = frame_size
        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_layers = rnn_layers

        self.pre_cfc = CFC(input_dim=self.input_dim, output_dim=self.input_dim)

        self.cgru = CGRU(input_dim=self.input_dim, rnn_units=self.rnn_units, rnn_layers=self.rnn_layers)

        self.proj_cfc = nn.Sequential(
            CFC(input_dim=self.rnn_units, output_dim=self.output_dim),
            CFC(input_dim=self.output_dim, output_dim=self.output_dim)
        )

    def forward(self, in_source, h_state = None):

        mask_mags = torch.abs(in_source)
        mask_mags = torch.log(mask_mags + 1)
        mask_phase = torch.angle(in_source)
        source_real = mask_mags * torch.cos(mask_phase)
        source_imag = mask_mags * torch.sin(mask_phase)

        in_r = torch.reshape(source_real, [-1, self.input_dim])
        in_i= torch.reshape(source_imag, [-1, self.input_dim])
        in_r, in_i = self.pre_cfc([in_r, in_i])

        if h_state is None:
            real_h, imag_h = self.cgru([in_r, in_i])
        else:
            real_h, imag_h = self.cgru([in_r, in_i, torch.real(h_state), torch.imag(h_state)])

        out_r, out_i = self.proj_cfc([real_h, imag_h])

        out_real = torch.reshape(out_r, [-1, self.frame_size, self.output_dim // self.frame_size])
        out_imag = torch.reshape(out_i, [-1, self.frame_size, self.output_dim // self.frame_size])

        out_c = torch.complex(out_real, out_imag)
        out_abs = torch.abs(out_c)
        out_abs = torch.log(torch.clip(out_abs, np.exp(-10), np.exp(10))) / 10 + 1
        out_angle = torch.angle(out_c)

        out_complex = torch.multiply(out_abs, torch.exp(1j * out_angle))

        hidden_state_complex = torch.complex(real_h, imag_h)

        return out_complex, hidden_state_complex.detach()


class TCARN_Optimizer(nn.Module):
    def __init__(self,
                input_dim = 10,
                output_dim = 1,
                rnn_units = 32,
                rnn_layers = 2,
                frame_size = 1,
                use_comp = False,
                 ):

        super(TCARN_Optimizer, self).__init__()
        self.frame_size = frame_size
        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_layers = rnn_layers
        self.cp = 2

        self.pre_cfc = CFC(input_dim=self.input_dim, output_dim=self.rnn_units * self.cp)# // self.frame_size

        self.cgru = CGRU(input_dim=self.rnn_units * self.cp, rnn_units=self.rnn_units, rnn_layers=self.rnn_layers) #// * frame_size // self.cp
        self.cra = CRA(input_dim=self.rnn_units, output_dim=self.rnn_units, use_comp=use_comp)

        self.proj_cfc = nn.Sequential(
            CFC(input_dim=self.rnn_units, output_dim=self.output_dim),
            CFC(input_dim=self.output_dim, output_dim=self.output_dim, ac=False)
        )

    def forward(self, in_source, h_state = None, mode="train"):
        #mask_mags = in_source[:, 0]
        #mask_phase = in_source[:, 1]

        mask_mags = torch.abs(in_source)
        mask_mags = torch.log(mask_mags + 1)
        mask_phase = torch.angle(in_source)
        source_real = mask_mags * torch.cos(mask_phase)
        source_imag = mask_mags * torch.sin(mask_phase)

        in_r = torch.reshape(source_real, [-1, self.input_dim])
        in_i = torch.reshape(source_imag, [-1, self.input_dim])
        in_r, in_i = self.pre_cfc([in_r, in_i])

        #in_r, in_i = self.pre_cfc([source_real, source_imag])

        #in_r = torch.reshape(in_r, [-1, self.rnn_units * self.frame_size // self.cp])
        #in_i = torch.reshape(in_i, [-1, self.rnn_units * self.frame_size // self.cp])

        if h_state is None:
            real_h, imag_h = self.cgru([in_r, in_i])
        else:
            real_h, imag_h = self.cgru([in_r, in_i, torch.real(h_state), torch.imag(h_state)])

        real_h, imag_h = self.cra([real_h, imag_h], mode)
        out_r, out_i = self.proj_cfc([real_h, imag_h])

        out_real = torch.reshape(out_r, [-1, self.frame_size, self.output_dim // self.frame_size])
        out_imag = torch.reshape(out_i, [-1, self.frame_size, self.output_dim // self.frame_size])

        #out_abs = torch.sqrt(out_real ** 2 + out_imag ** 2)
        out_c = torch.complex(out_real, out_imag)
        out_abs = torch.abs(out_c)
        out_abs = torch.log(torch.clip(out_abs, np.exp(-10), np.exp(10))) / 10 + 1
        out_angle = torch.angle(out_c)

        out_complex = torch.multiply(out_abs, torch.exp(1j * out_angle))

        hidden_state_complex = torch.complex(real_h, imag_h)

        return out_complex, hidden_state_complex.detach()


'''class TCARN_Optimizer(nn.Module):
    def __init__(self,
                input_dim = 10,
                output_dim = 1,
                rnn_units = 32,
                rnn_layers = 2,
                frame_size = 1,
                use_comp = False,
                 ):

        super(TCARN_Optimizer, self).__init__()
        self.frame_size = frame_size
        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_layers = rnn_layers
        self.cp = 4

        self.pre_cfc = CFC(input_dim=self.input_dim, output_dim=self.rnn_units // self.cp * 3)

        self.cgru = CGRU(input_dim=self.rnn_units * self.frame_size // self.cp * 3, rnn_units=self.rnn_units * self.cp // 2, rnn_layers=self.rnn_layers)
        self.cra = MCRA(input_dim=self.rnn_units // self.cp // 2, head_dim = self.frame_size, output_dim=self.rnn_units // self.cp // 2, use_comp=use_comp)
        #self.cra_2 = CRA(input_dim=self.rnn_units * self.cp // 2, output_dim=self.rnn_units * self.cp // 2, use_comp=use_comp)

        self.proj_cfc = nn.Sequential(
            CFC(input_dim=self.rnn_units // self.cp // 2, output_dim=self.output_dim),
            CFC(input_dim=self.output_dim, output_dim=self.output_dim)
        )

    def forward(self, in_source, h_state = None, mode="train"):

        mask_mags = torch.abs(in_source)
        mask_mags = torch.log(mask_mags + 1)
        mask_phase = torch.angle(in_source)
        source_real = mask_mags * torch.cos(mask_phase)
        source_imag = mask_mags * torch.sin(mask_phase)

        #in_r = torch.reshape(source_real, [-1, self.input_dim])
        #in_i = torch.reshape(source_imag, [-1, self.input_dim])
        in_r, in_i = self.pre_cfc([source_real, source_imag])

        in_r = torch.reshape(in_r, [-1, self.frame_size * self.rnn_units // self.cp * 3])
        in_i = torch.reshape(in_i, [-1, self.frame_size * self.rnn_units // self.cp * 3])

        if h_state is None:
            real_h, imag_h = self.cgru([in_r, in_i])
        else:
            real_h, imag_h = self.cgru([in_r, in_i, torch.real(h_state), torch.imag(h_state)])

        hidden_state_complex = torch.complex(real_h, imag_h)

        #real_h, imag_h = self.cra_2([real_h, imag_h], mode)

        real_h = torch.reshape(real_h, [-1, self.frame_size, self.rnn_units * self.cp // self.frame_size // 2])
        imag_h = torch.reshape(imag_h, [-1, self.frame_size, self.rnn_units * self.cp // self.frame_size // 2])

        real_h, imag_h = self.cra([real_h, imag_h], mode)

        out_r, out_i = self.proj_cfc([real_h, imag_h])

        out_c = torch.complex(out_r, out_i)
        out_abs = torch.abs(out_c)
        out_abs = torch.log(torch.clip(out_abs, np.exp(-10), np.exp(10))) / 10 + 1
        out_angle = torch.angle(out_c)

        out_complex = torch.multiply(out_abs, torch.exp(1j * out_angle))

        return out_complex, hidden_state_complex.detach()'''

class TCARN_Optimizer_test(nn.Module):
    def __init__(self,
                input_dim = 10,
                output_dim = 1,
                rnn_units = 32,
                rnn_layers = 2,
                frame_size = 1,
                use_comp = False,
                 ):

        super(TCARN_Optimizer_test, self).__init__()
        self.frame_size = frame_size
        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_layers = rnn_layers
        self.cp = 1

        self.pre_cfc = CFC(input_dim=self.input_dim, output_dim=self.rnn_units * 2)

        self.cgru = CGRU(input_dim=self.rnn_units * 2, rnn_units=self.rnn_units, rnn_layers=self.rnn_layers)
        self.cra = CRA(input_dim=self.rnn_units, output_dim=self.rnn_units, use_comp=use_comp)

        self.proj_cfc = nn.Sequential(
            CFC(input_dim=self.rnn_units, output_dim=self.output_dim // 2),
            CFC(input_dim=self.output_dim // 2, output_dim=self.output_dim)
        )

    def forward(self, in_source, h_state = None, mode="train"):
        mask_mags = in_source[:, 0]
        mask_phase = in_source[:, 1]
        #mask_mags = torch.abs(in_source)
        mask_mags = torch.log(mask_mags + 1)
        #mask_phase = torch.angle(in_source)
        source_real = mask_mags * torch.cos(mask_phase)
        source_imag = mask_mags * torch.sin(mask_phase)

        in_r = torch.reshape(source_real, [-1, self.input_dim])
        in_i = torch.reshape(source_imag, [-1, self.input_dim])
        in_r, in_i = self.pre_cfc([in_r, in_i])

        #in_r, in_i = self.pre_cfc([source_real, source_imag])
        #in_r = torch.reshape(in_r, [-1, self.rnn_units * self.frame_size // self.cp])
        #in_i = torch.reshape(in_i, [-1, self.rnn_units * self.frame_size // self.cp])

        if h_state is None:
            real_h, imag_h = self.cgru([in_r, in_i])
        else:
            real_h, imag_h = self.cgru([in_r, in_i, torch.real(h_state), torch.imag(h_state)])

        real_h, imag_h = self.cra([real_h, imag_h], mode)
        out_r, out_i = self.proj_cfc([real_h, imag_h])

        out_real = torch.reshape(out_r, [-1, self.frame_size, self.output_dim // self.frame_size])
        out_imag = torch.reshape(out_i, [-1, self.frame_size, self.output_dim // self.frame_size])

        out_abs = torch.sqrt(out_real ** 2 + out_imag ** 2)
        #out_c = torch.complex(out_real, out_imag)
        #out_abs = torch.abs(out_c)
        out_abs = torch.log(torch.clip(out_abs, np.exp(-10), np.exp(10))) / 10 + 1
        #out_angle = torch.angle(out_c)

        #out_complex = torch.multiply(out_abs, torch.exp(1j * out_angle))

        #hidden_state_complex = torch.complex(real_h, imag_h)

        return out_abs, real_h.detach(), imag_h.detach()
class CARN_Optimizer(nn.Module):
    def __init__(self,
                input_dim = 10,
                output_dim = 1,
                rnn_units = 32,
                rnn_layers = 2,
                frame_size = 1,
                use_comp = False,
                 ):

        super(CARN_Optimizer, self).__init__()
        self.frame_size = frame_size
        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_layers = rnn_layers

        self.pre_cfc = CFC(input_dim=self.input_dim, output_dim=self.input_dim)

        self.cgru = CGRU(input_dim=self.input_dim, rnn_units=self.rnn_units, rnn_layers=self.rnn_layers)
        self.cra = CRA(input_dim=self.rnn_units, output_dim=self.rnn_units, use_comp=use_comp)

        self.proj_cfc = nn.Sequential(
            CFC(input_dim=self.rnn_units, output_dim=self.output_dim),
            CFC(input_dim=self.output_dim, output_dim=self.output_dim)
        )

    def forward(self, in_source, h_state = None, mode="train"):

        mask_mags = torch.abs(in_source)
        mask_mags = torch.log(mask_mags + 1)
        mask_phase = torch.angle(in_source)
        source_real = mask_mags * torch.cos(mask_phase)
        source_imag = mask_mags * torch.sin(mask_phase)

        in_r = torch.reshape(source_real, [-1, self.input_dim])
        in_i= torch.reshape(source_imag, [-1, self.input_dim])
        in_r, in_i = self.pre_cfc([in_r, in_i])

        if h_state is None:
            real_h, imag_h = self.cgru([in_r, in_i])
        else:
            real_h, imag_h = self.cgru([in_r, in_i, torch.real(h_state), torch.imag(h_state)])

        real_h, imag_h = self.cra([real_h, imag_h], mode)
        out_r, out_i = self.proj_cfc([real_h, imag_h])

        out_real = torch.reshape(out_r, [-1, self.frame_size, self.output_dim // self.frame_size])
        out_imag = torch.reshape(out_i, [-1, self.frame_size, self.output_dim // self.frame_size])

        out_c = torch.complex(out_real, out_imag)
        out_abs = torch.abs(out_c)
        out_abs = torch.log(torch.clip(out_abs, np.exp(-10), np.exp(10))) / 10 + 1
        out_angle = torch.angle(out_c)

        out_complex = torch.multiply(out_abs, torch.exp(1j * out_angle))

        hidden_state_complex = torch.complex(real_h, imag_h)

        return out_complex, hidden_state_complex.detach()


'''class Optimizer(nn.Module):
    def __init__(self,
                input_dim = 10,
                output_dim = 1,
                rnn_units = 32,
                frame_size = 1
                 ):

        super(Optimizer, self).__init__()
        self.frame_size = frame_size
        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        #print(self.output_dim)

        self.in_linear_r = nn.Sequential(
            nn.Linear(in_features = self.input_dim, out_features = self.input_dim, dtype = torch.float32),
            nn.PReLU(dtype = torch.float32),
        )

        self.in_linear_i = nn.Sequential(
            nn.Linear(in_features = self.input_dim, out_features = self.input_dim, dtype = torch.float32),
            nn.PReLU(dtype = torch.float32),
        )

        self.rnn_r = nn.GRUCell(
            input_size=self.input_dim,
            hidden_size=self.rnn_units,
            dtype=torch.float32)

        self.rnn_i = nn.GRUCell(
            input_size=self.input_dim,
            hidden_size=self.rnn_units,
            dtype=torch.float32)

        self.rnn_r_1 = nn.GRUCell(
            input_size=self.rnn_units,
            hidden_size=self.rnn_units,
            dtype=torch.float32)

        self.rnn_i_1 = nn.GRUCell(
            input_size=self.rnn_units,
            hidden_size=self.rnn_units,
            dtype=torch.float32)

        self.ln_r = nn.LayerNorm(normalized_shape=self.rnn_units, dtype=torch.float32)
        self.rnn_lr = nn.Linear(self.rnn_units, self.output_dim, dtype=torch.float32)
        self.ln_i = nn.LayerNorm(normalized_shape=self.rnn_units, dtype=torch.float32)
        self.rnn_li = nn.Linear(self.rnn_units, self.output_dim, dtype=torch.float32)

        self.out_linear_r = nn.Sequential(
            nn.PReLU(dtype = torch.float32),
            nn.Linear(in_features = self.output_dim, out_features = self.output_dim, dtype = torch.float32),
            nn.PReLU(dtype = torch.float32),
        )
        self.out_linear_i = nn.Sequential(
            nn.PReLU(dtype = torch.float32),
            nn.Linear(in_features = self.output_dim, out_features = self.output_dim, dtype = torch.float32),
            nn.PReLU(dtype = torch.float32),
        )

        self.real_conv_1 = nn.Conv2d(3, 4, 3, 3,padding=1,dilation=1, groups=1, dtype = torch.float32)
        self.imag_conv_1 = nn.Conv2d(3, 4, 3, 3,padding=1,dilation=1, groups=1, dtype = torch.float32)
        self.bn_r_1 = nn.BatchNorm2d(4,dtype=torch.float32)
        self.bn_i_1 = nn.BatchNorm2d(4, dtype=torch.float32)
        self.conv_relu_r1 = nn.ReLU()
        self.conv_relu_i1 = nn.ReLU()

        self.real_conv_2 = nn.Conv2d(4, 4, 1, 3, padding=1,dilation=2, groups=1, dtype = torch.float32)
        self.imag_conv_2 = nn.Conv2d(4, 4, 1, 3, padding=1,dilation=2, groups=1, dtype = torch.float32)
        self.bn_r_2 = nn.BatchNorm2d(4,dtype=torch.float32)
        self.bn_i_2 = nn.BatchNorm2d(4, dtype=torch.float32)
        self.conv_relu_r2 = nn.ReLU()
        self.conv_relu_i2 = nn.ReLU()

        self.weight_init()

    def flatten_parameters(self):
        if isinstance(self.enhance, nn.GRU):
            self.enhance.flatten_parameters()

    def weight_init(self):
        if isinstance(self.in_linear_r, nn.Linear):
            nn.init.xavier_normal_(self.in_linear_r.weight)
            nn.init.constant_(self.in_linear_r.bias, 0)

        if isinstance(self.in_linear_i, nn.Linear):
            nn.init.xavier_normal_(self.in_linear_i.weight)
            nn.init.constant_(self.in_linear_i.bias, 0)

        if isinstance(self.in_linear_r, nn.PReLU):
            nn.init.xavier_normal_(self.in_linear_r.weight)

        if isinstance(self.in_linear_i, nn.PReLU):
            nn.init.xavier_normal_(self.in_linear_i.weight)

        if isinstance(self.out_linear_r, nn.Linear):
            nn.init.xavier_normal_(self.out_linear_r.weight)
            nn.init.constant_(self.out_linear_r.bias, 0)

        if isinstance(self.out_linear_i, nn.Linear):
            nn.init.xavier_normal_(self.out_linear_i.weight)
            nn.init.constant_(self.out_linear_i.bias, 0)

        if isinstance(self.out_linear_r, nn.PReLU):
            nn.init.xavier_normal_(self.out_linear_r.weight)

        if isinstance(self.out_linear_i, nn.PReLU):
            nn.init.xavier_normal_(self.out_linear_i.weight)

        if isinstance(self.rnn_r, nn.GRUCell):
            nn.init.orthogonal_(self.rnn_r.weight_ih)
            nn.init.orthogonal_(self.rnn_r.weight_hh)
            nn.init.constant_(self.rnn_r.bias_ih, 0)
            nn.init.constant_(self.rnn_r.bias_hh, 0)

        if isinstance(self.rnn_i, nn.GRUCell):
            nn.init.orthogonal_(self.rnn_i.weight_ih)
            nn.init.orthogonal_(self.rnn_i.weight_hh)
            nn.init.constant_(self.rnn_i.bias_ih, 0)
            nn.init.constant_(self.rnn_i.bias_hh, 0)

        if isinstance(self.rnn_r_1, nn.GRUCell):
            nn.init.orthogonal_(self.rnn_r_1.weight_ih)
            nn.init.orthogonal_(self.rnn_r_1.weight_hh)
            nn.init.constant_(self.rnn_r_1.bias_ih, 0)
            nn.init.constant_(self.rnn_r_1.bias_hh, 0)

        if isinstance(self.rnn_i_1, nn.GRUCell):
            nn.init.orthogonal_(self.rnn_i_1.weight_ih)
            nn.init.orthogonal_(self.rnn_i_1.weight_hh)
            nn.init.constant_(self.rnn_i_1.bias_ih, 0)
            nn.init.constant_(self.rnn_i_1.bias_hh, 0)

    def forward(self, in_source, h_state = None):

        max_error, _ = torch.max(torch.abs(torch.pow(in_source[:, :, -1:], 2)),dim=1, keepdim=True)
        mask_mags = torch.abs(in_source)
        mask_mags = torch.log(mask_mags + 1)
        mask_phase = torch.angle(in_source)
        source_real = mask_mags * torch.cos(mask_phase)
        source_imag = mask_mags * torch.sin(mask_phase)

        in_r = torch.reshape(source_real, [-1, self.input_dim])
        in_i= torch.reshape(source_imag, [-1, self.input_dim])
        in_r = self.in_linear_r(in_r)
        in_i = self.in_linear_i(in_i)

        if h_state is None:
            h_real_1 = self.rnn_r(in_r)
            h_imag_1 = self.rnn_i(in_i)
            h_real = self.rnn_r_1(h_real_1)
            h_imag = self.rnn_i_1(h_imag_1)
        else:
            h_real_1 = self.rnn_r(in_r, torch.real(h_state[0]))
            h_imag_1 = self.rnn_i(in_i, torch.imag(h_state[0]))
            h_real = self.rnn_r_1(h_real_1, torch.real(h_state[1]))
            h_imag = self.rnn_i_1(h_imag_1, torch.imag(h_state[1]))

        out_real = self.rnn_lr(h_real)
        out_imag = self.rnn_li(h_imag)

        out_r = self.out_linear_r(out_real)
        out_i = self.out_linear_i(out_imag)

        out_real = torch.reshape(out_r, [-1, self.frame_size, self.output_dim // self.frame_size])
        out_imag = torch.reshape(out_i, [-1, self.frame_size, self.output_dim // self.frame_size])

        out_c = torch.complex(out_real, out_imag)
        out_abs = torch.abs(out_c)
        out_abs = torch.log(torch.clip(out_abs, np.exp(-10), np.exp(10))) / 10 + 1
        out_angle = torch.angle(out_c)
        out_complex = torch.multiply(out_abs, torch.exp(1j * out_angle))

        h_state_complex_1 = torch.complex(h_real_1, h_imag_1)
        h_state_complex_2 = torch.complex(h_real, h_imag)
        hidden_state_complex = torch.stack((h_state_complex_1, h_state_complex_2),dim=0)

        return out_complex, hidden_state_complex.detach()'''