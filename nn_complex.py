import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class CFC(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 ac=True
                 ):
        super(CFC, self).__init__()

        self.in_dim = input_dim
        self.out_dim = output_dim

        self.real_ln = nn.Linear(self.in_dim, self.out_dim)
        self.imag_ln = nn.Linear(self.in_dim, self.out_dim)

        self.real_ac = nn.PReLU()
        self.imag_ac = nn.PReLU()
        self.ac = ac

        # self.init_params()

    def init_params(self):
        if isinstance(self.real_ln, nn.Linear):
            nn.init.normal_(self.real_ln.weight)
            nn.init.constant_(self.real_ln.bias, 0)

        if isinstance(self.imag_ln, nn.Linear):
            nn.init.normal_(self.imag_ln.weight)
            nn.init.constant_(self.imag_ln.bias, 0)

        if isinstance(self.real_ac, nn.PReLU):
            nn.init.xavier_normal_(self.real_ac.weight)

        if isinstance(self.imag_ac, nn.PReLU):
            nn.init.xavier_normal_(self.imag_ac.weight)

    def forward(self, input):
        real_in, imag_in = input[0], input[1]

        if self.ac == False:
            real_out = self.real_ln(real_in)
            imag_out = self.imag_ln(imag_in)
        else:
            real_out = self.real_ac(self.real_ln(real_in))
            imag_out = self.imag_ac(self.imag_ln(imag_in))

        return real_out, imag_out


class CGRUcell(nn.Module):
    def __init__(self,
                 input_dim,
                 rnn_units,
                 ):
        super(CGRUcell, self).__init__()

        self.in_dim = input_dim
        self.rnn_units = rnn_units

        self.real_gru = nn.GRUCell(self.in_dim, self.rnn_units)
        self.imag_gru = nn.GRUCell(self.in_dim, self.rnn_units)

    def forward(self, input):
        real_in, imag_in = input[0], input[1]
        real_h, imag_h = None, None
        if len(input) > 2:
            real_h, imag_h = input[2], input[3]

        if real_h is None and imag_h is None:
            real_out = self.real_gru(real_in)
            imag_out = self.real_gru(imag_in)
        else:
            real_out = self.real_gru(real_in, real_h)
            imag_out = self.real_gru(imag_in, imag_h)
        return real_out, imag_out


class CGRU(nn.Module):
    def __init__(self,
                 input_dim,
                 rnn_units,
                 rnn_layers
                 ):
        super(CGRU, self).__init__()

        self.in_dim = input_dim
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers

        rnns = []
        for idx in range(self.rnn_layers):
            rnns.append(
                CGRUcell(
                    input_dim=self.in_dim if idx == 0 else self.rnn_units,
                    rnn_units=self.rnn_units,
                )
            )

        self.gru = nn.Sequential(*rnns)

    def forward(self, input):
        real_in, imag_in = input[0], input[1]
        real_h, imag_h = None, None
        if len(input) > 2:
            real_h, imag_h = input[2], input[3]

        if real_h is None and imag_h is None:
            real_h, imag_h = self.gru([real_in, imag_in])
        else:
            real_h, imag_h = self.gru([real_in, imag_in, real_h, imag_h])

        return real_h, imag_h


class CRA(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 use_comp = False
                 ):
        super(CRA, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.q_linear_r = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.k_linear_r = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.v_linear_r = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.q_r = nn.Parameter(torch.zeros(1, self.input_dim))
        self.k_r = nn.Parameter(torch.zeros(1, self.input_dim))
        self.v_r = nn.Parameter(torch.zeros(1, self.input_dim))
        self.v_linear_r_sigm = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.Sigmoid()
        )
        self.v_linear_r_tanh = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.Tanh()
        )
        self.ac_r = nn.Softmax(dim=2)
        self.out_linear_r = nn.Sequential(
            nn.Linear(self.input_dim, 2 * self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
            nn.Linear(in_features=2 * self.output_dim, out_features=self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
        )
        self.out_l_r = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
        )
        self.ln_q_r = nn.LayerNorm(self.input_dim)
        self.ln_k_r = nn.LayerNorm(self.input_dim)
        self.ln_v_r = nn.LayerNorm(self.input_dim)
        self.ln_attn_r = nn.LayerNorm(self.input_dim)

        self.q_linear_i = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.k_linear_i = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.v_linear_i = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.q_i = nn.Parameter(torch.zeros(1, self.input_dim))
        self.k_i = nn.Parameter(torch.zeros(1, self.input_dim))
        self.v_i = nn.Parameter(torch.zeros(1, self.input_dim))
        self.v_linear_i_sigm = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.Sigmoid()
        )
        self.v_linear_i_tanh = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.Tanh()
        )
        self.ac_i = nn.Softmax(dim=2)
        self.out_linear_i = nn.Sequential(
            nn.Linear(self.input_dim, 2 * self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
            nn.Linear(in_features=2 * self.output_dim, out_features=self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
        )
        self.out_l_i = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
        )
        self.ln_q_i = nn.LayerNorm(self.input_dim)
        self.ln_k_i = nn.LayerNorm(self.input_dim)
        self.ln_v_i = nn.LayerNorm(self.input_dim)
        self.ln_attn_i = nn.LayerNorm(self.input_dim)
        self.use_comp = use_comp

    def forward(self, x, mode = "train"):
        h_real, h_imag = x[0], x[1]

        Q_r = self.q_linear_r(self.ln_q_r(h_real)) * F.sigmoid(self.q_r)
        K_r = self.k_linear_r(self.ln_k_r(h_real)) * F.sigmoid(self.k_r)
        if mode == "train":
            V_r = self.ln_v_r(h_real) * self.v_linear_r_sigm(F.sigmoid(self.v_r)) * self.v_linear_r_tanh(
                F.sigmoid(self.v_r))
        else:
            V_r = self.ln_v_r(h_real)
        attn_r = self.ac_r(torch.matmul(torch.transpose(torch.unsqueeze(Q_r, dim=1), dim1=-1, dim0=-2),
                                        torch.unsqueeze(K_r, dim=1), ) / np.sqrt(self.output_dim))

        Q_i = self.q_linear_i(self.ln_q_i(h_imag)) * F.sigmoid(self.q_i)
        K_i = self.k_linear_i(self.ln_k_i(h_imag)) * F.sigmoid(self.k_i)
        if mode == "train":
            V_i = self.v_linear_i(self.ln_v_i(h_imag)) * self.v_linear_i_sigm(F.sigmoid(self.v_i)) * self.v_linear_i_tanh(
                F.sigmoid(self.v_i))
        else:
            V_i = self.ln_v_i(h_imag)
        attn_i = self.ac_i(torch.matmul(torch.transpose(torch.unsqueeze(Q_i, dim=1), dim1=-1, dim0=-2),
                                        torch.unsqueeze(K_i, dim=1), ) / np.sqrt(self.output_dim))

        if self.use_comp:
            attn_r = attn_r - attn_i
            attn_i = attn_r + attn_i

        attn_r = torch.squeeze(torch.matmul(attn_r, torch.unsqueeze(V_r, dim=2)))
        attn_ln_r = self.ln_attn_r(attn_r + Q_r)
        out_r = self.out_linear_r(attn_ln_r) + self.out_l_r(attn_ln_r)

        attn_i = torch.squeeze(torch.matmul(attn_i, torch.unsqueeze(V_i, dim=2)))
        attn_ln_i = self.ln_attn_i(attn_i + Q_i)
        out_i = self.out_linear_i(attn_ln_i) + self.out_l_i(attn_ln_i)


        return out_r, out_i

'''class CRA(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 use_comp = False
                 ):
        super(CRA, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.q_linear_r = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.k_linear_r = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.v_linear_r = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.q_r = nn.Parameter(torch.zeros(1, self.input_dim))
        self.k_r = nn.Parameter(torch.zeros(1, self.input_dim))
        self.v_r = nn.Parameter(torch.zeros(1, self.input_dim))
        self.v_linear_r_sigm = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.Sigmoid()
        )
        self.v_linear_r_tanh = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.Tanh()
        )
        self.ac_r = nn.Softmax(dim=2)
        self.out_linear_r = nn.Sequential(
            nn.Linear(self.input_dim, 2 * self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
            nn.Linear(in_features=2 * self.output_dim, out_features=self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
        )
        self.out_l_r = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
        )
        self.ln_q_r = nn.LayerNorm(self.input_dim)
        self.ln_k_r = nn.LayerNorm(self.input_dim)
        self.ln_v_r = nn.LayerNorm(self.input_dim)
        self.ln_attn_r = nn.LayerNorm(self.input_dim)

        self.q_linear_i = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.k_linear_i = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.v_linear_i = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.q_i = nn.Parameter(torch.zeros(1, self.input_dim))
        self.k_i = nn.Parameter(torch.zeros(1, self.input_dim))
        self.v_i = nn.Parameter(torch.zeros(1, self.input_dim))
        self.v_linear_i_sigm = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.Sigmoid()
        )
        self.v_linear_i_tanh = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.Tanh()
        )
        self.ac_i = nn.Softmax(dim=2)
        self.out_linear_i = nn.Sequential(
            nn.Linear(self.input_dim, 2 * self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
            nn.Linear(in_features=2 * self.output_dim, out_features=self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
        )
        self.out_l_i = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
        )
        self.ln_q_i = nn.LayerNorm(self.input_dim)
        self.ln_k_i = nn.LayerNorm(self.input_dim)
        self.ln_v_i = nn.LayerNorm(self.input_dim)
        self.ln_attn_i = nn.LayerNorm(self.input_dim)
        self.use_comp = use_comp

    def forward(self, x, mode = "train"):
        h_real, h_imag = x[0], x[1]

        Q_r = self.q_linear_r(self.ln_q_r(h_real) * F.sigmoid(self.q_r) )
        K_r = self.k_linear_r(self.ln_k_r(h_real)  * F.sigmoid(self.k_r))
        if mode == "train":
            V_r = self.v_linear_r(self.ln_v_r(h_real) * F.sigmoid(self.k_r))
        else:
            V_r = self.v_linear_r(self.ln_v_r(h_real) * F.sigmoid(self.k_r))
        attn_r = self.ac_r(torch.matmul(torch.transpose(torch.unsqueeze(Q_r, dim=1), dim1=-1, dim0=-2),
                                        torch.unsqueeze(K_r, dim=1), ) / np.sqrt(self.output_dim))

        Q_i = self.q_linear_i(self.ln_q_i(h_imag) * F.sigmoid(self.q_i))
        K_i = self.k_linear_i(self.ln_k_i(h_imag) * F.sigmoid(self.k_i))
        if mode == "train":
            V_i = self.v_linear_i(self.ln_v_i(h_imag) * F.sigmoid(self.v_i))
        else:
            V_i = self.v_linear_i(self.ln_v_i(h_imag) * F.sigmoid(self.v_i))
        attn_i = self.ac_i(torch.matmul(torch.transpose(torch.unsqueeze(Q_i, dim=1), dim1=-1, dim0=-2),
                                        torch.unsqueeze(K_i, dim=1), ) / np.sqrt(self.output_dim))

        if self.use_comp:
            attn_r = attn_r - attn_i
            attn_i = attn_r + attn_i

        attn_r = torch.squeeze(torch.matmul(attn_r, torch.unsqueeze(V_r, dim=2)))
        attn_ln_r = self.ln_attn_r(attn_r + Q_r)
        out_r = self.out_linear_r(attn_ln_r) + self.out_l_r(attn_ln_r)

        attn_i = torch.squeeze(torch.matmul(attn_i, torch.unsqueeze(V_i, dim=2)))
        attn_ln_i = self.ln_attn_i(attn_i + Q_i)
        out_i = self.out_linear_i(attn_ln_i) + self.out_l_i(attn_ln_i)


        return out_r, out_i'''

class CRA(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 use_comp=False
                 ):
        super(CRA, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.q_linear_r = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.k_linear_r = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.v_linear_r = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.q_r = nn.Parameter(torch.zeros(1, self.input_dim))
        self.k_r = nn.Parameter(torch.zeros(1, self.input_dim))
        self.v_r = nn.Parameter(torch.zeros(1, self.input_dim))
        self.ac_r = nn.Softmax(dim=2)
        self.out_linear_r = nn.Sequential(
            nn.Linear(self.input_dim, 2 * self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
            nn.Linear(in_features=2 * self.output_dim, out_features=self.output_dim, dtype=torch.float32),
        )

        self.ln_r = nn.LayerNorm(self.input_dim)
        self.ln_attn_r = nn.LayerNorm(self.input_dim)

        self.q_linear_i = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.k_linear_i = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.v_linear_i = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.q_i = nn.Parameter(torch.zeros(1, self.input_dim))
        self.k_i = nn.Parameter(torch.zeros(1, self.input_dim))
        self.v_i = nn.Parameter(torch.zeros(1, self.input_dim))
        self.ac_i = nn.Softmax(dim=2)
        self.out_linear_i = nn.Sequential(
            nn.Linear(self.input_dim, 2 * self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
            nn.Linear(in_features=2 * self.output_dim, out_features=self.output_dim, dtype=torch.float32),
        )
        self.ln_i = nn.LayerNorm(self.input_dim)
        self.ln_attn_i = nn.LayerNorm(self.input_dim)
        self.use_comp = use_comp

    def forward(self, x, mode="train"):
        h_real, h_imag = x[0], x[1]

        h_real = self.ln_r(h_real)
        h_imag = self.ln_i(h_imag)

        Q_r = self.q_linear_r(h_real * F.sigmoid(self.q_r))
        K_r = self.k_linear_r(h_real * F.sigmoid(self.k_r))
        if mode == "train":
            V_r = self.v_linear_r(h_real * F.sigmoid(self.v_r))
        else:
            V_r = self.v_linear_r(h_real * F.sigmoid(self.v_r))
        attn_r = self.ac_r(torch.matmul(torch.transpose(torch.unsqueeze(Q_r, dim=1), dim1=-1, dim0=-2),
                                        torch.unsqueeze(K_r, dim=1)) / np.sqrt(self.output_dim))

        Q_i = self.q_linear_i(h_imag * F.sigmoid(self.q_i))
        K_i = self.k_linear_i(h_imag * F.sigmoid(self.k_i))
        if mode == "train":
            V_i = self.v_linear_i(h_imag * F.sigmoid(self.v_i))
        else:
            V_i = self.v_linear_i(h_imag * F.sigmoid(self.v_i))
        attn_i = self.ac_i(torch.matmul(torch.transpose(torch.unsqueeze(Q_i, dim=1), dim1=-1, dim0=-2),
                                        torch.unsqueeze(K_i, dim=1)) / np.sqrt(self.output_dim))

        if self.use_comp:
            attn_r = attn_r - attn_i
            attn_i = attn_r + attn_i

        attn_r = torch.squeeze(torch.matmul(attn_r, torch.unsqueeze(V_r, dim=2)))
        attn_ln_r = self.ln_attn_r(attn_r + h_real)
        out_r = self.out_linear_r(attn_ln_r) + attn_ln_r

        attn_i = torch.squeeze(torch.matmul(attn_i, torch.unsqueeze(V_i, dim=2)))
        attn_ln_i = self.ln_attn_i(attn_i + h_imag)
        out_i = self.out_linear_i(attn_ln_i) + attn_ln_i

        return out_r, out_i


class MCRA(nn.Module):
    def __init__(self,
                 input_dim,
                 head_dim,
                 output_dim,
                 use_comp=False
                 ):
        super(MCRA, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim

        self.q_linear_r = nn.Linear(self.input_dim, self.output_dim, dtype=torch.float32)
        self.k_linear_r = nn.Linear(self.input_dim, self.output_dim, dtype=torch.float32)
        self.v_linear_r = nn.Linear(self.input_dim, self.output_dim, dtype=torch.float32)
        self.q_r = nn.Parameter(torch.zeros(self.head_dim, self.input_dim))
        self.k_r = nn.Parameter(torch.zeros(self.head_dim, self.input_dim))
        self.v_r = nn.Parameter(torch.zeros(self.head_dim, self.input_dim))
        self.ac_r = nn.Softmax(dim=2)
        self.out_linear_r = nn.Sequential(
            nn.Linear(self.output_dim, 2 * self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
            nn.Linear(in_features=2 * self.output_dim, out_features=self.output_dim, dtype=torch.float32),
        )
        '''self.out_l_r = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
        )'''
        self.ln_r = nn.LayerNorm(self.input_dim)
        self.ln_attn_r = nn.LayerNorm(self.output_dim)

        self.q_linear_i = nn.Linear(self.input_dim, self.output_dim, dtype=torch.float32)
        self.k_linear_i = nn.Linear(self.input_dim, self.output_dim, dtype=torch.float32)
        self.v_linear_i = nn.Linear(self.input_dim, self.output_dim, dtype=torch.float32)
        self.q_i = nn.Parameter(torch.zeros(self.head_dim, self.input_dim))
        self.k_i = nn.Parameter(torch.zeros(self.head_dim, self.input_dim))
        self.v_i = nn.Parameter(torch.zeros(self.head_dim, self.input_dim))
        self.ac_i = nn.Softmax(dim=2)
        self.out_linear_i = nn.Sequential(
            nn.Linear(self.output_dim, 2 * self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
            nn.Linear(in_features=2 * self.output_dim, out_features=self.output_dim, dtype=torch.float32),
        )
        '''self.out_l_i = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
        )'''
        self.ln_i = nn.LayerNorm(self.input_dim)
        self.ln_attn_i = nn.LayerNorm(self.output_dim)
        self.use_comp = use_comp

    def forward(self, x, mode="train"):
        h_real, h_imag = x[0], x[1]

        h_real = self.ln_r(h_real)
        h_imag = self.ln_r(h_imag)

        Q_r = self.q_linear_r(h_real * F.sigmoid(self.q_r))
        K_r = self.k_linear_r(h_real * F.sigmoid(self.k_r))
        if mode == "train":
            V_r = self.v_linear_r(h_real * F.sigmoid(self.v_r))
        else:
            V_r = self.v_linear_r(h_real * F.sigmoid(self.v_r))

        attn_r = self.ac_i(torch.matmul(torch.transpose(Q_r, dim1=-1, dim0=-2),
                                        K_r) / np.sqrt(self.output_dim))

        Q_i = self.q_linear_i(h_imag * F.sigmoid(self.q_i))
        K_i = self.k_linear_i(h_imag * F.sigmoid(self.k_i))
        if mode == "train":
            V_i = self.v_linear_i(h_imag * F.sigmoid(self.v_i))
        else:
            V_i = self.v_linear_i(h_imag * F.sigmoid(self.v_i))
        attn_i = self.ac_i(torch.matmul(torch.transpose(Q_i, dim1=-1, dim0=-2),
                                        K_i) / np.sqrt(self.output_dim))

        if self.use_comp:
            attn_r = attn_r - attn_i
            attn_i = attn_r + attn_i
        attn_r = torch.squeeze(torch.matmul(V_r, attn_r))
        attn_ln_r = self.ln_attn_r(attn_r + Q_r)
        out_r = self.out_linear_r(attn_ln_r) + attn_ln_r

        attn_i = torch.squeeze(torch.matmul(V_i, attn_i))
        attn_ln_i = self.ln_attn_i(attn_i + Q_i)
        out_i = self.out_linear_i(attn_ln_i) + attn_ln_i

        return out_r, out_i


class CRAN(nn.Module):
    def __init__(self,
                 input_dim,
                 frame_dim,
                 output_dim,
                 use_comp=False,
                 ):
        super(CRAN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.frame_dim = frame_dim

        self.q_linear_r = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.v_linear_r = nn.Linear(self.input_dim, self.output_dim, dtype=torch.float32)
        self.q_r = nn.Parameter(torch.zeros(self.input_dim, self.output_dim))
        self.k_r = nn.Parameter(torch.zeros(self.input_dim, self.output_dim))
        self.v_r = nn.Parameter(torch.zeros(self.frame_dim, self.output_dim))
        self.v_linear_r_sigm = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.Sigmoid()
        )
        self.v_linear_r_tanh = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.Tanh()
        )
        self.ac_r = nn.Softmax(dim=2)
        self.out_linear_r = nn.Sequential(
            nn.Linear(self.output_dim, 4 * self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
            nn.Linear(in_features=4 * self.output_dim, out_features=self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
        )
        self.out_l_r = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
        )
        '''self.ln_k_r = nn.LayerNorm(self.input_dim)
        self.ln_v_r = nn.LayerNorm(self.output_dim)'''
        self.ln_attn_r = nn.LayerNorm(self.output_dim)

        self.q_linear_i = nn.Linear(self.input_dim, self.input_dim, dtype=torch.float32)
        self.v_linear_i = nn.Linear(self.input_dim, self.output_dim, dtype=torch.float32)
        self.q_i = nn.Parameter(torch.zeros(self.input_dim, self.output_dim))
        self.k_i = nn.Parameter(torch.zeros(self.input_dim, self.output_dim))
        self.v_i = nn.Parameter(torch.zeros(self.frame_dim, self.output_dim))
        self.v_linear_i_sigm = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.Sigmoid()
        )
        self.v_linear_i_tanh = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.Tanh()
        )
        self.ac_i = nn.Softmax(dim=2)
        self.out_linear_i = nn.Sequential(
            nn.Linear(self.output_dim, 2 * self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
            nn.Linear(in_features=2 * self.output_dim, out_features=self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
        )
        self.out_l_i = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim, dtype=torch.float32),
            nn.PReLU(dtype=torch.float32),
        )
        '''self.ln_k_i = nn.LayerNorm(self.output_dim)
        self.ln_v_i = nn.LayerNorm(self.output_dim)'''
        self.ln_attn_i = nn.LayerNorm(self.output_dim)
        self.use_comp = use_comp

    def forward(self, x, mode="train"):
        h_real, h_imag = x[0], x[1]

        Q_r = torch.matmul(self.q_linear_r(h_real), F.sigmoid(self.q_r))
        K_r = torch.matmul(h_real, F.sigmoid(self.k_r))
        if mode == "train":
            V_r = self.v_linear_r(h_real) * (
                        self.v_linear_r_sigm(F.sigmoid(self.v_r)) * self.v_linear_r_tanh(F.sigmoid(self.v_r)))
        else:
            V_r = self.v_linear_r(h_real)
        attn_r = self.ac_r(torch.matmul(K_r, torch.transpose(Q_r, dim1=-1, dim0=-2)) / np.sqrt(self.output_dim))
        attn_r = torch.matmul(attn_r, V_r)

        Q_i = torch.matmul(self.q_linear_i(h_imag), F.sigmoid(self.q_i))
        K_i = torch.matmul(h_imag, F.sigmoid(self.k_i))
        if mode == "train":
            V_i = self.v_linear_i(h_imag) * (
                        self.v_linear_i_sigm(F.sigmoid(self.v_i)) * self.v_linear_i_tanh(F.sigmoid(self.v_i)))
        else:
            V_i = self.v_linear_i(h_imag)
        attn_i = self.ac_r(torch.matmul(K_i, torch.transpose(Q_r, dim1=-1, dim0=-2)) / np.sqrt(self.output_dim))
        attn_i = torch.matmul(attn_i, V_i)

        if self.use_comp:
            attn_r = attn_r - attn_i
            attn_i = attn_r + attn_i

        attn_ln_r = self.ln_attn_r(attn_r + Q_r)
        out_r = self.out_linear_r(attn_ln_r) + self.out_l_r(attn_ln_r)
        attn_ln_i = self.ln_attn_i(attn_i + Q_i)
        out_i = self.out_linear_i(attn_ln_i) + self.out_l_i(attn_ln_i)

        return out_r, out_i
