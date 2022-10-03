# coding: utf-8

import torch
import torch.nn as nn


class Extraction(nn.Module):
  """
    iteration
  """

  def __init__(self, A_now, B, channels):
    super(Extraction, self).__init__()
    self.A_now = A_now
    self.B = B
    self.bn1 = nn.BatchNorm3d(channels)
    self.relu = nn.ReLU(inplace=True)
    self.bn2 = nn.BatchNorm3d(channels)

  def forward(self, out):
    u, f = out
    out = f - self.bn1(self.relu(self.A_now(u)))
    out = self.bn2(u + self.relu(self.B(out)))
    out = (out, f)
    return out


class Conv_block_left(nn.Module):
    """
        LSB
    """

  def __init__(self, A_now, channels):
    super(Conv_block_left, self).__init__()
    B_conv_1 = nn.Conv3d(channels, channels, kernel_size=3,
                         stride=1, padding=1, bias=False)
    self.extraction_1 = Extraction(A_now, B_conv_1, channels)

    B_conv_2 = nn.Conv3d(channels, channels, kernel_size=3,
                         stride=1, padding=1, bias=False)
    self.extraction_2 = Extraction(A_now, B_conv_2, channels)

    B_conv_3 = nn.Conv3d(channels, channels, kernel_size=3,
                         stride=1, padding=1, bias=False)
    self.extraction_3 = Extraction(A_now, B_conv_3, channels)

  def forward(self, u_f):
    out = self.extraction_1(u_f)
    out = self.extraction_2(out)
    out = self.extraction_3(out)
    return out    


class Restriction(nn.Module):

  def __init__(self, A_now, A_next, channels):
    super(Restriction, self).__init__()
    self.Pai = nn.Conv3d(channels, channels, 3, 2, 1)
    self.A_now = A_now
    self.A_next = A_next

    self.R = nn.Conv3d(channels, channels, 3, 2, 1)
    self.relu = nn.ReLU(inplace=True)
    self.bn = nn.BatchNorm3d(channels)

  def forward(self, out):
    u, f = out
    u_next = self.Pai(u)
    f = self.R(f-self.A_now(u)) + self.A_next(u_next)
    f = self.relu(f)
    f = self.bn(f)

    return (u_next, f)


class Conv_block_right(nn.Module):
  """
        RSB
  """

  def __init__(self, A_now, channels):
    super(Conv_block_right, self).__init__()
    B_conv_1 = nn.Conv3d(channels, channels, kernel_size=3,
                         stride=1, padding=1, bias=False)
    self.extraction_1 = Extraction(A_now, B_conv_1, channels)

    B_conv_2 = nn.Conv3d(channels, channels, kernel_size=3,
                         stride=1, padding=1, bias=False)
    self.extraction_2 = Extraction(A_now, B_conv_2, channels)

  def forward(self, u_f):
    out = self.extraction_1(u_f)
    out = self.extraction_2(out)
    return out[0]


class Conv_block_last(nn.Module):
  """
    CSB
  """

  def __init__(self, A_now, channels):
    super(Conv_block_last, self).__init__()
    B_conv_1 = nn.Conv3d(channels, channels, kernel_size=3,
                         stride=1, padding=1, bias=False)
    self.extraction_1 = Extraction(A_now, B_conv_1, channels)

    B_conv_2 = nn.Conv3d(channels, channels, kernel_size=3,
                         stride=1, padding=1, bias=False)
    self.extraction_2 = Extraction(A_now, B_conv_2, channels)

    B_conv_3 = nn.Conv3d(channels, channels, kernel_size=3,
                         stride=1, padding=1, bias=False)
    self.extraction_3 = Extraction(A_now, B_conv_3, channels)

    B_conv_4 = nn.Conv3d(channels, channels, kernel_size=3,
                         stride=1, padding=1, bias=False)
    self.extraction_4 = Extraction(A_now, B_conv_4, channels)

    B_conv_5 = nn.Conv3d(channels, channels, kernel_size=3,
                         stride=1, padding=1, bias=False)
    self.extraction_5 = Extraction(A_now, B_conv_5, channels)

  def forward(self, u_f):
    out = self.extraction_1(u_f)
    out = self.extraction_2(out)
    out = self.extraction_3(out)
    out = self.extraction_4(out)
    out = self.extraction_5(out)

    return out[0]


class Right_u_init(nn.Module):

  def __init__(self, channels):
    super(Right_u_init, self).__init__()
    self.upsample = nn.ConvTranspose3d(channels, channels,
                                       kernel_size=3, stride=2, padding=1, output_padding=1)

  def forward(self, coarseU_initU_refineU):
    out = coarseU_initU_refineU[0] - coarseU_initU_refineU[1]
    out = self.upsample(out)
    out = coarseU_initU_refineU[2] + out

    return out


class FASUNet_3D(nn.Module):


  def __init__(self, in_dim, out_dim, init_c=16):
    super(FASUNet_3D, self).__init__()

    self.conv_start = nn.Conv3d(
        in_dim, init_c, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn_0 = nn.BatchNorm3d(init_c)


    A_1 = nn.Conv3d(init_c, init_c, kernel_size=3,
                    stride=1, padding=1, bias=False)
    self.encode_1 = Conv_block_left(A_1, init_c)  

    A_2 = nn.Conv3d(init_c, init_c, kernel_size=3,
                    stride=1, padding=1, bias=False)
    self.restriction_1_2 = Restriction(A_1, A_2, init_c)  


    self.encode_2 = Conv_block_left(A_2, init_c)
    A_3 = nn.Conv3d(init_c, init_c, kernel_size=3,
                    stride=1, padding=1, bias=False)
    self.restriction_2_3 = Restriction(A_2, A_3, init_c)


    self.encode_3 = Conv_block_left(A_3, init_c)
    A_4 = nn.Conv3d(init_c, init_c, kernel_size=3,
                    stride=1, padding=1, bias=False)
    self.restriction_3_4 = Restriction(A_3, A_4, init_c)


    self.encode_4 = Conv_block_last(A_4, init_c)
    A_5 = nn.Conv3d(init_c, init_c, kernel_size=3,
                    stride=1, padding=1, bias=False)


    self.get_decoce_3_init_u = Right_u_init(init_c)
    self.decode_3 = Conv_block_right(A_5, init_c)


    A_6 = nn.Conv3d(init_c, init_c, kernel_size=3,
                    stride=1, padding=1, bias=False)

    self.get_decoce_2_init_u = Right_u_init(init_c)
    self.decode_2 = Conv_block_right(A_6, init_c)


    A_7 = nn.Conv3d(init_c, init_c, kernel_size=3,
                    stride=1, padding=1, bias=False)
    self.get_decoce_1_init_u = Right_u_init(init_c)
    self.decode_1 = Conv_block_right(A_7, init_c)


    self.final_1 = nn.Conv3d(init_c, out_dim, 3, 1, 1, bias=False)

  def forward(self, f):
    f = self.conv_start(f)
    f = self.bn_0(f)
    out = (f, f)


    f_1 = out[1]
    out = self.encode_1(out)              
    u_12 = out[0]
    out = self.restriction_1_2(out)    


    u_20 = out[0]
    f_2 = out[1]
    out = self.encode_2(out)
    u_22 = out[0]
    out = self.restriction_2_3(out)


    u_30 = out[0]
    f_3 = out[1]
    out = self.encode_3(out)
    u_32 = out[0]
    out = self.restriction_3_4(out)


    u_40 = out[0]
    out = self.encode_4(out)


    out = self.get_decoce_3_init_u((out, u_40, u_32))
    out = self.decode_3((out, f_3))


    out = self.get_decoce_2_init_u((out, u_30, u_22))
    out = self.decode_2((out, f_2))


    out = self.get_decoce_1_init_u((out, u_20, u_12))
    out = self.decode_1((out, f_1))
    out1 = self.final_1(out)

    return out1


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = "1"
  
  x = torch.randn(1, 1, 32, 32, 32).to("cuda")

  model = FASUNet_3D(in_dim=1, out_dim=2, init_c=32).to("cuda")
  p1 = model(x)
  
  
  from torchsummaryX import summary
  summary(model, x)
  print("the number of paremeters：", count_parameters(model))