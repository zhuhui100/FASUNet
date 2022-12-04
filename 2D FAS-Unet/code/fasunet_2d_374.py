"""
        fas-unet 2D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F




        
class Extraction(nn.Module):
    """
    iteration
    """
    def __init__(self, A_now, B, channels):
        super(Extraction,self).__init__()
        self.A_now = A_now
        self.B = B
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        u, f = x

        x = F.relu(self.bn1(f - self.A_now(u)))
        x = F.relu(self.bn2(self.B(x)) + u  )

        x = (x, f)
        return x


class Conv_block_left(nn.Module):

    def __init__(self, A_now, channels):
        super(Conv_block_left, self).__init__()
        B_conv_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_1 = Extraction(A_now, B_conv_1, channels)

        B_conv_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_2 = Extraction(A_now, B_conv_2, channels)

        B_conv_3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_3 = Extraction(A_now, B_conv_3, channels)

    def forward(self, u_f):
        u_f = self.extraction_1(u_f)
        u_f = self.extraction_2(u_f)
        u_f = self.extraction_3(u_f)
        return u_f    # new_u and f


class Restriction(nn.Module):

    def __init__(self, A_now, A_next, channels):
        super(Restriction,self).__init__()
        self.Pai = nn.Conv2d(channels, channels, 3, 2, 1, bias=False)
        self.A_now = A_now
        self.A_next = A_next

        self.R = nn.Conv2d(channels, channels, 3, 2, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, out):
        u, f = out
        del out
        # Pool u
        u_next = self.Pai(u)

        # update f. follow:  pool(f) - pool( a_now(u) ) + a_next( pool(u_next) )
        f = self.R(f-self.A_now(u)) + self.A_next(u_next)
        del u
        f = self.relu(f)
        f = self.bn(f)

        return (u_next, f)


class Conv_block_right(nn.Module):

    def __init__(self, A_now, channels):
        super(Conv_block_right, self).__init__()
        B_conv_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_1 = Extraction(A_now, B_conv_1, channels)

        B_conv_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_2 = Extraction(A_now, B_conv_2, channels)

        B_conv_3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_3 = Extraction(A_now, B_conv_3, channels)
        
        B_conv_4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_4 = Extraction(A_now, B_conv_4, channels)

        
    def forward(self, u_f):
        u_f = self.extraction_1(u_f)
        u_f = self.extraction_2(u_f)
        u_f = self.extraction_3(u_f)
        u_f = self.extraction_4(u_f)
        return u_f[0]


class Conv_block_last(nn.Module):

    def __init__(self, A_now, channels):
        super(Conv_block_last, self).__init__()
        B_conv_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_1 = Extraction(A_now, B_conv_1, channels)

        B_conv_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_2 = Extraction(A_now, B_conv_2, channels)

        B_conv_3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_3 = Extraction(A_now, B_conv_3, channels)

        B_conv_4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_4 = Extraction(A_now, B_conv_4, channels)

        B_conv_5 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_5 = Extraction(A_now, B_conv_5, channels)

        B_conv_6 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_6 = Extraction(A_now, B_conv_6, channels)

        B_conv_7 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.extraction_7 = Extraction(A_now, B_conv_7, channels)

    def forward(self, u_f):
        u_f = self.extraction_1(u_f)
        u_f = self.extraction_2(u_f)
        u_f = self.extraction_3(u_f)
        u_f = self.extraction_4(u_f)
        u_f = self.extraction_5(u_f)
        u_f = self.extraction_6(u_f)
        u_f = self.extraction_7(u_f)

        return u_f[0]


class Right_u_init(nn.Module):

    def __init__(self, channels):
        super(Right_u_init, self).__init__()
        self.upsample = nn.ConvTranspose2d(channels, channels, 
                                kernel_size=3, stride=2, padding=1, output_padding=1)
    
    def forward(self, coarseU_initU_refineU):
        # error
        out = coarseU_initU_refineU[0] - coarseU_initU_refineU[1]
        out = self.upsample(out)

        # u + u_0
        out = coarseU_initU_refineU[2] + out

        return out


class FASUNet(nn.Module):
    """
     fas-unet
    """
    def __init__(self, in_dim, n_classes, init_c=16):
        super(FASUNet, self).__init__()
        
        self.conv_start = nn.Conv2d(in_dim, init_c, kernel_size=3,stride=1,padding=1, bias=False)
        self.bn_0 = nn.BatchNorm2d(init_c)
        
        ### encode 1
        E1_A = nn.Conv2d(init_c, init_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.encode_1 = Conv_block_left(E1_A, init_c) 

        E2_A = nn.Conv2d(init_c, init_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.restriction_1_2 = Restriction(E1_A, E2_A, init_c)   

        ### encode 2
        self.encode_2 = Conv_block_left(E2_A, init_c)
        E3_A = nn.Conv2d(init_c, init_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.restriction_2_3 = Restriction(E2_A, E3_A, init_c)      

        ### encode 3
        self.encode_3 = Conv_block_left(E3_A, init_c)
        E4_A = nn.Conv2d(init_c, init_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.restriction_3_4 = Restriction(E3_A, E4_A, init_c)

        ### encode 4
        self.encode_4 = Conv_block_left(E4_A, init_c)
        E5_A = nn.Conv2d(init_c, init_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.restriction_4_5 = Restriction(E4_A, E5_A, init_c)

        ### encode 5
        self.encode_5 = Conv_block_last(E5_A, init_c)
        
        ### decode 4
        D4_A = nn.Conv2d(init_c, init_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.get_decoce_4_init_u = Right_u_init(init_c)
        self.decode_4 = Conv_block_right(D4_A, init_c)

        ### decode 3
        D3_A = nn.Conv2d(init_c, init_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.get_decoce_3_init_u = Right_u_init(init_c)
        self.decode_3 = Conv_block_right(D3_A, init_c)

        # ### decode 2
        D2_A = nn.Conv2d(init_c, init_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.get_decoce_2_init_u = Right_u_init(init_c)
        self.decode_2 = Conv_block_right(D2_A, init_c)

        ### decode 1
        D1_A = nn.Conv2d(init_c, init_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.get_decoce_1_init_u = Right_u_init(init_c)
        self.decode_1 = Conv_block_right(D1_A, init_c)

        ###### stage 2
        self.final_1 = nn.Conv2d(init_c, n_classes, 3, 1, 1, bias=False)

    def forward(self, f):
    
        #################### stage 1
        # get initialization feature
        f = self.conv_start(f)
        f = self.bn_0(f)
        out = (f, f)
        
        ## encode layer 1
        f_1 = out[1]  
        out = self.encode_1(out)              
        u_12 = out[0]
        out = self.restriction_1_2(out)       

        ## encode layer 2   
        u_20 = out[0]    
        f_2 = out[1]                   
        out = self.encode_2(out)  
        u_22 = out[0]      
        out = self.restriction_2_3(out)       

        ## encode layer 3
        u_30 = out[0]
        f_3 = out[1]
        out = self.encode_3(out)   
        u_32 = out[0]          
        out = self.restriction_3_4(out)

        ## encode layer 4
        u_40 = out[0]
        f_4 = out[1]
        out = self.encode_4(out)   
        u_42 = out[0]          
        out = self.restriction_4_5(out)

        ### encode layer 5
        u_50 = out[0]
        out = self.encode_5(out)   
        # p4 = out  

        ### decode layer  4
        out = self.get_decoce_4_init_u((out, u_50, u_42))  
        out = self.decode_4((out, f_4))

        ### decode layer  3
        out = self.get_decoce_3_init_u((out, u_40, u_32))  
        out = self.decode_3((out, f_3))      
        # p3 = out            

        ### decode layer 2
        out = self.get_decoce_2_init_u((out, u_30, u_22)) 
        out = self.decode_2((out, f_2))    

        ### decode layer 1
        out = self.get_decoce_1_init_u((out, u_20, u_12))
        out = self.decode_1((out, f_1))     

        # stage 2
        out = self.final_1(out)


        return out



        

if __name__ == "__main__":

    f = torch.zeros((1, 1, 32, 32))
    u = f

    model = FASUNet(1, 5, init_c=64)

    out = model(f)
    print(out.size())

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

    from torchsummaryX import summary
    summary(model, f)
    print("number of parameters：", count_parameters(model))
