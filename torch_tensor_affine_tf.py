import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class affine_ap(nn.Module):
    def __init__(self, img_size, flip_lr, flip_ud, theta, tr_x, tr_y):
        super().__init__()
        self.theta = theta*2
        self.flip_lr = flip_lr
        self.flip_ud = flip_ud
        self.tr_x = tr_x/img_size[0]*2
        self.tr_y = tr_y/img_size[1]*2

    def m_affine(self, N):
        
        m = []
        for i in range(N):
            i = torch.eye(3)
            if self.theta != 0:
                t = np.pi/180. * ((np.random.rand() - 0.5) * self.theta)

                i[:2, :2] = torch.tensor([[np.cos(t), -1.0*np.sin(t)],
                                          [np.sin(t), np.cos(t)]])
            # if self.flip_lr:
            #     if torch.rand(1) > 0.5:
            #         i *= torch.tensor([-1, 0, 0, \
            #                             0, 1, 0, \
            #                             0, 0, 1]).view(3,3)
            # if self.flip_ud:
            #     if torch.rand(1) > 0.5:
            #         i *= torch.tensor([1, 0, 0, \
            #                             0, -1, 0, \
            #                             0, 0, 1]).view(3,3)
            if self.tr_x != 0 and self.tr_y != 0:
                tr_x1 = (torch.rand(1) - 0.5)*self.tr_x
                tr_y1 = (torch.rand(1) - 0.5)*self.tr_y
                i += torch.tensor([0, 0, tr_x1, \
                                    0, 0, tr_y1, \
                                    0, 0, 0]).view(3,3)
            m.append(i[:2, :])
        m = torch.stack(m, 0)
        return m

    def forward(self, x):
        with torch.no_grad():
            N,C,H,W = x.size()

            self.affine_matrix = self.m_affine(N).type_as(x)

            if self.flip_lr:
                x = torch.flip(x, [3])
            if self.flip_ud:
                x = torch.flip(x, [2])
                
            grids = F.affine_grid(self.affine_matrix, [N,C,H,W], align_corners=True)
            x = F.grid_sample(x, grids, align_corners=True, padding_mode='zeros')
        
        return x
        
def check(w=400, h=400, c0=0, c1=1, blocksize=4):
    return np.kron([[1, 0] * blocksize, [0, 1] * blocksize] * blocksize, np.ones((h//blocksize//2, w//blocksize//2)))


if __name__ == "__main__":
    # i = torch.4([-1, 0, 0, 0, 1, 0, 0, 0, 1]).view(3,3)
    NT = affine_ap([400,400], True, True, 50, 0, 0)
    txtn = check()
    txt = torch.tensor(txtn, dtype=torch.float32)
    txt = txt.view(1,1,400,400)
    new = NT(txt).numpy().squeeze()


    theta = torch.zeros(1, 2, 3)
    angle = np.pi/4.
    theta[:, :, :2] = torch.tensor([[np.cos(angle), -1.0*np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
    theta[:, :, 2] = 0

    grid = F.affine_grid(theta, txt.size())
    new1 = F.grid_sample(txt, grid).numpy().squeeze()

    # i = NT.m_affine(10)
    # print(i.size(), i)
    # print(NT.affine_matrix)
    import matplotlib.pyplot as plt

    plt.imshow(np.stack([new, np.zeros_like(new), txtn], axis=-1))
    plt.show()
    pass
        
