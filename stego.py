#PRIPREMA PODATAKA#
import cv2
import numpy as np

img = cv2.imread("image.png")
watermark = cv2.imread("watermark.png", cv2.IMREAD_GRAYSCALE)

img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

img_y = img_ycbcr[:,:,0]
img_y = img_y.astype(np.float64)
#print(img_y.shape)

#print(watermark.shape)   
#print(watermark)

#STVARANJE KLJUČA I PERMUTIRANJE WATERMARKA#
import hashlib

def permute_watermark(wm, key):
    wm_bits = (wm > 127).astype(np.uint8)
    flat = wm_bits.flatten()  
    seed = int(hashlib.sha256(key.encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
   
    N = flat.size
    perm = rng.permutation(N)  
    
    
    permuted_flat = flat[perm]
   
    
    permuted_wm = permuted_flat.reshape(32, 32).astype(np.uint8)

    return permuted_wm, perm

key = "Metroflex"

permuted_wm, perm = permute_watermark(watermark, key)

image = (permuted_wm*255).astype(np.uint8)
#cv2.imwrite("permutedwatermark.png", image)

#PARTICIONIRANJE SLIKE#
img_tl = img_y[0:256,0:256]
img_br = img_y[256:512,256:512]
#print(img_tl.shape)
#print(img_br.shape)
M = 8
num_blocks = 1024

#SVD na top left subimage-u#
import math
import matplotlib.pyplot as plt
from numpy.linalg import svd
T = 20
cols = rows = 256 // 8
Dlarge = np.zeros((rows,cols), dtype=np.float64)
Ustore = {}
Vstore = {}
Dstore = {}
for i in range(rows):
    for j in range(cols):
        r = i * M
        c = j * M
        block = img_tl[r:r+M,c:c+M]
        U,D,V = svd(block)
        Dlarge[i,j] = D[0]
        Ustore[(i,j)] = U
        Vstore[(i,j)] = V
        Dstore[(i,j)] = D
dmin = np.min(Dlarge)
dmax = np.max(Dlarge)
bins = np.arange(dmin - T, dmax + T, T)

def quantize_d(dval, bit):
        idx = np.searchsorted(bins, dval, side='right') - 1
        if idx < 0: idx = 0
        if idx >= len(bins)-1: idx = len(bins)-2
        dlow = bins[idx]; dhigh = bins[idx+1]; mid = 0.5*(dlow + dhigh)
        if bit == 1:
            newd = 0.5*(dlow + mid)
        else:
            newd = 0.5*(mid + dhigh)
        return newd, (dlow, dhigh)
Dlarge_mod = Dlarge
for i in range(rows):
    for j in range(cols):
        bit = int(permuted_wm[i,j])
        Dlarge_mod[i,j], _ = quantize_d(Dlarge[i,j], bit)
img_tl_w = np.zeros_like(img_tl, dtype=np.float64)
for i in range(rows):
    for j in range(cols):
        U = Ustore[(i,j)]
        V = Vstore[(i,j)]
        D = Dstore[(i,j)]
        D[0] = Dlarge_mod[i,j]
        block_mod = (U * D) @ V
        r = i * M; c = j * M
        img_tl_w[r:r+M,c:c+M] = block_mod
print(img_tl_w)
print(img_tl)

img_y[0:256,0:256] = img_tl_w
img_ycbcr[:,:,0] = img_y
bgr_restored = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCrCb2BGR)

cv2.imwrite("permuted_image.png", bgr_restored)

