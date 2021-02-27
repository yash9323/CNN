# Convolutional Neural Networks From scratch

#Loading the dependencies 
import numpy as np 
import h5py
import matplotlib.pyplot as plt 
%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)

# Zero padding function 
def zero_pad(X, pad):
    X_pad=np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values=(0,0))
    return X_pad

#testing zero pad function 
x=[[[[ 1.62434536, -0.61175641],
         [-0.52817175, -1.07296862],
         [ 0.86540763, -2.3015387 ]],

        [[ 1.74481176, -0.7612069 ],
         [ 0.3190391 , -0.24937038],
         [ 1.46210794, -2.06014071]],

        [[-0.3224172 , -0.38405435],
         [ 1.13376944, -1.09989127],
         [-0.17242821, -0.87785842]]]]    

x=np.array(x)
ab=zero_pad(x,2) # padding with 2 
print(x.shape)
print(ab.shape)
plt.imshow(x[0,:,:,0])
plt.imshow(ab[0,:,:,0])

# Single Step in Conv 
def single_single_step(prev,W,b):
    s=np.multiply(prev,W) #Multiplying the weights matrix with the new values 
    z=np.sum(s)# adding them
    z+=float(b)# adding bias to calculate z
    return z 

#Testing the single conv_function
#using ab as the prev and initializing the weights matrix
W=np.random.randn(7,7,2)
b=np.random.randn(1,1,1)
z=single_single_step(ab[0],W,b)
print("z=",z)

#Conv forward pass to use multiple filters and stack the output together 
def conv_forward(arr,W,b,hyp):
    (m,n_h_prev,n_w_prev,n_c_prev)=arr.shape
    (f,f,n_c_prev,n_c)=W.shape
    stride=hyp['stride']
    pad=hyp['pad']
    n_H=int((n_h_prev-f+(2*pad))/stride + 1 )
    n_w=int((n_w_prev-f+(2*pad))/stride + 1 )
    z=np.zeros((m,n_H,n_w,n_c))
    arr_pad=zero_pad(arr,pad)
    
    for i in range(m):
        Arr_pad=arr[i]
        for h in range(n_H):
            ver_start=stride*h
            vert_end=stride*h + f
            for w in range(n_w):
                hori_start=stride*w
                hori_end=stride*w+f
                for c in range(n_c):
                    prev=arr_pad[i,ver_start:vert_end,hori_start:hori_end,:]
                    Weights=W[:,:,:,c]
                    bias=b[:,:,:,c]
                    z[i,h,w,c]=single_single_step(prev,Weights,bias)

    assert(z.shape==(m,n_H,n_w,n_c))
    cache=(arr,W,b,hyp)
    return z,cache

#TESTING VONV FORWARD 
A_prev = np.random.randn(10,5,7,4)
W = np.random.randn(3,3,4,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad" : 1,
               "stride": 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
print("Z's mean =\n", np.mean(Z))
print("Z[3,2,1] =\n", Z[3,2,1])
print("cache_conv[0][1][2][3] =\n", cache_conv[0][1][2][3])

#Pooling Layer 
def pool_forward(A_prev, hparameters, mode = "max"):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))              
    for i in range(m):                         
        for h in range(n_H):                    
            vert_start = stride*h
            vert_end = stride*h+f
            for w in range(n_W):                 
                horiz_start = stride*w
                horiz_end = stride*w+f
                for c in range (n_C):          
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.average(a_prev_slice)
    cache = (A_prev, hparameters)
    assert(A.shape == (m, n_H, n_W, n_C))
    return A, cache

np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
hparameters = {"stride" : 1, "f": 1}

A,cache=pool_forward(A_prev,hparameters)
fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('Image')
axarr[0].imshow(A_prev[0,:,:,:])
axarr[1].set_title('Pooled')
axarr[1].imshow(A[0,:,:,:])
