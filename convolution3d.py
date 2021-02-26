import numpy as np
import cv2 #opencv
import skvideo
import skvideo.io
import skvideo.datasets


def myConv3D(A, B, param):
   
    # Flip the kernel
    B = np.flipud(np.fliplr(B))
    
    #Check for zero padding
    if A.shape == param:
        pad = 3 
        A = pad_image(A,pad)
    else:
        pad = 0
        A = A
    print(A) 
    print(A.shape)
    #Create an empty matrix    
    C = np.zeros_like(A)  
    
    # Gather Shapes of Kernel 
    xKernShape = B.shape[2]
    yKernShape = B.shape[1] 
    zKernShape = B.shape[0]

    
    for z in range(A.shape[0]):     
        
        for y in range(A.shape[1]):
          
            for x in range(A.shape[2]):
                
                    try:
                        
                        d_start = z
                        d_end = z + zKernShape      
                        vert_start = y
                        vert_end = y + yKernShape
                        horiz_start = x 
                        horiz_end = x + xKernShape
                        s = B[:,:,:]*A[d_start:d_end,  vert_start:vert_end, horiz_start:horiz_end]     
                        C[z, y, x] = np.sum(s)
                                
                    except:
                        break
                           
    return(C)

#Create kernel             
def create_smooth_kernel(size):
    
    matrix = np.ones((size, size,size), dtype="float") * (1.0 / (size **3))

    return(matrix)

# Add zero padding         
def pad_image(A, size):
    pad = (size-1)/2 #stride is 1
    pad = int(pad)
    Apad = np.pad(A, ((pad,pad), (pad,pad),(pad,pad)), 'constant', constant_values = (0,0))
    
    return(Apad)

def main():
	#Read the video using skvideo
    video = skvideo.io.vread('video.mp4') 
    print(video.shape)
    #Convert to grayscale
    grayframe = np.zeros_like(video[...,0])
    for i in range(video.shape[0]):
        grayframe[i,:,:]=cv2.cvtColor(video[i],cv2.COLOR_RGB2GRAY)
               
    print(grayframe.shape)
    kernel = create_smooth_kernel(3) #size=3
    print(kernel)
    print(kernel.shape)
    convv = myConv3D(grayframe,kernel,grayframe.shape)  #param
    print(convv.shape)
    #print(convv[20,15,15])
    
    #Save the video    
    skvideo.io.vwrite("output_video.mp4", convv)

if __name__ == "__main__":
    main()