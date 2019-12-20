import numpy as np
import cv2
import argparse

def extract_line(im,wr=8,diff_thresh=0.2,cnt_thresh=20,dtype=np.float64):
    H,W = im.shape
    ws = 2*wr + 1
    pad_im = np.zeros([H+2*wr,W+2*wr])
    pad_im[wr:wr+H,wr:wr+W] = im

    im_ws = np.zeros([H,W,ws*ws])
    for i in range(ws):
        for j in range(ws):
            im_ws[:,:,i*ws+j] =  pad_im[i:i+H,j:j+W]

    diff = im_ws - im[:,:,np.newaxis]
    count = np.sum((diff>diff_thresh).astype(np.int32) - (diff<-diff_thresh).astype(np.int32),2)
    line= count<cnt_thresh
    return line.astype(dtype)

def extract_sketch(im,wr=8,diff_thresh=0.2,cnt_thresh=20,dark_thresh=0.1,dtype=np.float64):
    line = extract_line(im,wr,diff_thresh,cnt_thresh,dtype)
    black = im>dark_thresh
    edge = 1 - cv2.Canny(im.astype(np.uint8),100,200).astype(np.float64)/255.0
    sketch = np.logical_and(np.logical_and(black,line),edge)
    return sketch.astype(dtype)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Cartoon sketch extractor')
	parser.add_argument('--path', type=str, default=None, help='Color image path')
	parser.add_argument('--wr', type=int, default=5, help='Window radius')
	parser.add_argument('--diffth', type=float, default=0.2, help='Diff threshold')
	parser.add_argument('--cntth', type=int, default=40, help='Count thrshold')
	parser.add_argument('--darkth', type=float, default=0.1, help='Dark threshold')
	args = parser.parse_args()

	im8U = cv2.imread(args.path)
	H,W,C = im8U.shape
	imgray8U = cv2.cvtColor(im8U,cv2.COLOR_BGR2GRAY)
	imgray64F = imgray8U.astype(np.float64)/255.0
	sketch = extract_sketch(imgray64F,args.wr,args.diffth,args.cntth,args.darkth,np.float64)

	cv2.imshow("Original image", im8U)
	
	cv2.imshow("Sketch image",sketch)

	cv2.waitKey(0)