caffe_root = 'caffe/'
path_to_img = "/images"
mean        = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
deploy      = caffe_root + 'models/bvlc_reference_caffenet/deploy_feature.prototxt'
model       = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
feat_layer = 'fc6wi'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', metavar='Inputs', type=str, default='filenames.npy',
                   help='npy filename containing image filenames')
parser.add_argument('-o', metavar='Outputs', type=str, default='features.npy',
                    help='npy filename wirtes extracted features in')
parser.add_argument('-g', metavar='Use GPU', type=str, default=-1,
                    help='GPU device ID (CPU if this negative)')
args = parser.parse_args()

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
if args.g < 0:
    caffe.set_mode_cpu()

net = caffe.Net(deploy, model, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(mean).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

images = np.load(args.i)
N = len(images)
net.blobs['data'].reshape(N,3,227,227)
for i in range(N):
    img = path_to_img + images[i]
    net.blobs['data'].data[i] = \
        transformer.preprocess('data', caffe.io.load_image(img))
net.forward()
np.save(args.o, net.blobs[feat_layer].data)
