# modified from https://github.com/xinntao/ESRGAN/blob/50fbd2de1d80a014e1c0e1c165913a128f5a8384/net_interp.py

import argparse
import caffe_pb2
import numpy as np


r"""
Network interpolator for waifu2x-caffe

It applies linear interpolation in the parameter space of two waifu2x-caffe models of the same architecture, 
which allows continuous imagery effect transition, e.g. adjusting the denoising strength.

caffe_pb2 is required, see the "protobuf" part of 
https://mxnet.incubator.apache.org/versions/master/faq/caffe.html#how-to-build

usage:
    network_interp.py -m1 ".\upconv_7_anime_style_art_rgb\scale2.0x_model.json.caffemodel" -m2 ".\upconv_7_anime_style_art_rgb\noise0_scale2.0x_model.json.caffemodel" --weight 0.5

ref: 
    [1] X. Wang, K. Yu, C. Dong, et al. Deep Network Interpolation for Continuous Imagery Effect Transition. CVPR 2019.
    [2] https://github.com/xinntao/DNI
"""

# parsing parameters
parser = argparse.ArgumentParser(description="Network interpolator for waifu2x-caffe")
parser.add_argument("-m1", "--model_1", type=str, required=True, help="the first model to interpolate (*.caffemodel)")
parser.add_argument("-m2", "--model_2", type=str, required=True, help="the second model to interpolate (*.caffemodel)")
parser.add_argument("-w", "--weight", type=float, required=True, help="weight used for interpolation [0-1]")
parser.add_argument("-o", "--output", type=str, default="interpolated.caffemodel", help="model output file name")
parser.add_argument("-v", "--verbose", type=bool, default=False, help="increase output verbosity")

args = parser.parse_args()

model_1_filename = args.model_1
model_2_filename = args.model_2
weight = args.weight
output_filename = args.output
verbose = args.verbose

# process
print(f"Loading {model_1_filename}\n")
proto_1 = caffe_pb2.NetParameter()
with open(model_1_filename, "rb") as f:
    proto_1.ParseFromString(f.read())

print(f"Loading {model_2_filename}\n")
proto_2 = caffe_pb2.NetParameter()
with open(model_2_filename, "rb") as f:
    proto_2.ParseFromString(f.read())


print(f"Start interpolation with weight={weight}:")
for idx, layer in enumerate(proto_2.layer):
    if len(layer.blobs) > 0:
        for i in range(len(layer.blobs)):
            tmp_1_data = np.asarray(proto_1.layer[idx].blobs[i].data)
            tmp_2_data = np.asarray(proto_2.layer[idx].blobs[i].data)

            assert tmp_1_data.shape == tmp_2_data.shape

            if verbose:
                print(f'Interpolating layer "{layer.name}": {layer.type}, size={tmp_1_data.shape}')

            proto_1.layer[idx].blobs[i].data[:] = (1.0 - weight) * tmp_1_data + weight * tmp_2_data
    else:
        if verbose:
            print(f'Skipping layer "{layer.name}": {layer.type}')

print(f"\nSaving interpolated model to {output_filename}")

with open(output_filename, "wb") as f:
    f.write(proto_1.SerializeToString())
