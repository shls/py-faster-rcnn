
# Make sure that caffe is on the python path:
caffe_root = '/home/ls/py-faster-rcnn/caffe-fast-rcnn/'  # this file is expected to be in {caffe_root}/examples
rcnn_lib = '/home/ls/py-faster-rcnn/lib/'

import sys
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, rcnn_lib)

import caffe
	
# Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('/home/ls/py-faster-rcnn/net_surgery/original.prototxt','/home/ls/py-faster-rcnn/data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel', caffe.TRAIN)
params_m = ['conv1']
params_s = ['conv2','conv3','conv4','conv5', 'rpn_conv/3x3', 'rpn_cls_score', 'rpn_bbox_pred', 'fc6','fc7','cls_score', 'bbox_pred']

# fc_params = {name: (weights, biases)}
fc_params_m = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params_m}
fc_params_s = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params_s}

# Load New net
new_net = caffe.Net('/home/ls/py-faster-rcnn/net_surgery/sdha4.prototxt', caffe.TRAIN)
new_params_m = ['conv1']
new_params_s = ['conv2','conv3','conv4','conv5', 'rpn_conv/3x3', 'rpn_cls_score', 'rpn_bbox_pred', 'fc6','fc7','cls_score', 'bbox_pred']

# fc_params = {name: (weights, biases)}
new_fc_params_m = {pr: (new_net.params[pr][0].data, new_net.params[pr][1].data) for pr in new_params_m} 
new_fc_params_s = {pr: (new_net.params[pr][0].data, new_net.params[pr][1].data) for pr in new_params_s}

# modify part
# give B to Channel 4
for pr, new_pr in zip(params_m, new_params_m):
	for n in range (0, fc_params_m[pr][0].shape[0]):
		for c_i in range (0, fc_params_m[pr][0].shape[1]):
			new_fc_params_m[new_pr][0][n][c_i] = fc_params_m[pr][0][n][c_i]
		new_fc_params_m[new_pr][0][n][3] = fc_params_m[pr][0][n][0]
	new_fc_params_m[new_pr][1][...]=fc_params_m[pr][1]

# same part
for pr, new_pr in zip(params_s, new_params_s):
	new_fc_params_s[new_pr][0][...]=fc_params_s[pr][0] #weights
	new_fc_params_s[new_pr][1][...]=fc_params_s[pr][1] #bias

new_net.save('/home/ls/py-faster-rcnn/net_surgery/VGG_CNN_M_1024.v2.sdha.bgrb.caffemodel')

print 'channel convert is done'