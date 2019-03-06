'''
Adapted from https://github.com/dshieble/Tensorflow_Deep_Taylor_LRP/blob/master/lrp.py
'''

import tensorflow as tf
from tensorflow.python.ops import nn_ops, gen_nn_ops

def lrp(F, input_lower_bound, input_upper_bound,
        graph=None, return_flist=False,
        conv_strides=None):
    '''
        Accepts a final output, and propagates back from there to compute LRP over a tensorflow graph.
        Performs a Taylor Decomp at each layer to assess the relevances of each neuron at that layer
    '''
    F_list = []
    traversed, graph, graph_dict, var_dict = _get_traversed(graph=graph)
    exit()
    #
    for nd in traversed:
        if (graph_dict[nd].op == "MatMul")\
                or (graph_dict[nd].op == "Conv2D"):
            val_name = next(ipt for ipt in (graph_dict[nd].input) \
                            if ipt in traversed).split("/read")[0]\
                            + ":0"
            X = graph.get_tensor_by_name(val_name)
            weight_name = next(
                        ipt for ipt in (graph_dict[nd].input) if ipt not in traversed\
                        ).split("/read")[0] + ":0"
            #W = var_dict[weight_name]
            W = graph.get_tensor_by_name(weight_name)
            
            # compute
            if graph_dict[nd].op == "MatMul":
                F = _fprop(F, W, X)
                F_list.append(F)
            elif graph_dict[nd].op == "Conv2D":
                if ("conv1/conv1d/ExpandDims" in graph_dict[nd].input):
                    #F = _fprop_conv_first(F, W, X,
                    #        input_lower_bound, input_upper_bound,
                    #        conv_strides, padding="SAME")
                    F = _fprop_conv(F, W, X, conv_strides)

                    fshape = F.get_shape().as_list()
                    F = tf.reshape(F, (-1, fshape[2], fshape[3]))
                    F_list.append(F)
                    break
                else:
                    F = _fprop_conv(F, W, X, conv_strides)
                    F_list.append(F)
            
    if return_flist:
        return F_list
    else:
        return F


### private functions
def _get_traversed(graph=None):
    '''
        Get the graph and graph traveral
    '''
    graph = tf.get_default_graph() if graph is None else graph

    # save graph nodes into a dictionary
    # e.g.
    # node: name, e.g. "conv5/batch_normalization/beta"
    #       op: "VariableV2"
    #       key: "_class"
    #       key: "container"
    #       key: "dtype"
    #       key: "shape"
    #       key: "shared_name"
    graph_dict = {node.name: node for node in graph.as_graph_def().node}

    # acquire all the variables in the model
    # e.g.
    # u'conv1/biases:0': <tf.Tensor 'conv1/biases/read:0' shape=(512,) dtype=float32>
    var_dict = {v.name:v.value() for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}

    # find the data flow
    # e.g.
    # [u'absolute_output', u'fc3/Add', u'fc3/MatMul', u'fc2/Relu', ... ]
    path = _traverse(graph_dict['absolute_output'], graph_dict)
    path = path[::-1]
    print (path)
    return path, graph, graph_dict, var_dict

def _traverse(node, graph_dict):
    '''
        Depth first search the network graph
    '''
    #path.append(node.name)
    if 'absolute_input' in node.name:
        path = [node.name]
    else:
        path = None

    inputs = node.input
    #print (node.name, node.input)
    for nodename in inputs:
        if nodename in graph_dict:
            path = _traverse(graph_dict[nodename], graph_dict)
            if path != None:
                path.append(node.name)
                break # Because there is only one path from inputs to outputs, for now.
        else:
            print (nodename)
    return path

def _fprop(F, W, X):
    '''
        Propagate over feedforward layer
    '''
    V = tf.maximum(0.0, W)
    Z = tf.matmul(X, V) + 1e-9
    S = F/Z
    C = tf.matmul(S, tf.transpose(V))
    F = X*C
    return F

def _fprop_conv(F, W, X, strides, padding="SAME"):
    xshape = X.get_shape().as_list()
    fshape = F.get_shape().as_list()
    if len(xshape) != len(fshape):
        F = tf.reshape(F, (-1, xshape[1]/strides[1], xshape[2]/strides[2],
                        fshape[-1]*strides[1]*strides[2]/(xshape[1]*xshape[2])))
    strides = [1,1,1,1] if strides is None else strides

    V = tf.maximum(0.0, W)
    Z = tf.nn.conv2d(X, V, strides, padding) + 1e-9
    S = F/Z
    C = nn_ops.conv2d_backprop_input(tf.shape(X), V, S, strides, padding)
    F = X*C
    return F

def _fprop_conv_first(F, W, X, lowest, highest, strides, padding="SAME"):
    strides = [1,1,1,1] if strides is None else strides

    Wn = tf.minimum(-1e-9, W)
    Wp = tf.maximum(1e-9, W)

    X, L, H = X+1e-9, lowest, highest
    c  = tf.nn.conv2d(X, W, strides, padding)
    cp = tf.nn.conv2d(H, Wp, strides, padding)
    cn = tf.nn.conv2d(L, Wn, strides, padding)
    Z = c - cp - cn + 1e-9
    S = F/Z

    g  = nn_ops.conv2d_backprop_input(tf.shape(X), W,  S, strides, padding)
    gp = nn_ops.conv2d_backprop_input(tf.shape(X), Wp, S, strides, padding)
    gn = nn_ops.conv2d_backprop_input(tf.shape(X), Wn, S, strides, padding)
    F = X*g - L*gp - H*gn
    return F
