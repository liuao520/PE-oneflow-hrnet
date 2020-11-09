# import tensorflow as tf
import oneflow as flow

def _conv2d_layer(name,
                  input,
                  filters,
                  kernel_size=3,
                  strides=1,
                  padding="SAME",
                  # group_num=1,
                  data_format="NHWC",
                  dilation_rate=1,
                  # activation='Relu',
                  use_bias=False,
                  # use_bn=True,
                  weight_initializer=flow.random_uniform_initializer(),
                  bias_initializer=flow.random_uniform_initializer(),
                  trainable=True,
                  ):
    if data_format == "NCHW":
        weight_shape = (int(filters), int(input.shape[1]), int(kernel_size[0]), int(kernel_size[0]))
    elif data_format == "NHWC":
        weight_shape = (int(filters), int(kernel_size[0]), int(kernel_size[0]), int(input.shape[3]))
    else:
        raise ValueError('data_format must be "NCHW" or "NHWC".')
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
        trainable=trainable,
    )
    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
            model_name="bias",
            trainable=trainable,
        )
        output = flow.nn.bias_add(output, bias, data_format)

    # if activation is not None:
    #     if activation == 'Relu':
    #         output = flow.nn.relu(output)
    #     else:
    #         raise NotImplementedError

    return output


def _batch_norm(inputs, axis, momentum, epsilon, center=True, scale=True, training=True, name=None):
    # if trainable:
    #     training = True
    # else:
    #     training = False
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        # trainable=trainable,
        training=training,
        name=name
    )


def BasicBlock(name, inputs, filter_num, stride=1, training=None, **kwargs):
    # residual = inputs
    conv1 = _conv2d_layer(name+'conv1', inputs, filter_num, 3, stride, "SAME")
    bn1 = _batch_norm(conv1, momentum=0.1, epsilon=1e-5, training=training, name=name+'_bn1')
    relu = flow.nn.relu(bn1)
    conv2 = _conv2d_layer(name+'conv2', relu, filter_num, 3, 1, "SAME")
    bn2 = _batch_norm(conv2, momentum=0.1, epsilon=1e-5, training=training, name=name+'_bn2')

    if stride != 1:
        residual = _conv2d_layer(name+'basicdown', inputs, filter_num, 1, stride, "SAME")
        residual = _batch_norm(residual, momentum=0.1, epsilon=1e-5, name=name + '_bn_dasicdown')
    else:
        residual = inputs

    output = flow.nn.relu(flow.math.add(residual, bn2))
    return output


def BottleNeck(name, inputs, filter_num, stride=1, training=None, **kwargs):
    residual = _conv2d_layer(name+'bottledown', inputs, filter_num * 4, 1, stride, "SAME",use_bias=False)
    residual = _batch_norm(residual, momentum=0.1, epsilon=1e-5, name=name + '_bn_bottledown')

    conv1 = _conv2d_layer(name + 'conv1', inputs, filter_num, 1, 1, "SAME", use_bias=False)
    bn1 = _batch_norm(conv1, momentum=0.1, epsilon=1e-5, name=name + 'boottledown_bn1', training=training)
    relu1 = flow.nn.relu(bn1)
    conv2 = _conv2d_layer(name + 'conv2', relu1, filter_num, 3, stride, "SAME", use_bias=False)
    bn2 = _batch_norm(conv2, momentum=0.1, epsilon=1e-5, name=name + 'boottledown_bn2', training=training)
    relu2 = flow.nn.relu(bn2)
    conv3 = _conv2d_layer(name + 'conv2', relu2, filter_num, 1, 1, "SAME", use_bias=False)
    bn3 = _batch_norm(conv3, momentum=0.1, epsilon=1e-5, name=name + 'boottledown_bn3', training=training)

    output = flow.nn.relu(flow.math.add(residual, bn3))
    return output


def make_basic_layer(inputs, filter_num, blocks, stride=1):
    # res_block = []
    # res_block = flow.math.add(BasicBlock('make_basic0', inputs, filter_num, stride=stride), res_block)
    x = BasicBlock('make_basic0', inputs, filter_num, stride=stride)
    for _ in range(1, blocks):
        # res_block = flow.math.add(BasicBlock('make_basic', res_block, filter_num, stride=1), res_block)
        x = BasicBlock('make_basic', x, filter_num, stride=1)
    return x


def make_bottleneck_layer(inputs, filter_num, blocks, stride=1):
    # res_block = tf.keras.Sequential()
    # res_block.add(BottleNeck(filter_num, stride=stride))
    # res_block = []
    # res_block = flow.math.add(BottleNeck('make_bottle0', inputs, filter_num, stride=stride), res_block)
    x = BottleNeck('make_bottle0', inputs, filter_num, stride=stride)
    for _ in range(1, blocks):
        # res_block = flow.math.add(BottleNeck('make_bottle0', res_block, filter_num, stride=1), res_block)
        x = BottleNeck('make_bottle0', x, filter_num, stride=1)
    return x
