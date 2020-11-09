# import tensorflow as tf
import oneflow as flow
from core.resnet_module import make_bottleneck_layer, make_basic_layer
from core.resnet_module import _conv2d_layer, _batch_norm

class HighResolutionModule(object):
    def __init__(self, num_branches, num_in_channels, num_channels, block, num_blocks, fusion_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        self.num_in_channels = num_in_channels
        self.fusion_method = fusion_method
        self.multi_scale_output = multi_scale_output
        self.num_channels = num_channels
        self.block = block
        self.num_blocks = num_blocks
        # self.branches = self.__make_branches(num_channels, block, num_blocks)
        # self.fusion_layer = self.__make_fusion_layers()

    def get_output_channels(self):
        return self.num_in_channels

    def __make_branches(self, inputs, num_channels, block, num_blocks):
        # branch_layers = []
        branch = inputs
        for i in range(self.num_branches):
            # branch_layers.append(self.__make_one_branch(block, num_blocks[i], num_channels[i]))
            branch = self.__make_one_branch(branch, block, num_blocks[i], num_channels[i])
        return branch

    @staticmethod
    def __make_one_branch(inputs, block, num_blocks, num_channels, stride=1):
        if block == "BASIC":
            return make_basic_layer(inputs, filter_num=num_channels, blocks=num_blocks, stride=stride)
        elif block == "BOTTLENECK":
            return make_bottleneck_layer(inputs, filter_num=num_channels, blocks=num_blocks, stride=stride)

    def __make_fusion_layers(self, x):
        if self.num_branches == 1:
            return x

        fusion_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fusion_layer = []
            for j in range(self.num_branches):
                if j > i:
                    with flow.scope.namespace('_fuse_layers' + str(i)+str(j)):
                        temp = _conv2d_layer(name="fuse1", input=x[j], filters=self.num_in_channels[i], kernel_size=1, strides=1, padding="SAME", use_bias=False)
                        temp = _batch_norm(inputs=temp, momentum=0.1, epsilon=1e-5)
                        temp = flow.layers.upsample_2d(x=temp, data_format="NHWC", size=(2**(j-i), 2**(j-i)))
                        fusion_layer.append(temp)
                    # fusion_layer.append(
                        # tf.keras.Sequential([
                        #     tf.keras.layers.Conv2D(filters=self.num_in_channels[i], kernel_size=(1, 1), strides=1, padding="same", use_bias=False),
                        #     tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5),
                        #     tf.keras.layers.UpSampling2D(size=2**(j-i))
                        # ])

                    # )
                elif j == i:
                    fusion_layer.append(x[j])
                else:
                    down_sample = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            with flow.scope.namespace('_fuse_layers' + str(i) + str(j)):
                                downsample_out_channels = self.num_in_channels[i]
                                temp = _conv2d_layer(name="fuse11", input=x[j], filters=downsample_out_channels,
                                                     kernel_size=3, strides=2, padding="SAME",use_bias=False)
                                temp = _batch_norm(inputs=temp, momentum=0.1, epsilon=1e-5)
                                down_sample.append(temp)
                            # down_sample.append(
                                #     tf.keras.Sequential([
                                #         tf.keras.layers.Conv2D(filters=downsample_out_channels,
                                #                                kernel_size=(3, 3),
                                #                                strides=2,
                                #                                padding="same",
                                #                                use_bias=False),
                                #         tf.keras.layers.BatchNormalization(momentum=0.1,
                                #                                            epsilon=1e-5)
                                #     ])
                                # )
                        else:
                            with flow.scope.namespace('_fuse_layers' + str(i) + str(j)):
                                downsample_out_channels = self.num_in_channels[j]
                                temp = _conv2d_layer(name="fuse11", input=x[j], filters=downsample_out_channels,
                                                     kernel_size=3, strides=2, padding="SAME", use_bias=False)
                                temp = _batch_norm(inputs=temp, momentum=0.1, epsilon=1e-5)
                                temp = flow.nn.relu(temp)
                                down_sample.append(temp)
                            # down_sample.append(
                            #     tf.keras.Sequential([
                            #         tf.keras.layers.Conv2D(filters=downsample_out_channels,
                            #                                kernel_size=(3, 3),
                            #                                strides=2,
                            #                                padding="same",
                            #                                use_bias=False),
                            #         tf.keras.layers.BatchNormalization(momentum=0.1,
                            #                                            epsilon=1e-5),
                            #         tf.keras.layers.ReLU()
                            #     ])
                            # )
                    fusion_layer.append(down_sample)
            fusion_layers.append(fusion_layer)
        return fusion_layers

    def call(self, inputs, training=None, **kwargs):
        if self.num_branches == 1:
            return [self.__make_branches(inputs[0], self.num_channels, self.block, self.num_blocks)[0]]

        for i in range(self.num_branches):
            # inputs[i] = self.branches[i](inputs[i], training=training)
            inputs[i] = self.__make_branches(inputs[i], self.num_channels, self.block, self.num_blocks)[i]
        x = inputs
        x_fusion = []
        result = self.__make_fusion_layers(x)
        # for i in range(len(self.fusion_layer)):
        for i in range(len(result)):
            y = x[0] if i == 0 else result[i][0]#(x[0], training=training)
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    # y = y + self.fusion_layer[i][j](x[j], training=training)
                    y = y + result[i][j]
            x_fusion.append(flow.nn.relu(y))
        return x_fusion


class StackLayers(object):
    def __init__(self, layers):
        super(StackLayers, self).__init__()
        self.layers = layers

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training)
        return x


class HRNet(object):
    def __init__(self, config):
        super(HRNet, self).__init__()
        self.config_params = config
        # self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="same", use_bias=False)
        # self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)
        # self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="same", use_bias=False)
        # self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)
        # self.layer1 = make_bottleneck_layer(filter_num=64, blocks=4)
        # self.transition1 = self.__make_transition_layer(previous_branches_num=1,
        #                                                 previous_channels=[256],
        #                                                 current_branches_num=self.config_params.get_stage("s2")[1],
        #                                                 current_channels=self.config_params.get_stage("s2")[0])
        # self.stage2 = self.__make_stages("s2", self.config_params.get_stage("s2")[0])
        # self.transition2 = self.__make_transition_layer(previous_branches_num=self.config_params.get_stage("s2")[1],
        #                                                 previous_channels=self.config_params.get_stage("s2")[0],
        #                                                 current_branches_num=self.config_params.get_stage("s3")[1],
        #                                                 current_channels=self.config_params.get_stage("s3")[0])
        # self.stage3 = self.__make_stages("s3", self.config_params.get_stage("s3")[0])
        # self.transition3 = self.__make_transition_layer(previous_branches_num=self.config_params.get_stage("s3")[1],
        #                                                 previous_channels=self.config_params.get_stage("s3")[0],
        #                                                 current_branches_num=self.config_params.get_stage("s4")[1],
        #                                                 current_channels=self.config_params.get_stage("s4")[0])
        # self.stage4 = self.__make_stages("s4", self.config_params.get_stage("s4")[0], False)
        # self.conv3 = tf.keras.layers.Conv2D(filters=self.config_params.num_of_joints,
        #                                     kernel_size=self.config_params.conv3_kernel,
        #                                     strides=1,
        #                                     padding="same")

    # def __choose_config(self, config_name):
    #     return get_config_params(config_name)

    def __make_stages(self, inputs, stage_name, in_channels, multi_scale_output=True):
        stage_info = self.config_params.get_stage(stage_name)
        channels, num_branches, num_modules, block, num_blocks, fusion_method = stage_info
        fusion = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            module_list = HighResolutionModule(num_branches=num_branches,
                                               num_in_channels=in_channels,
                                               num_channels=channels,
                                               block=block,
                                               num_blocks=num_blocks,
                                               fusion_method=fusion_method,
                                               multi_scale_output=reset_multi_scale_output)
            fusion = module_list.call(inputs)
        return fusion

    @staticmethod
    def __make_transition_layer(x, previous_branches_num, previous_channels, current_branches_num, current_channels):
        transition_layers = []
        for i in range(current_branches_num):
            if i < previous_branches_num:
                if current_channels[i] != previous_channels[i]:
                    temp = _conv2d_layer(name="trans1"+str(i), input=x, filters=current_channels[i], kernel_size=3,
                                         strides=1, padding="SAME", use_bias=False)
                    temp = _batch_norm(inputs=temp, momentum=0.1, epsilon=1e-5)
                    transition_layers.append(temp)
                    # transition_layers.append(
                    #     tf.keras.Sequential([
                    #         tf.keras.layers.Conv2D(filters=current_channels[i], kernel_size=(3, 3), strides=1, padding="same", use_bias=False),
                    #         tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5),
                    #         tf.keras.layers.ReLU()
                    #     ])
                    # )
                else:
                    transition_layers.append(x)
            else:
                down_sampling_layers = []
                for j in range(i + 1 - previous_branches_num):
                    in_channels = previous_channels[-1],
                    out_channels = current_channels[i] if j == i - previous_branches_num else in_channels
                    with flow.scope.namespace('transition_layers_'+str(j)):
                        temp = _conv2d_layer(name="fuse11", input=x, filters=out_channels,
                                             kernel_size=3, strides=2, padding="SAME", use_bias=False)
                        temp = _batch_norm(inputs=temp, momentum=0.1, epsilon=1e-5)
                        temp = flow.nn.relu(temp)
                        down_sampling_layers.append(temp)
                        # down_sampling_layers.append(
                        #     tf.keras.Sequential([
                        #         tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(3, 3), strides=2,
                        #                                padding="same", use_bias=False),
                        #         tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5),
                        #         tf.keras.layers.ReLU()
                        #     ])
                        # )
                transition_layers.append(down_sampling_layers)
        return transition_layers

    def call(self, inputs, training=None):#mask=None
        x = _conv2d_layer(name="conv1", input=inputs, filters=64,
                          kernel_size=3, strides=2, padding="SAME", use_bias=False)
        # x = self.conv1(inputs)
        x = _batch_norm(inputs=x, momentum=0.1, epsilon=1e-5)
        # x = self.bn1(x, training=training)
        x = flow.nn.relu(x)
        # x = tf.nn.relu(x)
        x = _conv2d_layer(name="conv2", input=x, filters=64,
                          kernel_size=3, strides=2, padding="SAME", use_bias=False)
        # x = self.conv2(x)
        x = _batch_norm(inputs=x, momentum=0.1, epsilon=1e-5)
        # x = self.bn2(x, training=training)
        x = flow.nn.relu(x)
        # x = tf.nn.relu(x)
        x = make_bottleneck_layer(x, training=training, filter_num=64, blocks=4)
        # x = self.layer1(x, training=training)

        feature_list = []
        for i in range(self.config_params.get_stage("s2")[1]):
            result = self.__make_transition_layer(x=x,
                                                  previous_branches_num=1,
                                                  previous_channels=[256],
                                                  current_branches_num=self.config_params.get_stage("s2")[1],
                                                  current_channels=self.config_params.get_stage("s2")[0])
            if result[i] is not None:
                feature_list.append(result[i])
            # if self.transition1[i] is not None:
            #     feature_list.append(self.transition1[i](x, training=training))
            else:
                feature_list.append(x)
        y_list = self.__make_stages(feature_list, "s2", self.config_params.get_stage("s2")[0])
        # y_list = self.stage2(feature_list, training=training)

        feature_list = []
        for i in range(self.config_params.get_stage("s3")[1]):
            result = self.__make_transition_layer(x=y_list[-1],
                                                  previous_branches_num=self.config_params.get_stage("s2")[1],
                                                  previous_channels=self.config_params.get_stage("s2")[0],
                                                  current_branches_num=self.config_params.get_stage("s3")[1],
                                                  current_channels=self.config_params.get_stage("s3")[0])
            if result[i] is not None:
                feature_list.append(result[i])
            # if self.transition2[i] is not None:
            #     feature_list.append(self.transition2[i](y_list[-1], training=training))
            else:
                feature_list.append(y_list[i])
        y_list = self.__make_stages(feature_list, "s3", self.config_params.get_stage("s3")[0])
        # y_list = self.stage3(feature_list, training=training)

        feature_list = []
        for i in range(self.config_params.get_stage("s4")[1]):
            result = self.__make_transition_layer(x=y_list[-1],
                                                  previous_branches_num=self.config_params.get_stage("s3")[1],
                                                  previous_channels=self.config_params.get_stage("s3")[0],
                                                  current_branches_num=self.config_params.get_stage("s4")[1],
                                                  current_channels=self.config_params.get_stage("s4")[0])
            if result[i] is not None:
                feature_list.append(result[i])
        # for i in range(self.config_params.get_stage("s4")[1]):
        #     if self.transition3[i] is not None:
        #         feature_list.append(self.transition3[i](y_list[-1], training=training))
            else:
                feature_list.append(y_list[i])
        y_list = self.__make_stages(feature_list, "s4", self.config_params.get_stage("s4")[0], False)
        # y_list = self.stage4(feature_list, training=training)
        outputs = _conv2d_layer(name="conv3",
                                input=y_list[0],
                                filters=self.config_params.num_of_joints,
                                kernel_size=self.config_params.conv3_kernel,
                                strides=1,
                                padding="SAME")
        # outputs = self.conv3(y_list[0])

        return outputs
