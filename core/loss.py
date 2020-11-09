# import tensorflow as tf
import oneflow as flow
import numpy as np

class JointsMSELoss(object):
    def __init__(self):
        super(JointsMSELoss, self).__init__()

    def __call__(self, y_pred, target, target_weight):
        batch_size = y_pred.shape[0]
        num_of_joints = y_pred.shape[-1]
        pred = flow.reshape(x=y_pred, shape=(batch_size, -1, num_of_joints))
        '''#注意一下,,这里的格式可能会出现问题'''
        # heatmap_pred_list = tf.split(value=pred, num_or_size_splits=num_of_joints, axis=-1)
        heatmap_pred_list = np.split(ary=pred, indices_or_sections=num_of_joints, axis=-1)
        gt = flow.reshape(x=target, shape=(batch_size, -1, num_of_joints))
        # heatmap_gt_list = tf.split(value=gt, num_or_size_splits=num_of_joints, axis=-1)
        heatmap_gt_list = np.split(ary=gt, indices_or_sections=num_of_joints, axis=-1)
        loss = 0.0
        for i in range(num_of_joints):
            heatmap_pred = flow.squeeze(heatmap_pred_list[i])
            heatmap_gt = flow.squeeze(heatmap_gt_list[i])
            temp = flow.math.square(heatmap_pred * target_weight[:, i] - heatmap_gt * target_weight[:, i])
            loss += 0.5 * flow.math.reduce_mean(temp, axis=1, keepdims=True)
            # loss += 0.5 * mse_(y_true=heatmap_pred * target_weight[:, i],
            #                        y_pred=heatmap_gt * target_weight[:, i])
        return loss / num_of_joints

    @staticmethod
    def mse_(x, y):
        temp = flow.math.square(x - y)
        mse = flow.math.reduce_mean(temp, axis=1, keepdims=True)
        return mse
