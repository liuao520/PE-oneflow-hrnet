# import tensorflow as tf
import oneflow as flow

from core.loss import JointsMSELoss
from core.make_dataset import CocoDataset
from core.make_ground_truth import GroundTruth
from core.metric import PCK
from test import test_during_training
from utils.work_flow import get_model, print_model_summary
from configuration.base_config import Config
from utils.tools import get_config_params

if __name__ == '__main__':
    # # GPU settings
    # gpus = tf.config.list_physical_devices("GPU")
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)

    cfg = get_config_params(Config.TRAINING_CONFIG_NAME)
    hrnet = get_model(cfg)
    print_model_summary(hrnet)

    # Dataset
    coco = CocoDataset(config_params=cfg, dataset_type="train")
    dataset, dataset_length = coco.generate_dataset()

    # loss and optimizer
    loss = JointsMSELoss()
    # optimizer = flow.optimizer.Adam(lr_scheduler=1e-3)

    # metircs
    # loss_metric = tf.metrics.Mean()
    pck = PCK()
    # accuracy_metric = tf.metrics.Mean()

    # def train_step(batch_data):
    #     gt = GroundTruth(cfg, batch_data)
    #     images, target, target_weight = gt.get_ground_truth()
    #     with tf.GradientTape() as tape:
    #         y_pred = hrnet(images, training=True)
    #         loss_value = loss(y_pred, target, target_weight)
    #     gradients = tape.gradient(loss_value, hrnet.trainable_variables)
    #     optimizer.apply_gradients(grads_and_vars=zip(gradients, hrnet.trainable_variables))
    #     loss_metric.update_state(values=loss_value)
    #     _, avg_accuracy, _, _ = pck(network_output=y_pred, target=target)
    #     accuracy_metric.update_state(values=avg_accuracy)

    check_point = flow.train.CheckPoint()
    start_epoch = cfg.LOAD_WEIGHTS_FROM_EPOCH

    if cfg.LOAD_WEIGHTS_BEFORE_TRAINING:
        # hrnet.load_weights(filepath=cfg.save_weights_dir + "epoch-{}".format(start_epoch))
        check_point.load(cfg.save_weights_dir + "epoch-{}".format(start_epoch))
        print("Successfully load weights!")
    else:
        start_epoch = -1

    for epoch in range(start_epoch + 1, cfg.EPOCHS):
        for step, batch_data in enumerate(dataset):
            # train_step(batch_data)
            gt = GroundTruth(cfg, batch_data)
            images, target, target_weight = gt.get_ground_truth()
            # with tf.GradientTape() as tape:
            y_pred = hrnet.call(images, training=True)
            loss_metric = loss.call(y_pred, target, target_weight)
            loss_metric = flow.math.reduce_mean(loss_metric)
            flow.losses.add_loss(loss_metric)
            # flow.optimizer.Adam(lr_scheduler=1e-3).minimize(loss_metric)

            _, accuracy_metric, _, _ = pck(network_output=y_pred, target=target)
            accuracy_metric = flow.math.reduce_mean(accuracy_metric)
            flow.losses.add_loss(accuracy_metric)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.10f}, accuracy: {:.5f}".format(epoch,
                                                                                      cfg.EPOCHS,
                                                                                      step,
                                                                                      flow.math.ceil(dataset_length / cfg.BATCH_SIZE),
                                                                                      loss_metric,
                                                                                      accuracy_metric))
            flow.optimizer.Adam(lr_scheduler=1e-3).minimize(loss_metric)

        if epoch % cfg.SAVE_FREQUENCY == 0:
            # hrnet.save_weights(filepath=cfg.save_weights_dir + "epoch-{}".format(epoch), save_format="tf")
            check_point.save(cfg.save_weights_dir + "epoch-{}".format(epoch))
        if cfg.TEST_DURING_TRAINING:
            test_during_training(cfg=cfg, epoch=epoch, model=hrnet)
        # loss_metric.reset_states()
        # accuracy_metric.reset_states()

    check_point.save(cfg.save_weights_dir+"saved_model")
    # hrnet.save_weights(filepath=cfg.save_weights_dir+"saved_model", save_format="tf")
