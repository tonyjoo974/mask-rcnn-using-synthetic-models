from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import masks_to_boxes
import evaluation
import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision


def create_trainer():

    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor="val_iou",
        filename="{epoch:02d}-{val_iou:.6f}-{val_loss:.6f}-{step}",
        save_top_k=1,
        save_last=True,
        mode="max")
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last-{epoch:02d}-{val_iou:.6f}-{val_loss:.6f}-{step}"

    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_iou",
        mode="max",
        patience=6,
        verbose=True)

    return pl.Trainer(accelerator="gpu",
                      gpus=1,
                      callbacks=[checkpoint_callback, early_stopping_callback])


class SMPModel(pl.LightningModule):
    def __init__(self, smp_model: nn.Module):
        super().__init__()
        self.model = smp_model

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        predict = self(x)
        loss = F.binary_cross_entropy_with_logits(predict, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predict = self(x)

        loss = F.binary_cross_entropy_with_logits(predict, y).item()
        iou = evaluation.iou(predict > 0, y > 0.5).mean().item()

        return [loss, iou]

    def validation_epoch_end(self, validation_outputs) -> None:

        vl, vi = np.mean(validation_outputs, axis=0)

        self.log("val_loss", vl, prog_bar=True)
        self.log("val_iou", vi, prog_bar=True)

        return

    def predict_step(self, batch, batch_idx):
        # prediction will return ([B,1,H,W] tensor of floats in[0,1])
        x, _ = batch
        predict = self(x).detach()
        return torch.sigmoid(predict)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.5*1e-3)


def pytorch_maskrcnn_resnet50_fpn(num_classes=2, disable_resnet_gradients=True):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # transfer learning, disable gradients
    if disable_resnet_gradients:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


class MaskRCNNModel(pl.LightningModule):
    def __init__(self, mrcnn: nn.Module):
        super().__init__()
        self.mrcnn = mrcnn

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.mrcnn([t for t in x])

    def _put_gpu(self, x, y):

        x_t = [t for t in x]
        y_t = [{k: v.cuda() for k, v in s.items()} for s in y]

        return x_t, y_t

    def training_step(self, batch, batch_idx):

        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x, y_t = self._put_gpu(x, self._y_transform(y))

        losses = self.mrcnn(x, y_t)
        loss = sum(l for l in losses.values())
        # Logging to TensorBoard by default
        self.log("train_loss", loss)

        return loss

    def predict_step(self, batch, batch_idx):
        # this calls forward
        x, _ = batch
        prediction = self(x)
        return self._y_inv_transform(prediction)

    def validation_step(self, batch, batch_idx):
        _, y = batch
        prediction = self.predict_step(batch, batch_idx)
        iou = evaluation.iou(prediction > 0.5, y > 0.5).mean().item()

        return iou

    def validation_epoch_end(self, validation_outputs) -> None:

        vi = np.mean(validation_outputs)

        self.log("val_iou", vi, prog_bar=True)

        return

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.5*1e-3)

    def _y_transform_single(self, y):
        y_byte = y.byte()
        # masks_to_boxes throws if y is all black
        # so need a seperate case
        if not torch.any(y_byte != 0):
            d = {
                "boxes": torch.zeros((0, 4), dtype=torch.float),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, *y_byte.shape[1:]), dtype=torch.uint8),
            }
        else:
            d = {
                "boxes": masks_to_boxes(y_byte),
                "labels": torch.tensor([1], dtype=torch.int64),
                "masks": y_byte,
            }
        return d

    def _y_transform(self, y_batch):

        y_t = [
            self._y_transform_single(y) for i, y in enumerate(y_batch)
        ]

        return y_t

    def _y_inv_transform(self, y_t_batch):
        # grab the first mask
        # if no mask available, return mask of 0s

        primary_masks = []

        for predict in y_t_batch:
            masks = predict['masks']
            mask = masks[0] if len(masks) > 0 \
                else torch.zeros(masks.shape[1:]).type_as(masks)
            primary_masks.append(mask[None, :])

        return torch.cat(primary_masks)
