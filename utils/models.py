import segmentation_models_pytorch as smp
import torch

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class Segmentor:
    def __init__(self, args):
        # create segmentation model with pretrained encoder
        self.model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=1,
            activation="sigmoid",
        )
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet50", "imagenet")
        self.loss = smp.utils.losses.DiceLoss()
        self.metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
            smp.utils.metrics.Accuracy(threshold=0.5),
            smp.utils.metrics.Precision(),
            smp.utils.metrics.Recall(),
        ]
        if torch.cuda.is_available():
            self.device = args.device
        else:
            self.device = "cpu"
        # self.optimizer = torch.optim.Adam([
        #     dict(params=self.model.parameters(), lr=0.0001),
        # ])

        # self.train_epoch = smp.utils.train.TrainEpoch(
        #                     self.model,
        #                     loss=self.loss,
        #                     metrics=self.metrics,
        #                     optimizer=self.optimizer,
        #                     device=args.device,
        #                     verbose=True,
        #                 )

        # self.valid_epoch = smp.utils.train.ValidEpoch(
        #                     self.model,
        #                     loss=self.loss,
        #                     metrics=self.metrics,
        #                     device=args.device,
        #                     verbose=True,
        #                 )

    def test_model(self, path):

        if self.device == "cpu":
            self.test_model = smp.utils.train.ValidEpoch(
                torch.load(path, map_location="cpu"),
                loss=self.loss,
                metrics=self.metrics,
                # device=self.device,
                device="cpu",
                verbose=True,
            )
            self.model = torch.load(path, map_location="cpu")
        else:
            self.test_model = smp.utils.train.ValidEpoch(
                torch.load(path),
                loss=self.loss,
                metrics=self.metrics,
                device=self.device,
                verbose=True,
            )
            self.model = torch.load(path)
