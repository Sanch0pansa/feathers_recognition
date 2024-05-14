import os
import torch
from models import Resnet50Model, Dense121Model, Dense201Model, Dense161Model, Dense169Model
from torchvision.transforms import transforms
import json
from PIL import Image


models = {
    "Resnet50": Resnet50Model,
    "Densenet121": Dense121Model,
    "Densenet161": Dense161Model,
    "Densenet169": Dense169Model,
    "Densenet201": Dense201Model,
}


class RecognitionService:
    def __init__(self):
        """
        Initializing service class
        """

        # Reading indices to class files for converting to class names
        with open("./data/idx-to-class_top100.json", "r") as f:
            self._idx_to_class_top100 = json.load(f)

        with open("./data/idx-to-class_all.json", "r") as f:
            self._idx_to_class_all = json.load(f)

        # Initializing top-100 model
        self._model_top_100 = models[os.environ.get("MODEL_TOP100_NAME", "Resnet50")](
            num_classes=len(self._idx_to_class_top100),
            model_config={
                "weights": os.environ.get(
                    "MODEL_TOP100_TRAINED_PATH",
                    "./models/weights/resnet50_weighted_trained.pt"
                ),
            }
        )
        # Switching to evaluation mode
        self._model_top_100.eval()

        # Initializing top-595 model
        self._model_top_595 = models[os.environ.get("MODEL_TOP595_NAME", "Densenet201")](
            num_classes=len(self._idx_to_class_all),
            model_config={
                "weights": os.environ.get(
                    "MODEL_TOP595_TRAINED_PATH",
                    "./models/weights/densenet201_trained.pt"
                ),
            }
        )
        # Switching to evaluation mode
        self._model_top_595.eval()

        # Initializing transforms
        self._transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((os.environ.get("IMG_WIDTH", 224), os.environ.get("IMG_HEIGHT", 224))),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _prepare_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)

        w, h = image.size
        if w > h:
            image = image.rotate(90, expand=True)

        input_tensor = self._transforms(image)
        input_batch = input_tensor.unsqueeze(0)

        return input_batch

    def _inference(self, batch: torch.Tensor, use_all_classes: bool = False) -> torch.Tensor:
        if use_all_classes:
            output = self._model_top_595(batch)
        else:
            output = self._model_top_100(batch)
        return torch.softmax(output, dim=1)[0]

    def predict(self, image_path: str, top_k: int = 5, use_all_classes: bool = False):
        """
        Make a prediction

        :param image_path: (str) path to image
        :param top_k: (int) how many species will be added to output range
        :param use_all_classes: (bool) use model_top_595 if true and else model_top_100
        :return: list of tuples (species name, probability)
        """
        with torch.no_grad():
            input_batch = self._prepare_image(image_path)
            probs = self._inference(input_batch, use_all_classes)

            prediction_top_k_prob, prediction_top_k_indices = (torch.topk(probs, top_k))

            if use_all_classes:
                return [
                    (self._idx_to_class_all[i], int(prob * 10000) / 100)
                    for (i, prob) in zip(prediction_top_k_indices, prediction_top_k_prob)
                ]

            return [
                (self._idx_to_class_top100[i], int(prob * 10000) / 100)
                for (i, prob) in zip(prediction_top_k_indices, prediction_top_k_prob)
            ]


if __name__ == "__main__":
    s = RecognitionService()
