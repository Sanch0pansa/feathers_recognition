from BaseModel import BaseModel
from models import get_resnet50


class Resnet50Model(BaseModel):
    def __init__(self, num_classes: int, model_config: dict, *args, **kwargs):
        super().__init__(get_resnet50, num_classes, model_config)