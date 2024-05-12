from models.BaseModel import BaseModel
from models.models import get_resnet50, get_resnet18, get_resnet34, get_resnet101, get_resnet152


class Resnet50Model(BaseModel):
    def __init__(self, num_classes: int, model_config: dict, *args, **kwargs):
        super().__init__(get_resnet50, num_classes, model_config)


class Resnet18Model(BaseModel):
    def __init__(self, num_classes: int, model_config: dict, *args, **kwargs):
        super().__init__(get_resnet18, num_classes, model_config)


class Resnet34Model(BaseModel):
    def __init__(self, num_classes: int, model_config: dict, *args, **kwargs):
        super().__init__(get_resnet34, num_classes, model_config)


class Resnet101Model(BaseModel):
    def __init__(self, num_classes: int, model_config: dict, *args, **kwargs):
        super().__init__(get_resnet101, num_classes, model_config)


class Resnet152Model(BaseModel):
    def __init__(self, num_classes: int, model_config: dict, *args, **kwargs):
        super().__init__(get_resnet152, num_classes, model_config)
