from BaseModel import BaseModel
from models import get_dense121, get_dense161, get_dense169, get_dense201


class Dense121Model(BaseModel):
    def __init__(self, num_classes: int, model_config: dict, *args, **kwargs):
        super().__init__(get_dense121, num_classes, model_config)


class Dense161Model(BaseModel):
    def __init__(self, num_classes: int, model_config: dict, *args, **kwargs):
        super().__init__(get_dense161, num_classes, model_config)


class Dense169Model(BaseModel):
    def __init__(self, num_classes: int, model_config: dict, *args, **kwargs):
        super().__init__(get_dense169, num_classes, model_config)


class Dense201Model(BaseModel):
    def __init__(self, num_classes: int, model_config: dict, *args, **kwargs):
        super().__init__(get_dense201, num_classes, model_config)
