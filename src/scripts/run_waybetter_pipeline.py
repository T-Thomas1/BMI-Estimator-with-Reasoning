from claid import CLAID  # type: ignore
from claid.module.module_factory import ModuleFactory  # type: ignore
from claid_modules.waybetter_data_loader import WayBetterDataLoader
from claid_modules.densenet import DenseNetInference

import os

current_path = os.path.dirname(os.path.abspath(__file__))

module_factory: ModuleFactory = ModuleFactory()
module_factory.register_module(WayBetterDataLoader)
module_factory.register_module(DenseNetInference)

claid = CLAID()
claid.start(
    "{}/../claid_configurations/waybetter_toy_example.json".format(current_path),
    "train_host",
    "train_user",
    "train_device",
    module_factory,
)