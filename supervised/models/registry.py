
# tasks that can be handled by the package
BINARY_CLASSIFICATION = 'binary_classification'
MULTICLASS_CLASSIFICATION = 'multiclass_classification'
REGRESSION = 'regression'


class ModelsRegistry():

    registry = {}

    def __init__(self, task_name, model_name, model_code, model_class):
        model_information = {"model_name": model_name,
                                "model_code": model_code,
                                "model_class": model_class}
        print(model_information)
        if task_name in ModelsRegistry.registry:
            ModelsRegistry.registry[task_name] += [model_information]
        else:
            ModelsRegistry.registry[task_name] = [model_information]
