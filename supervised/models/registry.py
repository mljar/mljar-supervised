
class ModelsRegistry():

    registry = {}

    def __init__(self, task_name, model_name, model_code, model_class):
        model_information = {"model_name": model_name,
                                "model_code": model_code,
                                "model_class": model_class}
        if task_name in ModelsRegistry.registry:
            ModelsRegistry[task_name] += [model_information]
        else:
            ModelsRegistry[task_name] = [model_information]
