import json


def json_loads(data, *args, **kwargs):
    return json.loads(data, *args, **kwargs)


def json_dumps(data, *args, **kwargs):
    return json.dumps(data, *args, **kwargs)
