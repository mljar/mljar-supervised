import json
from supervised.utils.jsonencoder import MLJSONEncoder


def json_loads(data, *args, **kwargs):
    return json.loads(data, *args, **kwargs)


def json_dumps(data, *args, **kwargs):
    return json.dumps(data, cls=MLJSONEncoder, *args, **kwargs)
