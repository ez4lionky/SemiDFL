import GlobalParameters


class Config:
    def __init__(self, **entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                entries[k] = Config(**v)
        self.__dict__.update(entries)
        
    def __str__(self):
        return '\n'.join(f"{key}: {value}" for key, value in self.__dict__.items())
        # return json.dumps(self.__dict__)


# convert the argument of GlobalParameters.py into the dict that can be output to the log file.
def init_global_config():
    config = {k: v for k, v in GlobalParameters.__dict__.items() if '__' not in k}
    g_c = Config(**config)
    return g_c
