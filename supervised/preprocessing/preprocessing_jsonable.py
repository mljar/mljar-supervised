#from utils.serve_json import ServeJson

class PreprocessingJsonable(object):

    def to_json(self):
        pass

    def from_json(self, data_json):
        pass

    def save(self, save_fullpath):
        pass
        #ServeJson.save(save_fullpath, self.to_json())

    def load(self, load_fullpath):
        pass
        #self.from_json(ServeJson.load(load_fullpath))
