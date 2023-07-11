from colors import bcolors

class Logger():

    def __init__(self, model_name, counter = 1):

        self.model_name = model_name
        self.f = open(self.model_name,"a")

    def write(self, x, color = bcolors.ENDC, end = "\n"):

        assert type(x) == str, "Invalid type for writing"

        print(color + x + bcolors.ENDC, end=end)

        self.f.write(x + end)
        self.f.flush()
    
    def draw_line(self, sz = 30):

        s = "-"*sz
        print(s)
        self.f.write(s + "\n")
        self.f.flush()


    def close(self):
        self.f.close()
