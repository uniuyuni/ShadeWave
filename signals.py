
class Signal():

    def __init__(self):
        self.funcs = []

    def connect(self, func):
        self.funcs.append(func)

    def disconnect(self, func):
        self.funcs.remove(func)

    def emit(self, *args):
        for func in self.funcs:
            func(*args)
    
    def delay_emit(self, *args):
        from kivy.clock import Clock

        Clock.schedule_once(lambda dt: self.emit(*args), 0)


blit_image = Signal()
