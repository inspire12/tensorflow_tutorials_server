import tornado


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


class TestController(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

