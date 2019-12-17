import tornado

from service.gan_service import GanService


class GanController(tornado.web.RequestHandler):

    def initialize(self):
        self.gan_service = GanService()


    def get(self):
        self.write("Hello, world")

    def post(self):
        self.gan_service.run()

