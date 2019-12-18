import json
import os

import tornado

from service.gan_service import GanService


class GanController(tornado.web.RequestHandler):

    def initialize(self):
        self.gan_service = GanService()

    def get(self):
        try:
            path = 'transfer.png'
            self.add_header('Content-Type', self.gan_service.get_content_type())
            self.write(self.gan_service.view_img(path))
        except:
            raise tornado.web.HTTPError(status_code=404)

    def post(self):
        target = self.get_argument('target')
        style = self.get_argument('style')
        result = json.dumps({"image_url": self.gan_service.run(target, style)})
        self.write(result)



