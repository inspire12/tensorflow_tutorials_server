import json
import os

import tornado

from service.gan_deepdream_service import GanDeepdreamService


class GanDeepdreamController(tornado.web.RequestHandler):

    def initialize(self):
        self.gan_service = GanDeepdreamService()

    def get(self):
        try:
            # path = 'transfer.png'
            self.add_header('Content-Type', self.gan_service.get_content_type())
            self.write(self.gan_service.transferred_path())
        except:
            raise tornado.web.HTTPError(status_code=404)

    def post(self):
        result = json.dumps({"image_url": ""})
        self.gan_service.run_deepdream()
        self.write(result)



