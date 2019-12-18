import json
import os

import tornado

from service.gan_style_transfer_service import GanStyleTransferService


class GanStyleTransferController(tornado.web.RequestHandler):

    def initialize(self):
        self.gan_service = GanStyleTransferService()

    def get(self):
        try:
            # path = 'transfer.png'
            self.add_header('Content-Type', self.gan_service.get_content_type())
            self.write(self.gan_service.transferred_path())
        except:
            raise tornado.web.HTTPError(status_code=404)

    def post(self):
        target = self.get_argument('target')
        style = self.get_argument('style')
        result = json.dumps({"image_url": self.gan_service.run(target, style)})
        self.write(result)



