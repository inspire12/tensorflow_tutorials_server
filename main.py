import os

import tornado.ioloop
import tornado.web

from controller.gan_controller import GanController
from controller.test_controller import MainHandler
from controller.test_controller import TestController
from controller.word_controller import WordController


class imageViewController(tornado.web.StaticFileHandler):
    def parse_url_path(self, url_path: str):
        return url_path




def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/test", TestController),
        (r"/gan", GanController),
        (r"/gan/(.*)",imageViewController , {'path': os.path.join(os.path.abspath("./"), "resource", "tmp_img")}),
        (r"/word/embeddings", WordController)

    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()

#
# def main():
#     print ("start project ")
#     runner = GenService()
#     runner.run()
#     print ("end project ")
#
#
# if __name__ == '__main__':
#     main()
