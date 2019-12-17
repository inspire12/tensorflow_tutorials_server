import tornado.ioloop
import tornado.web

from controller.gan_controller import GanController
from controller.test_controller import MainHandler
from controller.test_controller import TestController

class ImageViewController(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Length", 42)
        self.set_header("Content-Type", "image/gif")
        self.set_header("Pragma", "no-cache")
        self.set_header("Cache-Control",
                        "no-store, "
                        "no-cache=Set-Cookie, "
                        "proxy-revalidate, "
                        "max-age=0, "
                        "post-check=0, pre-check=0"
                        )
        self.set_header("Expires", "Wed, 2 Dec 1837 21:00:12 GMT")
        self.write(self.pixel_binary)


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/test", TestController),
        (r"/gan", GanController),
        (r"/img", ImageViewController)
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
