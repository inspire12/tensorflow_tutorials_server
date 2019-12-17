import tornado.ioloop
import tornado.web

from controller.test_controller import MainHandler
from controller.test_controller import TestController


from service.test_service import TestRun
# from service.gpu_option_test_service import TestRun
from service.gen_service import GenService



def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/test", TestController),
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
