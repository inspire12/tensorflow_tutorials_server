import tornado

from service.word_service import WordService


class WordController(tornado.web.RequestHandler):

    def initialize(self):
        self.word_service = WordService()


    def post(self):
        plot_image = self.word_service.run()
        self.set_header('Content-type', 'image/png')
        self.set_header('Content-length', len(plot_image))
        self.write(plot_image)