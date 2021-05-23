import tornado.web


class HealthHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header('Content-type', 'text/html')
        self.set_status(200)
        self.set_header('Server', '')
        self.write('OK')
        self.flush()
        self.finish()
        print('health ping')
