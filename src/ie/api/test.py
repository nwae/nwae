
#
# Testing gunicorn
#

def app(environ, start_response):
	# """Simplest possible application object"""
	data = b'Hello, World!\n'
	status = '200 OK'
	response_headers = [
		('Content-type', 'text/plain'),
		('Content-Length', str(len(data)))
	]
	start_response(status, response_headers)
	return iter([data])

if __name__ == '__main__':
	print("Privet Mir!")
	#app(environ=None, start_response=None)
