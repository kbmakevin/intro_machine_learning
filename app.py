from flask import Flask
import sklearn.datasets

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

# run code from another python file
import iris

if __name__ == '__main__':
    app.run()
