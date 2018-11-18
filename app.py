from flask import (
    Flask,
    render_template,
)

app = Flask(__name__, template_folder='templates')

# run code from another python file
from iris import (data_set, main)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/user/<username>')
def index(username):
    return render_template('index.html', username=username)


@app.route('/iris')
def iris():
    return render_template('iris.html',
                           feature_name_colspan=len(data_set.feature_names),
                           data_set=data_set)


if __name__ == '__main__':
    app.run()
