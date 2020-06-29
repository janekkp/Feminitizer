from flask import Flask, request, render_template
from predict import feminatize
app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('main_page.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = feminatize(text)
    return processed_text


if __name__ == '__main__':
    app.run()
