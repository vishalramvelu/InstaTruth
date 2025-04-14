from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api')
def api():
    return render_template('api.html')

if __name__ == '__main__':
    app.run(debug=True)