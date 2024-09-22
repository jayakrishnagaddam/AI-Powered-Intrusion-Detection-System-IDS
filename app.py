from flask import Flask, render_template, session
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')



if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True)
