from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Home."
    
@app.route('/tuna')
def tuna():
    return "<h2>tuna.</h2>" 

@app.route('/profile/<username>')
def profile(username):
    return "<h2>Hey %s</h2>" % username   
    
@app.route('/post/<int:post_id>')
def shiw_post(post_id):
    return "<h2>Post ID is %s</h2>" % post_id     
    
    
if __name__ == "__main__":
    app.run()