from flask import Blueprint

auth = Blueprint('auth', __name__, static_folder='static')

@auth.route('/upload', methods=['GET','POST'])
def upload():
    return auth.send_static_file('upload.html')