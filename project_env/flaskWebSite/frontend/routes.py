from flask import render_template, Blueprint

frontend = Blueprint('frontend', __name__)


@frontend.route("/", methods=['GET', 'POST'])
def home():
    
    return render_template("index.html", title='Ana Sayfa')


@frontend.route("/rlhfapi", methods=['GET', 'POST'])
def rlhfapi():

    return render_template("rlhfapi.html", title='rlhf')