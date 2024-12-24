from flask import render_template, redirect, url_for, Blueprint
from flaskWebSite.frontend.forms import UploadImgForm

frontend = Blueprint('frontend', __name__)


@frontend.route("/", methods=['GET', 'POST'])
def home():

    form = UploadImgForm()

    if form.validate_on_submit():

        return redirect(url_for('home'))
    
    return render_template("index.html", form=form, title='Ana Sayfa')


@frontend.route("/rlhfapi", methods=['GET', 'POST'])
def rlhfapi():

    return render_template("rlhfapi.html", title='rlhf')