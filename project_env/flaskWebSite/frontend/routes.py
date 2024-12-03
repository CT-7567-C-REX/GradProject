from flask import render_template, redirect, url_for, Blueprint
from flaskWebSite.models import INOUT, ClassificationImgs
from flaskWebSite.frontend.forms import UploadImgForm, UploadImg, FeedbackForm
from flaskWebSite.frontend.utils import save_picture, predict_single_image, classes, generate
from pathlib import Path

frontend = Blueprint('frontend', __name__)

BASE_DIR = Path(__file__).resolve().parent
CLASSIFICATION_IMAGES_DIR = BASE_DIR / "static" / "classificationimages"
GENERAL_IMAGES_DIR = BASE_DIR / "static" / "images"

@frontend.route("/", methods=['GET', 'POST'])
def home():

    form = UploadImgForm()

    if form.validate_on_submit():

        return redirect(url_for('home'))
    
    return render_template("index.html", form=form, title='Ana Sayfa')


@frontend.route("/classification", methods=['GET', 'POST'])
def classification():

    return render_template("class.html", title='Classification')



@frontend.route("/rlhfapi", methods=['GET', 'POST'])
def rlhfapi():

    return render_template("rlhfapi.html", title='rlhf')