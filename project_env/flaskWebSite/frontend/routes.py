from flask import render_template, redirect, url_for, Blueprint
from flaskWebSite import app, db
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
    form = UploadImg()
    feedback_form = FeedbackForm()

    if form.validate_on_submit():
       
        picture_file = save_picture(form.img.data, str(CLASSIFICATION_IMAGES_DIR))
        image_path = CLASSIFICATION_IMAGES_DIR / picture_file

       
        prediction = predict_single_image(str(image_path))
        db_entry = ClassificationImgs(InputtedPic=picture_file, prediction=prediction)
        db.session.add(db_entry)
        db.session.commit()
        
        return render_template( "class.html", form=form, feedback_form=feedback_form, title='Classification', pred=prediction, classes=classes, img_path=picture_file)

    if feedback_form.validate_on_submit():
       
        latest_img = ClassificationImgs.query.order_by(ClassificationImgs.id.desc()).first()
        if latest_img:
            latest_img.feedback = feedback_form.feedback.data
            db.session.commit()
            # this render line needs to modifed, most part not even works. we gonna look together guys :) yeeeee
        return render_template( "class.html", form=form, feedback_form=feedback_form, title='Classification', pred=latest_img.prediction if latest_img else None, classes=classes,img_path=latest_img.InputtedPic if latest_img else None)

    return render_template("class.html", form=form, feedback_form=feedback_form, title='Classification', classes=classes)


@frontend.route("/drawing", methods=['GET', 'POST'])
def drawing():

    return render_template("drawing.html", title='Drawing')


@frontend.route("/apiproccess", methods=['GET', 'POST'])
def apiproccess():

    return render_template("apiproccess.html", title='Drawing')


@frontend.route("/rlhf", methods=['GET', 'POST'])
def rlhf():
    form = UploadImg()
    if form.validate_on_submit():
        # Save uploaded image
        picture_file = save_picture(form.img.data, str(GENERAL_IMAGES_DIR))
        image_path = GENERAL_IMAGES_DIR / picture_file
        image_url = f"/static/images/{picture_file}"

        # Generate output
        output_image = generate(str(image_path))

        return render_template(
            "rlhf.html",
            form=form,
            title='Reinforcement Learning with Human Feedback',
            outimgae=output_image
        )

    return render_template("rlhf.html", form=form, title='Reinforcement Learning with Human Feedback')
