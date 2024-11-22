from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField, BooleanField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileAllowed

class UploadImgForm(FlaskForm):
    img = FileField('Upload img (jpg,png,jpeg)', validators=[DataRequired(), FileAllowed(['jpg', 'png','jpeg'])])
    SelectionOne = SelectField('Type', choices=[('Select'), ('ONE'), ('TWO'), ('THREE'), ('FOUR'), ('FIVE')] ,validators=[DataRequired()] )
    SelectionTwo = SelectField('Type', choices=[('Select'), ('ONE'), ('TWO'), ('THREE')] ,validators=[DataRequired()] )
    submit = SubmitField('Upload')

class UploadImg(FlaskForm):
    img = FileField('Upload img (jpg,png,jpeg)', validators=[DataRequired(), FileAllowed(['jpg', 'png','jpeg'])])
    submit = SubmitField('Upload')

class FeedbackForm(FlaskForm):
    # Create a SelectField for user feedback (e.g., choice between correct/incorrect, or rating)
    feedback = SelectField('Was the prediction correct?', choices=[
        ('', 'Select an option'),  # Empty option for better UI
        ('correct', 'Yes, Correct'),
        ('incorrect', 'No, Incorrect')
    ], validators=[DataRequired()])
    
    submit = SubmitField('Submit Feedback')
    
class SelectStuffForm(FlaskForm):

    choice_one = BooleanField()
    choice_two = BooleanField()

    submit = SubmitField('Select')
