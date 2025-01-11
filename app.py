from flask import Flask, render_template, redirect, url_for, session, flash
import json
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from authlib.integrations.flask_client import OAuth
from flask import Flask, render_template, request, jsonify, session
from indobert import SentimentAnalyzer
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import pickle
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain_community.vectorstores import FAISS  
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from werkzeug.security import generate_password_hash
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, request, jsonify
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
from flask_jwt_extended import create_access_token, JWTManager

app = Flask(__name__)

model_indobert = 'model'
analyzer_indobert = SentimentAnalyzer(model_indobert)

appConf = {
    "OAUTH2_META_URL": "https://accounts.google.com/.well-known/openid-configuration",
    "FLASK_SECRET": "467901b0-a75b-47fe-8971-80fd119c28c1",
    "FLASK_PORT": 5000
}

app.secret_key = appConf.get("FLASK_SECRET")

oauth = OAuth(app)

oauth.register("myApp",
               client_id=appConf.get("OAUTH2_CLIENT_ID"),
               client_secret=appConf.get("OAUTH2_CLIENT_SECRET"),
               server_metadata_url=appConf.get("OAUTH2_META_URL"),
               client_kwargs={
                   "scope": "openid profile email https://www.googleapis.com/auth/user.birthday.read https://www.googleapis.com/auth/user.gender.read",
               })

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/capstone'  
app.config['SECRET_KEY'] = 'thisisasecretkey'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

jwt = JWTManager(app)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=True, unique=True)
    email = db.Column(db.String(120), nullable=False, unique=True)  
    password = db.Column(db.String(80), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), nullable=False)
    feedback_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class Artikel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.String(255), nullable=False)
    judul = db.Column(db.String(255), nullable=False)
    deskripsi = db.Column(db.Text, nullable=False)

class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    
    email = StringField(validators=[InputRequired(), Length(min=4, max=120)], render_kw={"placeholder": "Email"})
    
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    
    submit = SubmitField("Register")
    
    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        
        if existing_user_username:
            raise ValidationError("That username already exists. Please choose a different one.")
    
    def validate_email(self, email):
        existing_user_email = User.query.filter_by(email=email.data).first()
        
        if existing_user_email:
            raise ValidationError("That email is already registered. Please choose a different one.")

class LoginForm(FlaskForm):
    email = StringField(validators=[InputRequired(), Length(min=4, max=120)], render_kw={"placeholder": "Email"})
    
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    
    submit = SubmitField("Login")

class ModelConfig:
    IMG_UPLOAD_FOLDER = 'static/images'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    LABEL_NAMES = ['cool', 'warm', 'neutral']

app.config['UPLOAD_FOLDER'] = ModelConfig.IMG_UPLOAD_FOLDER

class ImageDetectionService:
    def __init__(self):
        self.model = None
        self.label_names = ModelConfig.LABEL_NAMES
        self.initialize_model()

    def initialize_model(self):
        try:
            custom_objects = {}
            self.model = tf.keras.models.load_model(
                'models/model2.h5',
                custom_objects=custom_objects,
                compile=False  
            )

            if self.model is None:
                try:
                    self.model = tf.keras.models.load_model(
                        'models/model2.h5',
                        custom_objects=custom_objects,
                        options=tf.saved_model.LoadOptions(
                            experimental_io_device='/job:localhost'
                        )
                    )
                except:
                    self.model = self.build_model()
                    self.model.load_weights('models/model2.h5')

            print("Model loaded successfully")
            self.model.summary()
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def build_model(self):
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(200, 200, 3)),  
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax')  
            ])
            return model
        except Exception as e:
            print(f"Error building model: {str(e)}")
            raise

    def predict_image(self, image_path):
        try:
            input_shape = self.model.input_shape[1:3]  
            
            img = tf.keras.utils.load_img(
                image_path,
                target_size=input_shape
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            predicted_label = self.label_names[predicted_class]

            probabilities = [float(p) for p in predictions[0]]  
            
            return predicted_label, probabilities
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None, None

    def allowed_file(self, filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ModelConfig.ALLOWED_EXTENSIONS

detection_service = ImageDetectionService()

@app.route('/deteksi', methods=['GET', 'POST'])
@login_required
def upload_file():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)

            if file and detection_service.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(filepath)

                predicted_label, probabilities = detection_service.predict_image(filepath)

                if predicted_label:
                    return render_template('result.html',
                                      label=predicted_label,
                                      probabilities=probabilities,
                                      image_url=url_for('uploaded_file', filename=filename))
                
                flash('Error processing image', 'error')
                return redirect(request.url)
            
            flash('Invalid file type', 'error')
            return redirect(request.url)
            
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

    return render_template('deteksi.html')

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.errorhandler(500)
def server_error(e):
    return "Internal server error", 500

    
def initialize_llm():
    groq_api_key = "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"  
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables")
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=groq_api_key)
    return llm

def create_rag_chain(retriever, llm):
    system_prompt = (
        "Anda adalah asisten untuk tugas menjawab pertanyaan yang bernama gold. "
        "Gunakan konteks yang diambil untuk menjawab. "
        "Menjawab menggunakan bahasa Indonesia. "
        "Jika Anda tidak ada jawaban pada konteks, katakan saja 'saya tidak tahu' dan berikan jawaban yang sesuai. "
        "Gunakan maksimal empat kalimat dan pertahankan jawaban singkat.\n\n"
        "{context}"
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return rag_chain

llm = initialize_llm()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
with open('chatbot/vectorstore.pkl', 'rb') as f:
    vectorstore = pickle.load(f)

retriever = vectorstore.as_retriever()
rag_chain = create_rag_chain(retriever, llm)
    
@app.route("/", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first() 
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                flash('Login Successful!', 'success')

                if user.role == 'admin':
                    return redirect(url_for('adminDashboard'))
                
                flash('Welcome back!', 'info')  
                return redirect(url_for('dashboard'))  
            else:
                flash('Invalid email or password.', 'error')  
        else:
            flash('User not found.', 'error')  
    return render_template('login.html', form=form)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        question = data.get('question')

        if not question:
            return jsonify({"error": "Question is required"}), 400

        print(f"Question received: {question}")
        result = rag_chain.invoke({"query": question})
        print(f"Result: {result}")

        if isinstance(result, dict) and 'result' in result:
            answer = result['result']
        else:
            answer = 'Saya tidak tahu'

        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        
        new_user = User(
            username=form.username.data, 
            email=form.email.data,  
            password=hashed_password
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))
    
    return render_template('register.html', form=form)

@app.route("/api/login", methods=['POST'])
def apilogin():
    if not request.is_json:
        return jsonify({"error": "Invalid content type. Expected JSON."}), 400

    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Missing required fields"}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not bcrypt.check_password_hash(user.password, password):  
        return jsonify({"error": "Invalid email or password"}), 401

    access_token = create_access_token(identity={"id": user.id, "username": user.username})
    
    return jsonify({
        "message": "Login successful",
        "access_token": access_token
    }), 200

@app.route("/api/register", methods=['POST'])
def apiregister():
    if not request.is_json:
        return jsonify({"error": "Invalid content type. Expected JSON."}), 400

    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({"error": "Missing required fields"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already taken"}), 409

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "Registration successful"}), 201

@app.route("/google-login")
def googleLogin():
    return oauth.myApp.authorize_redirect(redirect_uri=url_for("googleCallback", _external=True))

@app.route("/signin-google")
def googleCallback():
    token = oauth.myApp.authorize_access_token()
    session["user"] = token  
    
    user_info = token.get("userinfo")  
    if not user_info:
        return redirect(url_for("login"))

    user = User.query.filter_by(email=user_info["email"]).first()  
    if not user:
        user = User(email=user_info["email"], password="")  
        db.session.add(user)
        db.session.commit()

    login_user(user)

    return redirect(url_for("dashboard"))


@app.route("/logout", methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/api/user', methods=['GET'])
@login_required
def get_logged_in_user():
    user_data = {
        "id": current_user.id,
        "username": current_user.username,
    }
    return jsonify(user_data)

@app.route('/api/users', methods=['GET'])
@login_required
def get_all_users():
    users = User.query.all() 
    users_data = [
        {
            "id": user.id,
            "username": user.username,
        }
        for user in users
    ]
    return jsonify(users_data)


@app.route("/dashboard", methods=['GET', 'POST'])
@login_required
def dashboard():
    articles = Artikel.query.all()
    return render_template(
        "dashboard.html", 
        articles=articles,
        session=session.get("user"), 
        pretty=json.dumps(session.get("user"), indent=4)
    )

@app.route("/product")
def product():
    return render_template("product.html")

@app.route("/about")
def about():
    reviews = session.get('reviews', [])
    return render_template("about.html", reviews=reviews)

@app.route("/feedback")
def feedback():
    reviews = session.get('reviews', [])
    return render_template("feedback.html", reviews=reviews)

@app.route('/sentimen')
def sentimen():
    reviews = session.get('reviews', [])
    
    sentiment_results = []
    for review in reviews:
        predicted_class, probabilities = analyzer_indobert.predict_sentiment(review['text'])
        sentiment = "Positif" if predicted_class == 1 else "Negatif"
        sentiment_results.append({
            "text": review['text'],
            "sentiment": sentiment
        })
    
    return render_template('sentimen.html', sentiment_results=sentiment_results)

@app.route('/admin/add_article', methods=['GET', 'POST'])
def add_article():
    if request.method == 'POST':
        image = request.files['image']
        judul = request.form['judul']
        deskripsi = request.form['deskripsi']

        if image and judul and deskripsi:
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)

            new_article = Artikel(image=f'images/{filename}', judul=judul, deskripsi=deskripsi)
            db.session.add(new_article)
            db.session.commit()
            flash('Artikel berhasil ditambahkan!', 'success')
            return redirect(url_for('add_article'))
        else:
            flash('Semua field harus diisi!', 'error')
    return render_template('admin/adminArtikel.html')

@app.route('/api/articles', methods=['GET'])
def apiarticles():
    articles = Artikel.query.all()
    articles_list = [
        {
            "id": article.id,
            "image": url_for('static', filename=article.image, _external=True),
            "judul": article.judul,
            "deskripsi": article.deskripsi
        }
        for article in articles
    ]

    return jsonify({
        "status": "success",
        "message": "Articles retrieved successfully",
        "data": articles_list
    }), 200

@app.route('/api/articles', methods=['POST'])
def postarticle():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid content type. Expected JSON."}), 400

    data = request.get_json()
    judul = data.get('judul')
    deskripsi = data.get('deskripsi')
    image = data.get('image') 

    if not judul or not deskripsi or not image:
        return jsonify({"status": "error", "message": "All fields are required."}), 400

    new_article = Artikel(judul=judul, deskripsi=deskripsi, image=image)
    db.session.add(new_article)
    db.session.commit()

    return jsonify({
        "status": "success",
        "message": "Article created successfully.",
        "data": {
            "id": new_article.id,
            "judul": new_article.judul,
            "deskripsi": new_article.deskripsi,
            "image": new_article.image
        }
    }), 201


@app.route('/api/article/<int:article_id>', methods=['PUT'])
def putarticle(article_id):
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid content type. Expected JSON."}), 400

    data = request.get_json()
    judul = data.get('judul')
    deskripsi = data.get('deskripsi')
    image = data.get('image') 

    article = Artikel.query.get(article_id)
    if not article:
        return jsonify({"status": "error", "message": "Article not found."}), 404

    if judul:
        article.judul = judul
    if deskripsi:
        article.deskripsi = deskripsi
    if image:
        article.image = image

    db.session.commit()

    return jsonify({
        "status": "success",
        "message": "Article updated successfully.",
        "data": {
            "id": article.id,
            "judul": article.judul,
            "deskripsi": article.deskripsi,
            "image": article.image
        }
    }), 200

@app.route('/api/articles/<int:article_id>', methods=['DELETE'])
def delete_article(article_id):
    article = Artikel.query.get(article_id)
    if not article:
        return jsonify({"status": "error", "message": "Article not found."}), 404

    db.session.delete(article)
    db.session.commit()

    return jsonify({
        "status": "success",
        "message": "Article deleted successfully."
    }), 200


@app.route('/add_review', methods=['POST'])
@login_required
def add_review():
    data = request.json
    review_text = data.get('text')

    if not review_text:
        return jsonify({"error": "Feedback text is required"}), 400

    email = current_user.email

    new_feedback = Feedback(
        email=email,
        feedback_text=review_text
    )
    db.session.add(new_feedback)
    db.session.commit()

    return jsonify({"message": "Feedback submitted successfully"})


@app.route("/admin/dashboard")
@login_required
def adminDashboard():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))

    total_users = User.query.filter_by(role='user').count()
    total_admins = User.query.filter_by(role='admin').count()

    feedbacks = Feedback.query.all()

    positive_count = 0
    negative_count = 0
    sentiment_results = []

    for feedback in feedbacks:
        predicted_class, _ = analyzer_indobert.predict_sentiment(feedback.feedback_text)
        sentiment = "Positif" if predicted_class == 1 else "Negatif"
        sentiment_results.append({
            "email": feedback.email,
            "text": feedback.feedback_text,
            "sentiment": sentiment
        })
        if predicted_class == 1:
            positive_count += 1
        else:
            negative_count += 1

    return render_template(
        "admin/adminDashboard.html",
        total_users=total_users,
        total_admins=total_admins,
        positive_count=positive_count,
        negative_count=negative_count,
        sentiment_results=sentiment_results
    )


@app.route("/admin/user")
@login_required
def adminUser():
    if current_user.role != 'admin':
        flash("Anda tidak memiliki izin untuk mengakses halaman ini.", 'danger')
        return redirect(url_for('dashboard'))
    
    users = User.query.filter(User.role != 'admin').all()
    return render_template("admin/adminUser.html", users=users)

@app.route("/setting", methods=['GET', 'POST'])
@login_required
def setting():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        
        if password:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            current_user.password = hashed_password

        
        current_user.username = username
        current_user.email = email

        db.session.commit()

        flash('Profile updated successfully!', 'success')
        return redirect(url_for('dashboard'))  

    return render_template('setting.html')


@app.route("/delete_account", methods=['POST'])
@login_required
def delete_account():
    user = current_user
    db.session.delete(user)
    db.session.commit()

    flash('Your account has been deleted successfully!', 'success')
    logout_user()
    return redirect(url_for('login'))

from flask import redirect, url_for, flash

@app.route('/delete_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    
    try:
        db.session.delete(user)
        db.session.commit()
        flash("Pengguna berhasil dihapus.", "success")
    except Exception as e:
        db.session.rollback()
        flash("Terjadi kesalahan saat menghapus pengguna.", "danger")
    
    return redirect(url_for('adminUser'))

@app.route('/article/<int:article_id>')
def show_article(article_id):
    article = Artikel.query.get_or_404(article_id)  
    return render_template('article.html', article=article)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        detection_service = ImageDetectionService()  
    app.run(debug=True)
