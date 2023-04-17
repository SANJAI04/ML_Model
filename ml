SVM
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__,template_folder="C:/Users/sanja/Downloads/ML_Sanjai/template")

# load the trained SVR model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=46)

svm = SVR(kernel='rbf', C=100, gamma='auto')
svm.fit(X_train, y_train)

# define a route for the home page
@app.route('/')
def home():
    return render_template('index1.html')

# define a route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # get the feature values from the user input form
    features = [float(x) for x in request.form.values()]

    # make a prediction using the trained SVR model
    prediction = svm.predict([features])

    # render the prediction on the prediction page
    return render_template('index1.html', prediction_text='Predicted price: ${:.2f}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(host='localhost',port=5000)
    
<!DOCTYPE html>
<html>
    <head>
        <title>House Price Prediction</title>
    </head>
    <body>
        <h1>House Price Prediction</h1>
        <form action="/predict" method="post">
            <label>CRIM:</label>
            <input type="text" name="feature1"><br>
            <label>ZN:</label>
            <input type="text" name="feature2"><br>
            <label>INDUS:</label>
            <input type="text" name="feature3"><br>
            <label>CHAS:</label>
            <input type="text" name="feature4"><br>
            <label>NOX:</label>
            <input type="text" name="feature5"><br>
            <label>RM:</label>
            <input type="text" name="feature6"><br>
            <label>AGE:</label>
            <input type="text" name="feature7"><br>
            <label>DIS:</label>
            <input type="text" name="feature8"><br>
            <label>RAD:</label>
            <input type="text" name="feature9"><br>
            <label>TAX:</label>
            <input type="text" name="feature10"><br>
            <label>PTRATIO:</label>
            <input type="text" name="feature11"><br>
            <label>B:</label>
            <input type="text" name="feature12"><br>
            <label>LSTAT:</label>
            <input type="text" name="feature13"><br>
            <input type="submit" value="Predict">
        </form>
        {% if prediction_text %}
        <p>{{ prediction_text }}</p>
        {% endif %}
    </body>
</html>


CNN MNIST_PY
import numpy as np
from tensorflow import keras
from keras import layers
from flask import Flask, request, jsonify, render_template
import io
from PIL import Image

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
model = keras.Sequential(
    [
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=3, validation_split=0.1)
model.save("my_model.h5")

app = Flask(__name__,template_folder="C:/Users/sanja/Downloads/ML_MODEL/template")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('L')
            img = np.array(img.resize((28, 28))).astype('float32') / 255.0
            pred = model.predict(img.reshape(1, 28, 28, 1)).argmax()
            return render_template('index.html', result=pred, image_url='/image')
    return render_template('index.html')


@app.errorhandler(400)
def handle_bad_request(e):
    print(str(e))
    return 'Bad request', 400


if __name__=='__main__':
    model = keras.models.load_model("my_model.h5")
    app.run(host='localhost',port=8000)
    
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <title>MNIST Digit Recognition</title>
    <style>
      #image-preview {
        max-width: 300px;
        max-height: 300px;
        margin-bottom: 20px;
      }
    </style>
  </head>
  <body>
    <h1>MNIST Digit Recognition</h1>
    <p>Select an image of a handwritten digit to recognize:</p>
    <form action="/" method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <br><br>
      <input type="submit" value="Recognize">
    </form>
    {% if result %}
    <div>
      <h2>Predicted Label: {{ result }}</h2>
      <img id="image-preview" src="{{ image_url }}" alt="Handwritten Digit">
    </div>
    {% endif %}
  </body>
</html>


CNN_MNIST FASHION MNIST
from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

app = Flask(__name__)

# Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255.
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Perform parameter tuning for finding the best batch size
model = KerasClassifier(build_fn=create_model)
batch_sizes = [32, 64, 128]
param_grid = dict(batch_size=batch_sizes)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

# Train the model with the best batch size
model = create_model()
history = model.fit(X_train, y_train, batch_size=grid_result.best_params_['batch_size'], epochs=3, validation_data=(X_test, y_test))

# Define a route to handle the image recognition
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the image from the form data
        image_file = request.files['file']
        image_data = image_file.read()
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = plt.imread(image_array, format='jpg')
        
        # Preprocess the image
        image = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.
        
        # Make a prediction
        prediction = model.predict_classes(image)[0]
        
        # Generate the graphs
        loss_fig = plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        acc_fig = plt.figure()
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        loss_fig.savefig('static/loss.png')
        acc_fig.savefig('static/accuracy.png')
        # Render the template with the prediction and graphs
        return render_template('mnist.html', prediction=prediction)
    # Render the index template
    return render_template('mnist.html')

if __name__ == 'main':
    app.run(debug=True)


#CNN fashion mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from flask import Flask, render_template, Response ,url_for

app = Flask(__name__)

# Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255.
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Perform parameter tuning for finding the best batch size
model = KerasClassifier(build_fn=create_model)
batch_sizes = [32, 64, 128]
param_grid = dict(batch_size=batch_sizes)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

# Print the best batch size
print("Best batch size: %d" % (grid_result.best_params_['batch_size']))

# Train the model with the best batch size
model = create_model()
history = model.fit(X_train, y_train, batch_size=grid_result.best_params_['batch_size'], epochs=3, validation_data=(X_test, y_test))

# Plot the graph of loss and accuracy in each epoch
@app.route('/')
def index():
    plt.switch_backend('Agg')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('static/loss.png')
    
    plt.clf()
    
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('static/accuracy.png')

    loss_plot_url = url_for('static', filename='loss.png')
    accuracy_plot_url = url_for('static', filename='accuracy.png')

    return render_template('index.html', loss_plot_url=loss_plot_url, accuracy_plot_url=accuracy_plot_url )

@app.route('/plot/<filename>')
def plot(filename):
    return Response(open(f'static/{filename}.png', 'rb').read(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='localhost',port=7000)
    
 #<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Fashion MNIST Model Training Results</title>
  </head>
  <body>
    <h1>Fashion MNIST Model Training Results</h1>
    <h2>Loss vs. Epochs</h2>
    <img src="{{ loss_plot_url }}" alt="Loss plot">
    <h2>Accuracy vs. Epochs</h2>
    <img src="{{ accuracy_plot_url }}" alt="Accuracy plot">
  </body>
</html>


#HOUSE
from flask import Flask, render_template, request
import numpy as np


app = Flask(__name__,template_folder="C:/Users/sanja/Downloads/ML_MODEL/template")

# load the trained SVR model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=46)

svm = SVR(kernel='rbf', C=100, gamma='auto')
svm.fit(X_train, y_train)

# define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# define a route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # get the feature values from the user input form
    features = [float(x) for x in request.form.values()]

    # make a prediction using the trained SVR model
    prediction = svm.predict([features])

    # render the prediction on the prediction page
    return render_template('index.html', prediction_text='Predicted price: ${:.2f}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)

#LOGISTIC REGRESSION HYPERPARAMETERS
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# load the breast cancer dataset
data = load_breast_cancer()

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# create an instance of the logistic regression class
logreg = LogisticRegression()

# specify hyperparameters and their possible values to search over
hyperparameters = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

# create an instance of GridSearchCV to perform hyperparameter tuning
grid_search = GridSearchCV(logreg, hyperparameters, cv=5, n_jobs=-1)

# fit the grid search on the training data
grid_search.fit(X_train, y_train)

# get the best hyperparameters found during the grid search
best_params = grid_search.best_params_
print('Best hyperparameters:', best_params)

# use the best hyperparameters to train a logistic regression model
best_logreg = LogisticRegression(**best_params)
best_logreg.fit(X_train, y_train)

# use the trained model to make predictions on the test data
y_pred = best_logreg.predict(X_test)

# calculate the accuracy of the model on the test data
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy:', accuracy)


#LOGISTIC REGRESSION
class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=1000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient
    
    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
            
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        y_pred = np.round(h)
        
        return y_pred
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        acc = np.mean(y_pred == y)
        
        return acc
