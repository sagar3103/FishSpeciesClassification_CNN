# 🎣 Fish Species Classifier

## 📌 Overview
This project is a **deep learning-based fish species classification system**.  
It uses a **Convolutional Neural Network (CNN)** to identify different types of fish from images.  
The trained model can recognize fish species and can be used for research, fish market sorting, and marine biodiversity monitoring.

---

## 🎯 Objective
The main goal of this project is to:
- Accurately classify fish species from images.
- Reduce manual effort in identifying fish.
- Provide a simple, automated, and efficient classification tool.

---

## 🗂 Dataset
The dataset contains **images of various fish species**.  
These images are:
- Preprocessed (resized, normalized) before training.
- Split into **training**, **validation**, and **testing** sets.

---

## 🛠 Technologies Used
- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **Google Colab** (for training)
- **H5 & JSON model files** (for saving the trained model)

---

## ⚙ How It Works
1. **Data Preprocessing**
   - All fish images are resized and normalized for uniformity.
   - Images are labeled according to species.

2. **Model Training**
   - A CNN model is built using Keras.
   - The model learns patterns in the images to distinguish between fish species.

3. **Model Saving**
   - The trained model’s structure is saved as `fish_classifier_model.json`.
   - The learned weights are saved as `fish_classifier_model.h5`.

4. **Prediction**
   - A new fish image can be passed to the model.
   - The model predicts the species with a probability score.

---

## 📂 Files in This Project
- **`fish_classifier_model.h5`** → Stores the trained model's weights.
- **`fish_classifier_model.json`** → Stores the model's architecture.
- **Training Notebook** → Contains code for training, validation, and testing.

---

## 🚀 How to Run
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/fish-classifier.git
2. Install dependencies:
   pip install tensorflow numpy matplotlib
   ```
3. Load and run the training notebook in Google Colab or locally.

4. Use the saved model to make predictions:

```
from keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model architecture
with open('fish_classifier_model.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load model weights
model.load_weights('fish_classifier_model.h5')

# Load and preprocess an image
img = image.load_img('test_fish.jpg', target_size=(64, 64))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
print("Predicted class:", prediction)
```

## 📊 Results
- Achieved high accuracy on the test dataset.

- Model can classify fish species in real-time from new images.

## 💡 Applications
- Marine biology research

- Fish market sorting and packaging

- Educational tools for students

- Fisheries management

## ✅ Conclusion
This project demonstrates the power of deep learning in automating fish species identification.
It reduces human effort, speeds up the process, and improves accuracy — proving AI’s potential in marine research and commercial applications.

🤝 Contributions
Feel free to fork this project and enhance the model with:

- More fish species

- Better image preprocessing

- Real-time webcam integration

MIT License

Copyright (c) 2025 Sagar Zujam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
