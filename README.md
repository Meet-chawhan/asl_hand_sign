This is an alphabet hand sign translator which detects hand sign on live cam and displays the corresponding alphabet.

How to setup-
1. Clone or copy this project
   "git clone https://github.com/yourusername/HandSignTranslator.git"  
   "cd HandSignTranslator"
2. Download the model from the "releases" page and place it in the same folder.
3. Install dependencies
   "pip install -r requirements.txt"
4. Run the live_predict.py file and it will start detecting the hand signs.
5. Press 'q' to exit the program.

Other files-
1. The cordinates.py file  was used to convert the images of  hand signs into a csv file for training of the mlp model.
2. model.py file takes csv file as input and trains the mlp model for the live predictor.  
NOTE- cordinates.py and model.py can be used to train model with custom hand signs but the hand signs need to be static or 1 frame long.

Technologies-  
Machine Learning & Deep Learning  
TensorFlow / Keras – Used to build, train, and save the MLP (Multi-Layer Perceptron) model that classifies hand gestures.
scikit-learn – Used for dataset preparation, label encoding, and splitting data into training/testing sets.

Data Handling  
Pandas – For reading and managing CSV files containing extracted hand landmark data.
NumPy – For numerical operations, reshaping data arrays, and feature processing.

Computer Vision & Landmark Extraction  
MediaPipe – For detecting and tracking 3D hand landmarks (21 points per hand).
OpenCV – For video capture from webcam, frame handling, and drawing detected hand connections on screen.

Utilities  
time – Used for timing predictions and controlling frame rate.
Counter (from collections) – Helps analyze class distribution and handle balanced predictions.
