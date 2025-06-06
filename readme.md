This project focuses on detecting and analyzing vehicle crash damage using a deep learning pipeline. It leverages CNN architectures for image classification, damage severity estimation, and Grad-CAM for model interpretability. The implementation is done in PyTorch within a Jupyter Notebook.

🧠 Features
Crash image classification using CNN (ResNet-based)

Severity percentage estimation for damage

Grad-CAM visualization for interpretability

Detailed report generation for input images

Ready for extension to video-based input streams

🗂️ Project Structure
damage_prediction_v2.ipynb - Main notebook with all implementation steps
data/ - Dataset directory
outputs/ - Visualizations and prediction results
models/ - Saved trained model weights

⚙️ Setup Instructions
Clone the repository:

git clone https://github.com/yourusername/crash-detection-elumina.git
cd crash-detection-elumina

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate (On Windows: venv\Scripts\activate)

Install dependencies:

pip install -r requirements.txt

If requirements.txt is missing, install essential packages manually:

pip install torch torchvision matplotlib opencv-python

Launch the notebook:

jupyter notebook damage_prediction_v2.ipynb

📊 Model Overview
Base Model: ResNet50 (pretrained)

Framework: PyTorch

Loss Function: CrossEntropyLoss

Optimizer: Adam

Accuracy: ~87% on validation set
