##Drone Detection and Classification Using RF Signals," Division of Computer Science, Indian Institute of Technology Patna, Bihta, 2024

[Drone Detection Paper.pdf](https://github.com/user-attachments/files/21615363/Drone.Detection.pdf)


# Drone Detection and Classification using RF Signals

**Author:** Adil Ansari  
**Affiliation:** Division of Computer Science, Indian Institute of Technology Patna, Bihta  
**Email:** adilansari57@gmail.com

---

## 📖 Overview

This repository contains code and resources for **drone detection and classification** using radio frequency (RF) signals. We leverage deep learning models (MobileNet & Xception) on spectrogram representations of RF data to accurately identify and categorize drone types and flight activities.

---
<img width="800" height="600" alt="tsne_embeddings_3" src="https://github.com/user-attachments/assets/1b5aceaa-4746-4a07-b989-dd47128c86b5" />
<img width="800" height="600" alt="tsne_embeddingss" src="https://github.com/user-attachments/assets/1f451d72-3ca0-4427-b892-a72bd463b09e" />
<img width="477" height="420" alt="tsne_embeddingss1" src="https://github.com/user-attachments/assets/3ce1d565-02b0-40bc-af79-b3f1988a406c" />
<img width="800" height="600" alt="tsne_embeddings_3" src="https://github.com/user-attachments/assets/bc28891a-f4ae-44ea-8af4-238f41fdc050" />


## 🎯 Objectives

- **Detect** the presence of a drone vs. background RF noise  
- **Classify** between multiple drone models (e.g., AR Drone, Bebop, Phantom)  
- **Extend** classification to flight modes and background activities  
- **Compare** performance of lightweight (MobileNet) vs. larger (Xception) architectures  

---

## 🗂️ Repository Structure

├── data/
│ ├── raw/ # Original RF recordings (.csv)
│ ├── spectrograms/ # Generated spectrogram images
│ ├── labels/ # CSV files mapping spectrogram → label
│ └── DroneRF_dataset.md # Dataset description & metadata
│
├── notebooks/
│ ├── 01_generate_spectrum.ipynb # Data preprocessing & spectrogram creation
│ ├── 02_train_mobilenet.ipynb # MobileNet training pipeline
│ └── 03_train_xception.ipynb # Xception training pipeline
│
├── models/
│ ├── mobilenet_weights.h5
│ └── xception_weights.h5
│
├── src/
│ ├── preprocess.py # RF filtering, normalization, chunking
│ ├── spectrogram.py # Spectrogram & Mel-spectrogram generation
│ ├── train.py # Unified training script for any model
│ └── utils.py # Helper functions (data loaders, metrics, plots)
│
├── README.md # ← You are here
├── requirements.txt # Python dependencies
└── LICENSE # MIT License


---

## 📥 Dataset

We use the publicly available **DroneRF Dataset** (Al-Sa’d et al., 2019), which provides:
- RF recordings from multiple drone types (AR Drone, Bebop, Phantom)
- Background RF signals (no-drone activities)
- Metadata: drone model, flight parameters, capture time

**Download & structure:**
1. Clone or download the dataset from [DroneRF GitHub](https://al-sad.github.io/DroneRF/).  
2. Place `.csv` recordings under `data/raw/`.  
3. Run `notebooks/01_generate_spectrum.ipynb` to preprocess and generate spectrogram images in `data/spectrograms/`.

---

## 🔧 Setup & Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/a3il/drone-rf-classification.git
   cd drone-rf-classification

    Create a virtual environment & install dependencies

    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

🔍 Data Preprocessing

    Filtering & Denoising: Median/Gaussian filters to remove high-frequency noise

    Normalization: Scale RF amplitude to −1,1−1,1

    Chunking: Split long recordings into 20,000-sample frames

    Spectrograms:

        Standard STFT spectrograms

        Mel-scaled spectrograms for improved high-frequency resolution

All steps are implemented in src/preprocess.py and src/spectrogram.py.
🚀 Training & Evaluation
1. MobileNet

python src/train.py \
  --model mobilenet \
  --data-dir data/spectrograms/ \
  --labels data/labels/labels_4class.csv \
  --epochs 30 \
  --batch-size 4 \
  --output models/mobilenet_weights.h5

2. Xception

python src/train.py \
  --model xception \
  --data-dir data/spectrograms/ \
  --labels data/labels/labels_4class.csv \
  --epochs 30 \
  --batch-size 4 \
  --output models/xception_weights.h5

Metrics logged: accuracy, precision, recall, F1-score, confusion matrix.
📊 Results Summary
Model	Classes	Accuracy
MobileNet	2-class	98.3%
MobileNet	4-class	97.5%
MobileNet	10-class	97.3%
MobileNet	23-class	81.1%
Xception	4-class	97.6%

    Note: Performance on 10 & 23 classes is limited by small sample sizes. Additional data collection is recommended.

📈 Visualization

    t-SNE Embeddings: Feature separability in 2D

    Training Curves: Loss & accuracy over epochs

    Confusion Matrices: Class-level performance
<img width="800" height="600" alt="tsne_embeddings_3" src="https://github.com/user-attachments/assets/c5795ec7-8999-40d6-b986-65629ed97ac4" />
<img width="640" height="480" alt="val_tsne" src="https://github.com/user-attachments/assets/9a33a702-8d48-46a3-8cc0-bb77fd18c48c" />
<img width="1000" height="1000" alt="tsne_plot_mob_4epoch" src="https://github.com/user-attachments/assets/13472fa2-8854-4443-b9d5-b64de823776a" />
<img width="1000" height="1000" alt="tsne_plot(1)" src="https://github.com/user-attachments/assets/46464f8d-00f9-4ede-a0ac-60f5c3c3cc37" />
<img width="477" height="420" alt="tsne_embeddingss1" src="https://github.com/user-attachments/assets/cddb1a35-d850-432f-a2ed-03363bf27b95" />
<img width="800" height="600" alt="tsne_embeddingss1(1)" src="https://github.com/user-attachments/assets/aa1fa016-812c-419e-8398-08005f584410" />
<img width="800" height="600" alt="tsne_embeddingss" src="https://github.com/user-attachments/assets/4c472d90-5a70-430d-af57-0ed0c014684d" />
<img width="800" height="800" alt="tsne" src="https://github.com/user-attachments/assets/0c118a2a-adc8-4b92-87f0-29052cb4db84" />
<img width="640" height="480" alt="train_tsne" src="https://github.com/user-attachments/assets/0d14d4bf-6320-4852-9614-afbbde735fd8" />
<img width="640" height="480" alt="4 Class_MobNetTuned_glorot_uniform_relu" src="https://github.com/user-attachments/assets/98c6c0f0-ff86-4a0a-98fc-7865388123c2" />
<img width="600" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/98117f95-bfb7-45ad-8236-6fd2f545c5b5" />

See example plots in notebooks/.
🎓 Citation

If you use this work, please cite:

    Mohammad Al-Sa’d et al., “RF-based drone detection and identification using deep learning approaches: an initiative towards a large open source drone database,” Future Generation Computer Systems, vol. 100, pp. 86–97, 2019. DOI: 10.1016/j.future.2019.05.007


📜 License

This project is licensed under the MIT License. See LICENSE for details.


This `README.md` provides a complete guide covering project overview, data preparation, 
