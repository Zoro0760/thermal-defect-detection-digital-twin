# Digital Twin for Thermal Defect Detection
Digital twin-based system for defect detection using IR thermal imaging. The pipeline includes conversion of thermal video to phase images followed by defect analysis (shape, location, size, and depth). Currently, shape detection is implemented; other modules are under development.

#  OVERVIEW
This project focuses on developing a digital twin-based defect detection system for composite structures using infrared (IR) thermal imaging.

The workflow integrates simulation, thermal signal processing, and deep learning for identifying and analyzing subsurface defects.

---

# 🧪 METHODOLOGY

 1. Simulation (MATLAB)
- A 3-layer composite model was created using MATLAB
- Material: Steel
- Dimensions: 30 × 30 × 15 cm³
- A hole defect was introduced at the center
- Thermal simulation was performed to generate IR thermal video data
  
---

 2. Thermal Signal Processing (Python)
- Thermal video converted into phase and amplitude images
- Fast Fourier Transform (FFT) applied
- # Convert thermal signal to phase image
  ```python
fft_result = np.fft.fft(frames)
phase = np.angle(fft_result)
- Frequency used: 3 Hz
- Frames used: 50
- Outputs:
  - Phase images
  - Amplitude images
  - Comparison visualizations
    
---

 3. Dataset Preparation
- Generated images were labeled for defect detection
- Dataset stored and managed using Google Drive
  
---

 4. DEFECT DETECTION MODEL
- Platform: Google Colab (T4 GPU)
- Model: YOLOv8 (Ultralytics)
- from ultralytics import YOLO
    model = YOLO("best.pt")
    results = model("test_image.png")
- Task: Object detection of defects
  
---

  CURRENT PROGRESS

✔ Completed:
- MATLAB simulation
- Thermal-to-phase conversion using FFT
- Dataset generation and labeling
- YOLOv8-based defect detection (bounding box)

🔄 IN PROGRESS:
- Dataset refinement using Roboflow (polygon annotations)
- YOLOv8 segmentation for precise defect boundaries
- Feature extraction (shape, size)

 PLANNED:
 **Phase 1: Data & Segmentation**
- Refine dataset using Roboflow polygon labeling  
- Train YOLOv8 segmentation model  

**Phase 2: Feature Extraction**
- Extract shape and size from segmentation masks  
- Implement defect localization  

**Phase 3: Advanced Analysis**
- Depth estimation using U-Net  
- Thermal phase-based feature learning  

**Phase 4: Integration**
- Digital twin visualization  
- Real-time defect detection system  

---

 FUTURE SCOPE

- Shape extraction using segmentation masks
- Size estimation from pixel-to-real scaling
- Depth prediction using thermal phase information
- Integration with digital twin for predictive analysis
  
---

 🛠️ Technologies Used

- MATLAB
- Python
- OpenCV
- NumPy
- YOLOv8 (Ultralytics)
- Google Colab (T4 GPU)
- Roboflow (planned)
- U-Net (planned)
  
---

 👤 Author
Adithya Shajee  
B-Tech Project
