# High Precision Laser Detection System

## 👤 Project Member
- Developed by: Hao Chen 2022213763

## 🔍 Project Overview

This is a high-precision laser point detection system based on OpenCV, specifically designed for real-time, accurate detection and tracking of red laser points. The system incorporates multiple innovative technologies to achieve high precision, low latency, and robust detection performance.

![img](img.png)

## 🌟 Core Features

- **Real-time High-precision Detection**: Accurate laser point localization within 2 meters
- **Fast Response**: Optimized algorithms ensure low latency and real-time performance
- **Strong Anti-interference Capability**: Multiple validation mechanisms ensure detection accuracy
- **Adaptive Threshold Adjustment**: Intelligent adaptation to different environmental conditions
- **Visualization Interface**: Intuitive visual feedback and data display
- **High-performance Optimization**: Carefully tuned parameters and algorithms

## 🛠️ Technical Implementation

### 1. Image Preprocessing Optimization
```python
def preprocess_frame(self, frame):
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
```
- CLAHE technology for enhanced image contrast
- LAB color space conversion for improved color recognition
- Adaptive histogram equalization for better performance under varying lighting conditions

### 2. Multi-feature Fusion Detection
- Combines multiple key features for comprehensive scoring:
  * Color features (HSV space dual threshold)
  * Shape features (circularity analysis)
  * Brightness features (intensity distribution)
  * Spatial features (density calculation)

### 3. Intelligent Threshold System
```python
self.params = {
    'red_lower1': np.array([0, 120, 160]),    # Lower bound for red range 1
    'red_upper1': np.array([15, 255, 255]),   # Upper bound for red range 1
    'red_lower2': np.array([160, 120, 160]),  # Lower bound for red range 2
    'red_upper2': np.array([180, 255, 255]),  # Upper bound for red range 2
}
```
- Carefully calibrated dual threshold ranges
- Adaptive brightness threshold
- Dynamic area constraints

### 4. Efficient Smoothing Algorithm
- Decreasing weight-based moving average
- Intelligent outlier filtering
- Fast-response buffering strategy

### 5. Performance Optimization Techniques
```python
# Camera parameter optimization
self.cap.set(cv2.CAP_PROP_FPS, 60)         # High frame rate acquisition
self.cap.set(cv2.CAP_PROP_EXPOSURE, -8)     # Exposure optimization
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # High resolution
```
- Camera parameter optimization
- Algorithm computation optimization
- Memory usage optimization

## 💡 Innovation Highlights

1. **Dual HSV Threshold Detection**
   - Innovative use of dual threshold ranges
   - Effective handling of red color distribution in HSV space
   - Improved detection accuracy

2. **Adaptive Preprocessing Pipeline**
   - Dynamic contrast enhancement
   - Intelligent noise suppression
   - Strong environmental adaptability

3. **Multi-dimensional Feature Fusion Scoring**
   ```python
   score = (circularity * 0.3 +        # Circle metric
           (mean_intensity/255) * 0.4 + # Brightness
           density * 0.3)              # Density
   ```
   - Integration of multiple feature dimensions
   - Intelligent weight distribution
   - Strong robustness

4. **Efficient Smoothing Strategy**
   - Minimized latency
   - Maintained tracking stability
   - Rapid response to changes

## 📊 Performance Metrics

- **Detection Range**: Stable detection within 2 meters
- **Response Time**: <50ms
- **Accuracy**: >95% (under ideal conditions)
- **Stability**: Jitter <5 pixels
- **Fault Tolerance**: Good adaptation to lighting changes and background interference

## 🔧 Usage Guide

### Install Dependencies
```bash
pip install opencv-python
pip install numpy
```

### Run Program
```python
detector = HighPrecisionLaserDetector()
detector.run()
```

### Hotkeys
- `q`: Exit program
- `r`: Reset detection
- `+`: Increase threshold
- `-`: Decrease threshold

## 🎯 Optimization Suggestions

For optimal detection performance:

1. **Environment Setup**
   - Maintain moderate ambient lighting
   - Avoid highly reflective surfaces
   - Keep camera stable

2. **Parameter Tuning**
   - Adjust HSV thresholds based on actual scene
   - Tune detection area range appropriately
   - Adjust smoothing parameters as needed

3. **Hardware Selection**
   - Use high-quality camera
   - Choose appropriate brightness laser pointer
   - Ensure stable computing performance

## 📈 Technical Details

### Core Algorithm Flow
1. Image preprocessing and enhancement
2. Dual HSV threshold segmentation
3. Morphological processing and noise removal
4. Contour detection and feature extraction
5. Multi-dimensional feature scoring
6. Smoothing processing and position estimation

### Key Parameter Optimization
- Precise calibration of HSV threshold ranges
- Morphological operation kernel size selection
- Feature weight distribution optimization
- Smoothing algorithm parameter tuning