# Low-Light-Video-Enhacement
# LIME-Based Low-Light Video Enhancement

This project implements an advanced **low-light video enhancement** algorithm based on **LIME** (Low-light Image Enhancement via Illumination Map Estimation) with several post-processing steps including **BM3D denoising**, **CLAHE**, **bilateral filtering**, and **sharpening**.

---

## 🚀 Features

- Illumination map refinement using FFT-based optimization  
- BM3D denoising with adjustable sigma  
- Adaptive CLAHE and soft sharpening  
- Bilateral filtering for smoothing block noise  
- Frame-by-frame video enhancement (demo: 5 frames)

---

## 🧩 Requirements

Install dependencies:

```bash
pip install -r requirements.txt

💡 Usage

Run the main enhancement script:
python src/lime_video_enhancement.py
Set your input and output video paths in the script:
INPUT_VIDEO = r"C:\path\to\input.mp4"
OUTPUT_VIDEO = r"C:\path\to\output.mp4"

The script processes the first 5 frames and saves:

The enhanced video (output_video_5frames.mp4)

Individual frames in the folder output_frames_5frames/

