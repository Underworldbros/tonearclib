# ToneArcLib – v0.1.1

**ToneArcLib** is a semantic audio analysis tool that bridges the gap between human perception and machine-readable audio features. It extracts tonal, rhythmic, and spectral characteristics from `.wav` files and outputs structured JSON or a comprehensive PDF report.

---

## 🚀 Features

- Audio structure and loudness profiling
- Key and mode estimation
- MFCC and chroma feature extraction
- Beat, onset, and silence detection
- Spectral centroid, flatness, contrast, rolloff
- Mood tagging and temporal metrics
- Visual PDF report export
- JSON output for downstream ML/AI use

---

## 🛠 Usage

Run from the command line:

```bash
python TrackAnalysis_full_demo.py "path/to/audio.wav" --out "output/folder"
