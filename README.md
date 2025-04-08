# ToneArcLib

**ToneArcLib** is a semantic audio analysis tool designed to bridge the gap between human perception and machine-readable audio features. It extracts musical structure, rhythm, tonality, and basic mood indicators from `.wav` audio files for further use by humans, AI models, or procedural systems.

---

## 🔧 Features

- Analyze `.wav` files from the command line
- Generate structured reports in either:
  - **Standard mode**: for human-friendly display
  - **Extended mode**: for JSON-based LLM integration
- Outputs detailed features including:
  - Track name, sample rate, duration
  - Tempo, key, mode
  - Beat structure and spectral centroid
  - (Basic) mood tagging
- CLI errors are handled gracefully with clear messages

---

## 📦 Installation

Clone and install the package locally:

```bash
git clone https://github.com/Underworldbros/tonearclib.git
cd tonearclib
pip install .
```

---

## 🚀 Usage

### Basic syntax

```bash
tonearc <filepath> [--extended] [--out <output_path_or_dir>]
```

### Example

```bash
tonearc "input.wav" --extended --out "output/"
```

- `--extended`: Outputs full JSON profile for use in AI pipelines
- `--out`: Optional file or folder path to save output

---

## 📄 Output (Extended Mode)

Generates a `.json` file like:

```json
{
  "track": "input.wav",
  "duration": 142.6,
  "sample_rate": 44100,
  "key": "D",
  "bpm": 110,
  "tonality": "minor",
  "mood_tag": "Undefined"
}
```

---

## 🧐 LLM Integration Use Case

The extended output format is ideal for:

- AI agents interpreting musical environments
- Generating adaptive game soundtracks
- Analyzing large track libraries semantically

---

## 📌 Known Limitations

- Mood tagging is a placeholder and will be enhanced in future versions
- Only `.wav` input is supported at this time
- PDF output in standard mode is still experimental

---

## 📜 Example Scripts

- `examples/TrackAnalysis_full_demo.py` – Legacy standalone version of the full analysis pipeline.  
  Useful for testing or understanding the full process outside the modular CLI.

---

## 📜 License

See LICENSE for details.

---

## 🤝 Contributing

ToneArcLib is in active development. PRs and issues are welcome.
