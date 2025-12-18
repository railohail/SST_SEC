# Speech Command App

A hands-free speech-to-text application with Chinese language support and voice-based correction commands. Works on **macOS** and **Windows**.

## Features

- **Speech-to-Text**: Uses Whisper for accurate Chinese transcription
- **Voice Corrections**: Fix mistakes using spoken commands instead of typing
- **Visual Feedback**: System tray icon with blinking indicator during recording
- **Shuffle Effect**: Text scrambles briefly when applying corrections
- **Global Hotkey**: Toggle recording from anywhere
  - macOS: `Cmd+Shift+Space`
  - Windows: `F9`
- **Multiple Processors**: Choose between ML model, rule-based, or Gemini API

## Workflow

```
1. [Press Hotkey] Start recording (icon blinks)
2. Speak: "今天天氣很好"
3. [Press Hotkey] Stop → text is typed at cursor

If Whisper made a mistake (e.g., "今天天器很好"):
4. [Press Hotkey] Start recording
5. Speak correction: "把器改成氣"
6. [Press Hotkey] Stop → text shuffles → corrected to "今天天氣很好"
```

## Supported Correction Commands

| Command | Format | Example |
|---------|--------|---------|
| Delete | `刪除X` | "刪除錯字" |
| Replace | `把X改成Y` | "把器改成氣" |
| Insert Before | `在X前面新增Y` | "在好前面新增很" |
| Insert After | `在X後面新增Y` | "在天後面新增氣" |

## Installation

### 1. Clone and setup

```bash
cd speech_command_app
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install PyAudio

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Windows:**
```bash
pip install pyaudio
```

### 4. Download model weights

```bash
# Create the model directory
mkdir -p models/model_weights

# Download using huggingface-cli
pip install huggingface_hub
huggingface-cli download railohail/speech-command-model model_crf_enhanced.pt --local-dir models/model_weights
```

Or manually download from:
https://huggingface.co/railohail/speech-command-model/blob/main/model_crf_enhanced.pt

## Usage

### Basic usage

```bash
python main.py
```

### With rule-based processor (faster, no GPU needed)

```bash
python main.py --notML
```

### With Gemini API (requires internet)

```bash
python main.py --api
```

### Custom hotkey

```bash
python main.py --hotkey "<cmd>+<alt>+<space>"
```

## System Requirements

### macOS

Grant these permissions in System Preferences → Security & Privacy:
1. **Microphone**: For audio recording
2. **Accessibility**: For global hotkey and keyboard simulation

### Windows

1. **Microphone**: Allow microphone access in Windows Settings
2. Run as Administrator if hotkey doesn't work

## Project Structure

```
speech_command_app/
├── main.py                 # Entry point
├── config.py               # Configuration (auto-detects platform)
├── requirements.txt        # Dependencies
├── LICENSE                 # MIT License
├── img/                    # Icons
│   ├── speech-synthesis.png
│   └── speech-synthesis_red.png
├── models/
│   ├── crf_model.py        # BERT+CRF model
│   └── model_weights/      # Model .pt files
├── services/
│   ├── audio_recorder.py   # Microphone recording
│   ├── whisper_service.py  # Speech-to-text
│   ├── sequence_labeler.py # Model inference
│   ├── command_processor.py# ML-based command parsing
│   ├── rule_based_processor.py # Rule-based fallback
│   └── gemini_processor.py # Gemini API processor
├── utils/
│   ├── hotkey_manager.py   # Global hotkey
│   ├── keyboard_simulator.py # Text typing + shuffle effect
│   ├── accessibility.py    # Platform-specific text selection
│   └── debug_overlay.py    # Status display
└── tests/
    ├── test_command_processor.py
    ├── test_mode.py        # Test without microphone
    └── test_with_model.py  # Test with full model
```

## Technical Details

### Model Architecture

```
BERT (768d) → Linear(768→512) → LayerNorm → GELU → Dropout
            → Linear(512→256) → LayerNorm → GELU → Dropout
            → Linear(256→3) → CRF
```

### Labels

- `O`: Normal text
- `B-Modify`: Position to modify
- `B-Filling`: Replacement content

### Device Support

- **CUDA**: Auto-detected for NVIDIA GPUs
- **Apple Silicon (M1/M2/M3)**: Uses MPS acceleration
- **CPU**: Fallback for all other systems

## Troubleshooting

### "No speech detected"
- Check microphone permissions
- Speak louder or closer to microphone
- Check audio input device in system settings

### Hotkey not working
- **macOS**: Grant Accessibility permission
- **Windows**: Try running as Administrator
- Check for hotkey conflicts with other apps

### Model loading slow
- First load downloads BERT weights (~700MB)
- Subsequent runs use cached model

### System tray icon not showing
- Check if other apps are blocking the system tray
- Try restarting the application

## License

MIT License - see [LICENSE](LICENSE) file.
