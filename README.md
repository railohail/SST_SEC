# Speech Command App

A hands-free speech-to-text application for Mac with Chinese language support and voice-based correction commands.

## Features

- **Speech-to-Text**: Uses Whisper (small model) for accurate Chinese transcription
- **Voice Corrections**: Fix mistakes using spoken commands instead of typing
- **Global Hotkey**: Toggle recording from anywhere with Cmd+Shift+Space
- **Debug Mode**: Transparent overlay showing status (optional)

## Workflow

```
1. [Press Hotkey] Start recording
2. Speak: "今天天氣很好"
3. [Press Hotkey] Stop → text is typed at cursor

If Whisper made a mistake (e.g., "今天天器很好"):
4. [Press Hotkey] Start recording
5. Speak correction: "把器改成氣"
6. [Press Hotkey] Stop → text is corrected to "今天天氣很好"
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
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install PyAudio (Mac specific)

```bash
brew install portaudio
pip install pyaudio
```

### 4. Copy model weights

Copy your trained model to:
```
models/model_weights/model_crf_enhanced.pt
```

Or create a symlink:
```bash
ln -s /path/to/your/model.pt models/model_weights/model_crf_enhanced.pt
```

## Usage

### Basic usage (debug mode)

```bash
python main.py --debug
```

### Production mode (invisible)

```bash
python main.py --no-debug
```

### Custom hotkey

```bash
python main.py --hotkey "<cmd>+<alt>+<space>"
```

## Mac Permissions

The app requires these permissions (System Preferences → Security & Privacy):

1. **Microphone**: For audio recording
2. **Accessibility**: For global hotkey and keyboard simulation

## Project Structure

```
speech_command_app/
├── main.py                 # Entry point
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── models/
│   ├── crf_model.py        # BERT+CRF model
│   └── model_weights/      # Model .pt files
├── services/
│   ├── audio_recorder.py   # Microphone recording
│   ├── whisper_service.py  # Speech-to-text
│   ├── sequence_labeler.py # Model inference
│   └── command_processor.py# Command parsing
├── utils/
│   ├── hotkey_manager.py   # Global hotkey
│   ├── keyboard_simulator.py # Text typing
│   └── debug_overlay.py    # Status display
└── tests/
    └── test_command_processor.py
```

## Technical Details

### Model Architecture

BERT (768d) → Linear(768→512) → LayerNorm → GELU → Dropout
            → Linear(512→256) → LayerNorm → GELU → Dropout
            → Linear(256→3) → CRF

### Labels

- `O`: Normal text
- `B-Modify`: Position to modify
- `B-Filling`: Replacement content

### Device Support

- Apple Silicon (M1/M2): Uses MPS acceleration
- Intel Mac: CPU inference
- CUDA: If available

## Troubleshooting

### "No speech detected"
- Check microphone permissions
- Speak louder or closer to microphone
- Check audio input device in System Preferences

### Hotkey not working
- Grant Accessibility permission
- Check for hotkey conflicts with other apps
- Try a different hotkey

### Model loading slow
- First load downloads BERT weights (~700MB)
- Subsequent runs use cached model

## License

For educational purposes.
