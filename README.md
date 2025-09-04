# Teagardan Video Deepsearch

Teagardan Video Deepsearch revolutionizes video content organization and retrieval by combining audio transcription, visual analysis, and scene detection. Powered by the Gemma 3N model via Ollama, it transcribes audio using Whisper, analyzes visuals for detailed descriptions, detects scene changes with OpenCV, and stores results in JSONL files for efficient querying. Ideal for educational tools, content summarization, and interactive media exploration, this project offers a seamless way to search and navigate video content.

## Features
- **Audio Transcription**: Uses Whisper to transcribe video audio accurately.
- **Visual Analysis**: Leverages Gemma 3N via Ollama for detailed frame descriptions.
- **Scene Detection**: Detects changes using Mean Squared Error (MSE) with OpenCV.
- **Processing Modes**: Casual (fast, low-res) and Pro (detailed, high-res) modes.
- **Web Interface**: Flask-based UI with "Ask" mode for querying video content and "Scene" mode for frame-specific analysis.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Teagardan/videosearch.git
   cd videosearch

brew install ffmpeg  # macOS
sudo apt-get install ffmpeg  # Ubuntu

conda create -n videoknowledge python=3.13
conda activate videoknowledge

pip install -r requirements.txt
ollama pull gemma3n:e4b-it-q8_0
ollama run gemma3n:e4b-it-q8_0

python src/video_processing.py
Enter the video file path (e.g., data/sample_video.mp4) and output directory (e.g., output/).
Choose Casual (1) or Pro (2) mode.

python src/server.py

Demo
Watch our 3-minute demo showcasing video search capabilities: [(https://www.youtube.com/watch?v=tlRAIjsrG_E)]
In action: https://0ea2cfce1492.ngrok-free.app

## Screenshots
![Web Interface](https://raw.githubusercontent.com/Teagardan/videosearch/main/screenshots/Teagardan-kaggle_1.png)
![Frame Output](https://raw.githubusercontent.com/Teagardan/videosearch/main/screenshots/Teagardan-kaggle_2.png)


## License

This project, including all code and the technical writeup, is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See [LICENSE.txt](LICENSE.txt) for details.

Attribution: Please credit 9497-2957 Québec Inc. when using or redistributing this work. For inquiries, contact teagardan@outlook.com.

## Contributing

Contributions are welcome! Submit pull requests for bug fixes or enhancements. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.