Technical Writeup: Teagardan Video Deepsearch
Sample Data

data/sample_video_text.jsonl: Audio and visual descriptions from a sample video.
data/sample_frame_text.jsonl: Frame-specific descriptions.
data/frames/sample_frame_0001.jpg: Sample extracted frame.
Sample video: [https://www.kaggle.com/datasets/teagarda/a-video-inside-an-automobile]

To test with sample data:
python src/server.py


Access this running on my machine and query using the JSONL files in data here: https://0ea2cfce1492.ngrok-free.app
Download the sample video from the Kaggle dataset link above.

Video Demo
[https://www.youtube.com/watch?v=tlRAIjsrG_E]


Web Interface
Frame Output

Architecture
Teagardan Video Deepsearch is a Flask-based web application integrated with a video processing pipeline. The system:

Video Processing: Uses FFmpeg and OpenCV to extract frames and detect scene changes via Mean Squared Error (MSE). Supports Casual (faster, low-res) and Pro (detailed, high-res) modes.
Audio Transcription: Employs Whisper to transcribe audio into segments, paired with visual data.
Visual Analysis: Leverages Gemma 3N via Ollama to generate frame and chunk descriptions, using multimodal capabilities.
Web Interface: Flask serves a dynamic UI with "Ask" mode for querying video content (using BM25 ranking) and "Scene" mode for frame-specific analysis.
Data Storage: Saves audio-visual descriptions in JSONL files (sample_video_text.jsonl, sample_frame_text.jsonl) for efficient retrieval.

Gemma 3N Usage
The Gemma 3N model (gemma3n:e4b-it-q8_0) is central to the project, running locally via Ollama. It:

Generates detailed visual descriptions for video chunks and individual frames, processing images and context-aware prompts.
Supports multimodal inputs (text and images) for accurate frame analysis in the web interface.
Optimizes for Apple Silicon (tested on Mac mini), leveraging unified memory for efficient inference.

Challenges Overcome

Scene Detection Accuracy: Tuned MSE thresholds (1000 for Casual, 500 for Pro) to balance sensitivity and performance.
Resource Constraints: Implemented batch processing (batch size 20 for Casual, 1 for Pro) to manage memory usage during frame analysis.
Context Integration: Enriched frame-level analysis by incorporating chunk-level audio-visual context, improving query relevance.

Technical Choices

Whisper for Audio: Chosen for robust speech-to-text performance across diverse audio conditions.
BM25 for Ranking: Efficiently ranks frames based on combined audio-visual text, ensuring relevant query results.
JSONL for Storage: Scalable format for storing timestamps, audio, and visual data, enabling fast retrieval.
Flask for Web: Lightweight and flexible for building an interactive UI with minimal overhead.
OpenCV and FFmpeg: Industry-standard tools for reliable video processing and frame extraction.

Code Repository
GitHub: Teagardan/videosearch
Live Demo
[https://0ea2cfce1492.ngrok-free.app]
This project demonstrates a functional proof-of-concept, with all components integrated to deliver a seamless video search experience, backed by Gemma 3N’s multimodal capabilities.