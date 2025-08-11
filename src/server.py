import os
import json
import re
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
from rank_bm25 import BM25Okapi
from bisect import bisect_left
import ollama
import logging
os.environ["OLLAMA_HOST"] = "https://teagardan.ngrok.app"

app = Flask(__name__, template_folder='templates')
app.config['BASE_OUTPUT_DIR'] = os.path.expanduser('/Users/siliconweaver/Documents/Teagardan/VIDEO_GEMMA3N/output')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STOP_WORDS = set([
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'for', 'of', 'with',
    'in', 'on', 'at', 'to', 'from', 'by', 'about', 'as', 'into', 'like', 'through',
    'after', 'over', 'between', 'out', 'against', 'during', 'without', 'before',
    'under', 'around', 'among'
])

def clean_text(text):
    lines = text.split('\n')
    cleaned_lines = []
    prev_line = None
    for line in lines:
        if line.strip() != prev_line:
            cleaned_lines.append(line)
            prev_line = line.strip()
    return '\n'.join(cleaned_lines)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    return [word for word in words if word not in STOP_WORDS]

def find_relevant_frames(paired_data, query):
    if not paired_data or not query:
        return []
    corpus = [preprocess_text(entry['combined_text']) for entry in paired_data]
    bm25 = BM25Okapi(corpus)
    tokenized_query = preprocess_text(query)
    scores = bm25.get_scores(tokenized_query)
    for entry, score in zip(paired_data, scores):
        entry['bm25_score'] = score
    return sorted([entry for entry in paired_data if entry.get('bm25_score', 0) > 0], key=lambda x: x['bm25_score'], reverse=True)

def select_diverse_frames(relevant_frames, max_frames=5, min_time_diff=5):
    if not relevant_frames:
        return []
    selected = []
    for frame in relevant_frames:
        if not selected or all(abs(frame['timestamp'] - s['timestamp']) >= min_time_diff for s in selected):
            selected.append(frame)
            if len(selected) == max_frames:
                break
    return selected

def find_closest_frame(paired_data, timestamp):
    if not paired_data:
        return None
    timestamps = [frame['timestamp'] for frame in paired_data]
    idx = bisect_left(timestamps, timestamp)
    if idx == 0:
        return paired_data[0]
    if idx == len(paired_data):
        return paired_data[-1]
    before = paired_data[idx - 1]
    after = paired_data[idx]
    return after if after['timestamp'] - timestamp < timestamp - before['timestamp'] else before

def generate_response(query, context, num_predict=300, response_type='detailed'):
    if not context:
        return "No relevant context found to answer the query."
    instruction = (
        f"Provide a {response_type} response using only the provided context. "
        "If the context lacks sufficient information, respond with 'The provided context does not contain enough information to answer this query.'"
    )
    prompt = (
        f"{instruction} to the query: '{query}'. "
        f"Provide your answer in a single, continuous paragraph without headings or numbered sections.\n\n"
        f"Context: {context}"
    )
    try:
        response = ollama.generate(model="gemma3n:e4b-it-q8_0", prompt=prompt, options={"num_predict": num_predict, "temperature": 0.7})
        return response['response'] if response else "No response generated"
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_frame_response(query, frame_context, screenshot_path=None):
    if not query or query.lower() == "describe this frame.":
        prompt = (
            f"Provide a concise description of the frame content based on the following context, starting directly with the description and avoiding any introductory phrases: {frame_context}"
        )
    else:
        prompt = (
            f"Answer the query '{query}' using only the following context from the frame. "
            f"If the context does not contain enough information to answer the query, start your response with '[INSUFFICIENT]'. "
            f"Frame context: {frame_context}"
        )
    try:
        images = [screenshot_path] if screenshot_path else []
        response = ollama.generate(model="gemma3n:e4b-it-q8_0", prompt=prompt, images=images, options={"num_predict": 100, "temperature": 0.7})
        return response['response'] if response else "No response generated"
    except Exception as e:
        return f"Error generating frame response: {str(e)}"

def load_json_data(jsonl_path):
    data = []
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        data.append(entry)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON line: {e}")
        return sorted(data, key=lambda x: x['timestamp'])
    except FileNotFoundError:
        logger.error(f"JSONL file {jsonl_path} not found.")
        return []

@app.route('/')
def home():
    video_dirs = [d for d in os.listdir(app.config['BASE_OUTPUT_DIR']) if d.startswith('output_video')]
    return render_template('home.html', video_dirs=video_dirs)

@app.route('/video/<video_dir>', methods=['GET'])
def video_page(video_dir):
    mode = request.args.get('mode', 'ask')
    if mode not in ['ask', 'pause', 'scene']:
        mode = 'ask'
    output_dir = os.path.join(app.config['BASE_OUTPUT_DIR'], video_dir)
    json_path = os.path.join(output_dir, 'frame_text_repo.jsonl')
    paired_data = load_json_data(json_path)
    video_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
    if len(video_files) != 1:
        logger.error(f"Expected exactly one .mp4 file in {output_dir}, found {len(video_files)}")
        return "Error: Expected exactly one .mp4 file in the directory.", 400
    video_file = video_files[0]
    return render_template('video.html', video_dir=video_dir, video_file=video_file, mode=mode, paired_data=paired_data)

@app.route('/api/chat/<video_dir>', methods=['POST'])
def chat(video_dir):
    output_dir = os.path.join(app.config['BASE_OUTPUT_DIR'], video_dir)
    json_path = os.path.join(output_dir, 'frame_text_repo.jsonl')
    paired_data = load_json_data(json_path)
    logger.info(f"Loaded {len(paired_data)} frames from {json_path}")
    
    query = request.json.get('query', '')
    if not query:
        return jsonify({'brief_response': 'Please provide a query.', 'detailed_response': 'Please provide a query.', 'frames': []}), 400
    
    relevant_frames = find_relevant_frames(paired_data, query)
    logger.info(f"Found {len(relevant_frames)} relevant frames with query: {query}")
    selected_frames = select_diverse_frames(relevant_frames)
    logger.info(f"Selected {len(selected_frames)} diverse frames")
    
    if selected_frames:
        primary_text = "\n".join([clean_text(frame['combined_text']) for frame in selected_frames])
        brief_response = generate_response(query, primary_text, num_predict=100, response_type='brief')
        detailed_response = generate_response(query, primary_text, num_predict=300, response_type='detailed')
        
        formatted_frames = []
        for frame in selected_frames:
            formatted_frames.append({
                'timestamp': frame['timestamp'],
                'formatted_timestamp': f"{frame['timestamp']}s",
                'combined_text': frame['combined_text'],
                'relevance_score': frame.get('bm25_score', 0),
                'screenshot': frame.get('screenshot', '')
            })
        
        logger.info(f"Returning {len(formatted_frames)} formatted frames")
        return jsonify({'brief_response': brief_response, 'detailed_response': detailed_response, 'frames': formatted_frames})
    return jsonify({'brief_response': "No relevant information found.", 'detailed_response': "No relevant information found.", 'frames': []})

@app.route('/api/frame_data/<video_dir>', methods=['POST'])
def frame_data(video_dir):
    output_dir = os.path.join(app.config['BASE_OUTPUT_DIR'], video_dir)
    json_path = os.path.join(output_dir, 'frame_text_repo.jsonl')
    paired_data = load_json_data(json_path)
    if not paired_data:
        return jsonify({'error': 'No data found', 'internet_search': False}), 404
    
    timestamp = request.json.get('timestamp')
    query = request.json.get('query', '')
    if timestamp is None:
        return jsonify({'error': 'Timestamp not provided', 'internet_search': False}), 400
    
    closest_frame = find_closest_frame(paired_data, timestamp)
    if closest_frame:
        screenshot_path = os.path.join(output_dir, closest_frame.get('screenshot', '')) if closest_frame.get('screenshot') else None
        frame_context = closest_frame['combined_text']
        frame_response = generate_frame_response(query, frame_context, screenshot_path)
        logger.info(f"Frame response for query '{query}' at {timestamp}s: {frame_response}")
        return jsonify({'response': frame_response, 'timestamp': closest_frame['timestamp'], 'internet_search': False})
    return jsonify({'error': 'No frame found', 'internet_search': False}), 404

@app.route('/api/image_description/<video_dir>', methods=['POST'])
def image_description(video_dir):
    output_dir = os.path.join(app.config['BASE_OUTPUT_DIR'], video_dir)
    json_path = os.path.join(output_dir, 'frame_text_repo.jsonl')
    paired_data = load_json_data(json_path)
    if not paired_data:
        return jsonify({'error': 'No data found', 'internet_search': False}), 404
    
    timestamp = request.json.get('timestamp')
    if timestamp is None:
        return jsonify({'error': 'Timestamp not provided', 'internet_search': False}), 400
    
    closest_frame = find_closest_frame(paired_data, timestamp)
    if closest_frame:
        screenshot_path = os.path.join(output_dir, closest_frame.get('screenshot', '')) if closest_frame.get('screenshot') else None
        frame_context = closest_frame['combined_text']
        description = generate_frame_response("describe this frame.", frame_context, screenshot_path)
        logger.info(f"Image description at {timestamp}s: {description}")
        return jsonify({'description': description, 'timestamp': closest_frame['timestamp'], 'internet_search': False})
    return jsonify({'error': 'No frame found', 'internet_search': False}), 404

@app.route('/api/image_query/<video_dir>', methods=['POST'])
def image_query(video_dir):
    output_dir = os.path.join(app.config['BASE_OUTPUT_DIR'], video_dir)
    json_path = os.path.join(output_dir, 'frame_text_repo.jsonl')
    paired_data = load_json_data(json_path)
    if not paired_data:
        return jsonify({'error': 'No data found', 'internet_search': False}), 404
    
    timestamp = request.json.get('timestamp')
    query = request.json.get('query', '')
    if timestamp is None or not query:
        return jsonify({'error': 'Timestamp or query not provided', 'internet_search': False}), 400
    
    closest_frame = find_closest_frame(paired_data, timestamp)
    if closest_frame:
        screenshot_path = os.path.join(output_dir, closest_frame.get('screenshot', '')) if closest_frame.get('screenshot') else None
        frame_context = closest_frame['combined_text']
        response = generate_frame_response(query, frame_context, screenshot_path)
        logger.info(f"Image query response for '{query}' at {timestamp}s: {response}")
        return jsonify({'response': response, 'timestamp': closest_frame['timestamp'], 'internet_search': False})
    return jsonify({'error': 'No frame found', 'internet_search': False}), 404

@app.route('/videos/<video_dir>/<path:filename>')
def serve_file(video_dir, filename):
    output_dir = os.path.join(app.config['BASE_OUTPUT_DIR'], video_dir)
    try:
        return send_from_directory(output_dir, filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        return "File not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)