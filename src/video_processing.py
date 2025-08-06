import os
import cv2
import json
import subprocess
import tempfile
from pathlib import Path
import whisper
import ollama
import time
import numpy as np
import gc

# Configuration
CHUNK_DURATION = 5
FRAMES_PER_CHUNK = 5
MODEL = "gemma3n:e4b-it-q8_0"
PROMPT = "Analyze these frames from a video chunk and describe what is happening over time."

# Common Functions
def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def extract_and_transcribe_audio(video_path, output_dir):
    audio_path = os.path.join(output_dir, "audio.wav")
    cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        log(f"Error extracting audio: {e}")
        return []
    
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, verbose=False)
        os.remove(audio_path)
        return result["segments"]
    except Exception as e:
        log(f"Error transcribing audio: {e}")
        return []

def get_audio_text_for_chunk(audio_segments, start_time, end_time):
    relevant_segments = [seg for seg in audio_segments if seg["start"] >= start_time and seg["end"] <= end_time]
    return " ".join([seg["text"].strip() for seg in relevant_segments]) or "No audio available"

# Functions from First Code
def extract_frames_to_temp(video_path, start_time, end_time, num_frames, resize=(640, 480)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Unable to open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int((end_time - start_time) * fps)
    step = max(1, total_frames // num_frames)
    
    temp_files = []
    for i in range(num_frames):
        frame_num = int(start_time * fps + i * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, image = cap.read()
        if success:
            image = cv2.resize(image, resize)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, image)
                temp_files.append(tmp.name)
    cap.release()
    return temp_files

def process_video(video_path, output_dir, audio_segments):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    chunks = [(i, min(i + CHUNK_DURATION, duration)) for i in range(0, int(duration), CHUNK_DURATION)]
    log(f"Processing {len(chunks)} chunks")
    
    chunk_results = []
    for start, end in chunks:
        temp_files = extract_frames_to_temp(video_path, start, end, FRAMES_PER_CHUNK)
        if not temp_files:
            chunk_results.append({"start_time": start, "end_time": end, "visual_text": "No frames extracted"})
            continue
        
        prompt = PROMPT
        try:
            response = ollama.generate(model=MODEL, prompt=prompt, images=temp_files, options={"num_predict": 200, "temperature": 0.7})
            visual_text = response['response'] if response else "No response"
        except Exception as e:
            log(f"Error generating response for chunk {start}-{end}: {e}")
            visual_text = "Error generating response"
        
        for f in temp_files:
            os.remove(f)
        
        chunk_results.append({
            "start_time": start,
            "end_time": end,
            "visual_text": visual_text
        })
    
    paired_data = []
    for chunk in chunk_results:
        audio_text = get_audio_text_for_chunk(audio_segments, chunk["start_time"], chunk["end_time"])
        combined_text = f"Audio: {audio_text}\nVisual Text: {chunk['visual_text']}"
        paired_data.append({
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "audio_text": audio_text,
            "visual_text": chunk['visual_text'],
            "combined_text": combined_text
        })
    
    output_jsonl = os.path.join(output_dir, "video_text_repo.jsonl")
    with open(output_jsonl, 'w') as f:
        for item in paired_data:
            f.write(json.dumps(item) + '\n')
    log(f"Saved results to {output_jsonl}")
    return paired_data

# Functions from Second Code
def add_timestamp_to_image(image_path, timestamp):
    img = cv2.imread(image_path)
    if img is None:
        return
    timestamp_text = f"{timestamp:.1f}s"
    cv2.putText(img, timestamp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(image_path, img)

def extract_frames(video_path, output_dir, interval, prefix="frame", quality="1"):
    screenshot_dir = os.path.join(output_dir, "screenshots")
    Path(screenshot_dir).mkdir(exist_ok=True)
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps=1/{interval}",
        "-q:v", quality,
        f"{screenshot_dir}/{prefix}_%04d.jpg"
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        log(f"Error extracting frames: {e}")
        return []
    
    frames = []
    for file in sorted(os.listdir(screenshot_dir)):
        if file.startswith(prefix) and file.endswith(".jpg"):
            frame_num = int(file.split("_")[1].split(".")[0])
            timestamp = (frame_num - 1) * interval
            frame_path = os.path.join(screenshot_dir, file)
            add_timestamp_to_image(frame_path, timestamp)
            rel_path = os.path.relpath(frame_path, output_dir)
            frames.append({"timestamp": timestamp, "path": rel_path})
    return frames

def compute_mse(imageA, imageB):
    if imageA is None or imageB is None:
        return float('inf')
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageB.shape[1])
    return err

def detect_changes(frames, threshold, output_dir):
    changes = []
    if not frames:
        return changes
    prev_frame = cv2.imread(os.path.join(output_dir, frames[0]["path"]), cv2.IMREAD_GRAYSCALE)
    for frame in frames[1:]:
        current_frame = cv2.imread(os.path.join(output_dir, frame["path"]), cv2.IMREAD_GRAYSCALE)
        mse = compute_mse(prev_frame, current_frame)
        if mse > threshold:
            changes.append((frames[frames.index(frame) - 1], frame))
        prev_frame = current_frame
    return changes

def extract_refined_frames(video_path, output_dir, start_time, end_time, interval, change_id, quality="1"):
    refined_dir = os.path.join(output_dir, "refined_screenshots")
    Path(refined_dir).mkdir(exist_ok=True)
    
    prefix = f"refined_change{change_id}"
    cmd = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(end_time - start_time),
        "-vf", f"fps=1/{interval}",
        "-q:v", quality,
        f"{refined_dir}/{prefix}_%04d.jpg"
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        log(f"Error extracting refined frames: {e}")
        return []
    
    refined_frames = []
    for file in sorted(os.listdir(refined_dir)):
        if file.startswith(prefix) and file.endswith(".jpg"):
            frame_num = int(file.split("_")[-1].split(".")[0])
            timestamp = start_time + (frame_num - 1) * interval
            frame_path = os.path.join(refined_dir, file)
            add_timestamp_to_image(frame_path, timestamp)
            rel_path = os.path.relpath(frame_path, output_dir)
            refined_frames.append({"timestamp": timestamp, "path": rel_path})
    return refined_frames

def select_stable_frames(frames, output_dir, threshold=500):
    if not frames:
        return []
    
    clusters = []
    current_cluster = [frames[0]]
    prev_frame = cv2.imread(os.path.join(output_dir, frames[0]["path"]), cv2.IMREAD_GRAYSCALE)
    
    for frame in frames[1:]:
        current_frame = cv2.imread(os.path.join(output_dir, frame["path"]), cv2.IMREAD_GRAYSCALE)
        mse = compute_mse(prev_frame, current_frame)
        if mse < threshold:
            current_cluster.append(frame)
        else:
            clusters.append(current_cluster[0])
            current_cluster = [frame]
        prev_frame = current_frame
    
    if current_cluster:
        clusters.append(current_cluster[0])
    return clusters

def load_chunk_jsonl(jsonl_path):
    chunks = []
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                chunks.append(json.loads(line.strip()))
    except FileNotFoundError:
        log(f"Chunk JSONL file not found: {jsonl_path}")
        return []
    return chunks

def find_chunk_context(timestamp, chunks):
    for chunk in chunks:
        if chunk["start_time"] <= timestamp < chunk["end_time"]:
            return chunk["combined_text"]
    return "No chunk context available"

def generate_image_transcription(image_path, prompt, context=""):
    full_prompt = f"This frame is part of a video. The sequence of events in this snippet is: {context}\n\n{prompt}"
    try:
        response = ollama.generate(model=MODEL, prompt=full_prompt, images=[image_path], options={"num_predict": 200, "temperature": 0.7})
        return response['response'] if response else "No transcription generated"
    except Exception as e:
        log(f"Error generating transcription: {e}")
        return "Error generating transcription"

def sync_data_with_text(frames, audio_segments, output_dir, image_prompt, batch_size, chunks):
    paired_data = []
    total_batches = (len(frames) + batch_size - 1) // batch_size
    log(f"Processing {len(frames)} frames in {total_batches} batches")
    
    for batch_num in range(total_batches):
        log(f"Processing batch {batch_num + 1}/{total_batches}")
        batch_frames = frames[batch_num * batch_size:(batch_num + 1) * batch_size]
        for frame in batch_frames:
            timestamp = frame["timestamp"]
            context = find_chunk_context(timestamp, chunks)
            transcription = generate_image_transcription(os.path.join(output_dir, frame["path"]), image_prompt, context)
            audio_text = "No audio available"
            if audio_segments:
                closest_segment = min(audio_segments, key=lambda x: abs(x["start"] - timestamp))
                if abs(closest_segment["start"] - timestamp) <= 5:
                    audio_text = closest_segment["text"].strip()
            combined_text = f"Audio: {audio_text}\nVisual Text: {transcription}"
            paired_data.append({
                "screenshot": frame["path"],
                "timestamp": timestamp,
                "audio_text": audio_text,
                "visual_text": transcription,
                "combined_text": combined_text
            })
        gc.collect()
    log(f"Generated {len(paired_data)} entries for JSONL")
    return paired_data

def save_to_jsonl(paired_data, output_path):
    with open(output_path, 'w') as f:
        for item in paired_data:
            f.write(json.dumps(item) + '\n')
    log(f"Data saved to {output_path}")

# Main Execution
if __name__ == "__main__":
    video_path = input("Enter video file path (or 'quit' to exit): ").strip()
    if video_path.lower() == 'quit':
        print("Cancelled.")
        exit(0)
    output_dir = input("Enter output directory path (or 'quit' to exit): ").strip()
    if output_dir.lower() == 'quit':
        print("Cancelled.")
        exit(0)
    
    video_path = os.path.expanduser(video_path)
    output_dir = os.path.expanduser(output_dir)
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        exit(1)
    if not os.path.isdir(output_dir):
        print(f"Output directory not found: {output_dir}")
        exit(1)
    
    log("Transcribing audio...")
    audio_segments = extract_and_transcribe_audio(video_path, output_dir)
    
    log("Processing video into chunks...")
    chunk_paired_data = process_video(video_path, output_dir, audio_segments)
    chunk_jsonl = os.path.join(output_dir, "video_text_repo.jsonl")
    
    log("Loading chunk-level data...")
    chunks = load_chunk_jsonl(chunk_jsonl)
    
    print("Please choose the processing mode:")
    print("1. Casual Mode - Faster processing with lower resolution.")
    print("2. Pro Mode - Detailed processing with high resolution.")
    mode_choice = input("Enter 1 for Casual or 2 for Pro: ").strip()
    
    if mode_choice == "1":
        FFMPEG_QUALITY = "2"
        INITIAL_INTERVAL = 10
        REFINED_INTERVAL = 2
        CHANGE_THRESHOLD = 1000
        IMAGE_PROMPT = "Transcribe all visible text in the image exactly as it appears. Describe the scene briefly."
        batch_size = 20
        log("Selected Mode: Casual Mode")
        log(f"Batch size {batch_size}")
    
    elif mode_choice == "2":
        FFMPEG_QUALITY = "1"
        INITIAL_INTERVAL = 5
        REFINED_INTERVAL = 1
        CHANGE_THRESHOLD = 500
        IMAGE_PROMPT = "Transcribe all visible text in the image exactly as it appears, including any text on buttons, UI elements, or other visual components. Describe the scene in detail."
        batch_size = 1
        log("Selected Mode: Pro Mode")
        log("Batch size 1")
    
    else:
        FFMPEG_QUALITY = "2"
        INITIAL_INTERVAL = 10
        REFINED_INTERVAL = 2
        CHANGE_THRESHOLD = 1000
        IMAGE_PROMPT = "Transcribe all visible text in the image exactly as it appears. Describe the scene briefly."
        batch_size = 20
        log("Invalid mode, defaulting to Casual Mode")
        log(f"Batch size {batch_size}")
    
    log("Extracting initial frames...")
    initial_frames = extract_frames(video_path, output_dir, INITIAL_INTERVAL, prefix="initial", quality=FFMPEG_QUALITY)
    
    log("Detecting changes...")
    change_points = detect_changes(initial_frames, CHANGE_THRESHOLD, output_dir)
    
    log("Extracting refined frames...")
    refined_frames = []
    for i, (prev_frame, current_frame) in enumerate(change_points):
        frames = extract_refined_frames(video_path, output_dir, prev_frame["timestamp"], current_frame["timestamp"], REFINED_INTERVAL, i, quality=FFMPEG_QUALITY)
        refined_frames.extend(frames)
    refined_frames = sorted(refined_frames, key=lambda x: x["timestamp"])
    
    log("Selecting stable frames...")
    selected_frames = select_stable_frames(refined_frames, output_dir)
    if not selected_frames:
        selected_frames = initial_frames
    
    log("Generating transcriptions with chunk context...")
    paired_data = sync_data_with_text(selected_frames, audio_segments, output_dir, IMAGE_PROMPT, batch_size, chunks)
    
    output_jsonl = os.path.join(output_dir, "frame_text_repo.jsonl")
    save_to_jsonl(paired_data, output_jsonl)