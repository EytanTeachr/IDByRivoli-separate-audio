import os
import subprocess
import threading
import shutil
import time
import re
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, TIT2, TPE1, APIC, COMM
from pydub import AudioSegment
import librosa
import numpy as np
import scipy.io.wavfile as wavfile
import urllib.parse

app = Flask(__name__)

# Use absolute paths to avoid confusion
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to track progress
job_status = {
    'state': 'idle', 
    'progress': 0,
    'total_files': 0,
    'current_file_idx': 0,
    'current_filename': '',
    'results': [],
    'error': None
}

def clean_filename(filename):
    """
    Cleans filename: removes underscores, specific patterns, and unnecessary IDs.
    Example: DJ_Mustard_ft.Travis_Scott-Whole_Lotta_Lovin_Edits_and_Intro_Outros-Radio_Edit-77055446
    Result: DJ Mustard ft. Travis Scott - Whole Lotta Lovin Edits and Intro Outros
    """
    name, ext = os.path.splitext(filename)
    name = name.replace('_', ' ')
    name = re.sub(r'-\d+$', '', name)
    name = re.sub(r'\.(?=[A-Z])', '. ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name, ext

def update_metadata(filepath, artist, title):
    try:
        try:
            audio = MP3(filepath, ID3=EasyID3)
            audio.delete()
            audio.save()
        except:
            pass
        
        tags = ID3(filepath)
        tags.add(TIT2(encoding=3, text=title))
        tags.add(TPE1(encoding=3, text=artist))
        tags.add(COMM(encoding=3, lang='eng', desc='ID By Rivoli', text='https://idbyrivoli.com'))
        tags.save()
    except Exception as e:
        print(f"Error updating metadata for {filepath}: {e}")

def create_edits(vocals_path, inst_path, original_path, base_output_path, base_filename):
    print(f"Loading audio for edits: {base_filename}")
    vocals = AudioSegment.from_mp3(vocals_path)
    inst = AudioSegment.from_mp3(inst_path)
    original = AudioSegment.from_mp3(original_path)
    
    # Use librosa for BPM detection on the original track
    y, sr = librosa.load(original_path, duration=60)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    if hasattr(tempo, 'item'):
         bpm = round(tempo.item())
    elif isinstance(tempo, np.ndarray):
        bpm = round(float(tempo[0])) if tempo.size > 0 else 120
    else:
        bpm = round(tempo)
        
    print(f"Detected BPM: {bpm}")
    
    beat_ms = (60 / bpm) * 1000
    ms_16_beats = 16 * beat_ms
    ms_32_beats = 32 * beat_ms
    
    edits = []

    def export_edit(audio_segment, suffix):
        clean_name, _ = clean_filename(base_filename)
        out_name_mp3 = f"{clean_name} {suffix}.mp3"
        out_name_wav = f"{clean_name} {suffix}.wav"
        
        out_path_mp3 = os.path.join(base_output_path, out_name_mp3)
        out_path_wav = os.path.join(base_output_path, out_name_wav)
        
        audio_segment.export(out_path_mp3, format="mp3", bitrate="320k")
        audio_segment.export(out_path_wav, format="wav")
        
        update_metadata(out_path_mp3, "ID By Rivoli", f"{clean_name} {suffix}")
        
        # Determine the relative path for URL (subdirectory + filename)
        # We need to URL encode the components to handle spaces safely
        subdir = clean_name
        
        # Return URL-safe paths
        # Note: We will decode these in the Flask route
        return {
            'name': f"{clean_name} {suffix}",
            'mp3': f"/download_processed/{urllib.parse.quote(subdir)}/{urllib.parse.quote(out_name_mp3)}",
            'wav': f"/download_processed/{urllib.parse.quote(subdir)}/{urllib.parse.quote(out_name_wav)}"
        }

    # 1. Super Short
    super_short = original[:ms_16_beats] + original[ms_32_beats:ms_32_beats*3]
    edits.append(export_edit(super_short, "Super Short"))

    # 2. Short
    short = inst[:ms_32_beats] + original[ms_32_beats:ms_32_beats*4]
    edits.append(export_edit(short, "Short"))

    # 3. Acap Out
    acap_out_segment = vocals[-ms_32_beats:]
    acap_out = original + acap_out_segment
    edits.append(export_edit(acap_out, "Acap Out"))

    # 4. Acap In
    acap_in = vocals[:ms_32_beats] + original[ms_32_beats:]
    edits.append(export_edit(acap_in, "Acap In"))

    # 5. Slam Intro Short Acap Out
    slam_intro = original[0:500] 
    slam_edit = slam_intro + short + acap_out_segment
    edits.append(export_edit(slam_edit, "Slam Intro Short Acap Out"))

    # 6. Clap In Short Acap Out
    clap_in_edit = inst[:ms_16_beats] + short + acap_out_segment 
    edits.append(export_edit(clap_in_edit, "Clap In Short Acap Out"))

    # 7. Short Acap Out
    short_acap_out = short + acap_out_segment
    edits.append(export_edit(short_acap_out, "Short Acap Out"))

    # 8. Slam Dirty Main
    slam_main = slam_intro + original
    edits.append(export_edit(slam_main, "Slam Dirty Main"))

    # 9. Acap In Acap Out Main
    acap_in_out_main = vocals[:ms_32_beats] + original + vocals[-ms_32_beats:]
    edits.append(export_edit(acap_in_out_main, "Acap In Acap Out Main"))

    # 10. Short Clap In
    short_clap_in = inst[:ms_16_beats] + short
    edits.append(export_edit(short_clap_in, "Short Clap In"))

    # 11. Short Acap In
    short_acap_in = vocals[:ms_32_beats] + short
    edits.append(export_edit(short_acap_in, "Short Acap In"))

    # 12. Short Main
    edits.append(export_edit(short, "Short Main"))

    # 13. Clap In Main
    clap_in_main = inst[:ms_16_beats] + original
    edits.append(export_edit(clap_in_main, "Clap In Main"))

    # 14. Main
    edits.append(export_edit(original, "Main"))

    # 15. Instrumental
    edits.append(export_edit(inst, "Instrumental"))
    
    # 16. Acapella
    edits.append(export_edit(vocals, "Acapella"))

    return edits

def run_demucs_thread(filepaths, original_filenames):
    global job_status
    try:
        job_status['state'] = 'processing'
        job_status['total_files'] = len(filepaths)
        job_status['results'] = []
        job_status['progress'] = 0

        command = [
            'python3', '-m', 'demucs',
            '--two-stems=vocals',
            '-n', 'htdemucs',
            '--mp3',
            '--mp3-bitrate', '320',
            '-j', '4', 
            '-o', OUTPUT_FOLDER
        ] + filepaths

        print(f"Starting batch processing of {len(filepaths)} files...")
        
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            universal_newlines=True
        )

        current_file_index = 0
        
        for line in process.stdout:
            print(line, end='')
            
            if "Separating track" in line:
                current_file_index += 1
                job_status['current_file_idx'] = current_file_index
                job_status['current_filename'] = os.path.basename(filepaths[current_file_index-1])
                chunk_size = 50 / len(filepaths)
                job_status['progress'] = int((current_file_index - 1) * chunk_size)

            elif "%|" in line:
                try:
                    parts = line.split('%|')
                    if len(parts) > 0:
                        percent_part = parts[0].strip()
                        percent_val = int(re.search(r'(\d+)$', percent_part).group(1))
                        chunk_size = 50 / len(filepaths)
                        current_file_base = (current_file_index - 1) * chunk_size
                        added_val = (percent_val / 100) * chunk_size
                        job_status['progress'] = int(current_file_base + added_val)
                except:
                    pass
        
        process.wait()
        
        if process.returncode != 0:
            job_status['state'] = 'error'
            job_status['error'] = 'Erreur lors du traitement Demucs'
            return

        print("Starting Edit Generation Phase...")
        job_status['progress'] = 50
        
        all_results = []
        
        for i, filepath in enumerate(filepaths):
            filename = os.path.basename(filepath)
            track_name = os.path.splitext(filename)[0]
            
            source_dir = os.path.join(OUTPUT_FOLDER, 'htdemucs', track_name)
            inst_path = os.path.join(source_dir, 'no_vocals.mp3')
            vocals_path = os.path.join(source_dir, 'vocals.mp3')
            
            if os.path.exists(inst_path) and os.path.exists(vocals_path):
                clean_name, _ = clean_filename(filename)
                track_output_dir = os.path.join(PROCESSED_FOLDER, clean_name)
                os.makedirs(track_output_dir, exist_ok=True)
                
                edits = create_edits(vocals_path, inst_path, filepath, track_output_dir, filename)
                
                all_results.append({
                    'original': clean_name,
                    'edits': edits
                })
                
                job_status['progress'] = 50 + int((i + 1) / len(filepaths) * 50)
            else:
                print(f"Warning: Output files not found for {track_name}")

        job_status['progress'] = 100
        job_status['results'] = all_results
        job_status['state'] = 'completed'

    except Exception as e:
        print(f"Error in thread: {e}")
        job_status['state'] = 'error'
        job_status['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global job_status
    
    if job_status['state'] == 'processing':
        return jsonify({'error': 'Un traitement est déjà en cours. Veuillez patienter.'}), 409

    if 'files[]' not in request.files:
         return jsonify({'error': 'Aucun fichier envoyé'}), 400
    
    files = request.files.getlist('files[]')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    saved_filepaths = []
    original_filenames = []
    
    job_status = {
        'state': 'starting',
        'progress': 0,
        'total_files': len(files),
        'current_file_idx': 0,
        'current_filename': '',
        'results': [],
        'error': None
    }
    
    for file in files:
        if file.filename:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            saved_filepaths.append(filepath)
            original_filenames.append(filename)
    
    thread = threading.Thread(target=run_demucs_thread, args=(saved_filepaths, original_filenames))
    thread.start()
    
    return jsonify({'message': 'Traitement démarré', 'total_files': len(files)})

@app.route('/status')
def status():
    return jsonify(job_status)

@app.route('/download_processed/<path:subdir>/<path:filename>')
def download_processed(subdir, filename):
    """
    Downloads a file from a specific subdirectory in processed folder.
    """
    # Decoding is handled by Flask, but just to be sure we are safe
    # We construct the absolute path
    directory = os.path.join(PROCESSED_FOLDER, subdir)
    
    print(f"Download request: {subdir} / {filename}")
    print(f"Looking in: {directory}")
    
    if not os.path.exists(os.path.join(directory, filename)):
        print("File not found!")
        abort(404)
        
    return send_from_directory(
        directory, 
        filename, 
        as_attachment=True,
        mimetype='audio/mpeg' if filename.lower().endswith('.mp3') else 'audio/wav'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
