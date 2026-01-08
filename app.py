import os
import subprocess
import threading
import shutil
import time
import re
from flask import Flask, render_template, request, jsonify, send_from_directory
from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, TIT2, TPE1, APIC, COMM
from pydub import AudioSegment
import librosa
import numpy as np
import scipy.io.wavfile as wavfile

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
PROCESSED_FOLDER = 'processed'
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
    # Remove extension for processing
    name, ext = os.path.splitext(filename)
    
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    
    # Remove trailing ID (digits at end)
    name = re.sub(r'-\d+$', '', name)
    
    # Remove "Radio Edit" if desired? (Keeping it based on example)
    
    # Fix dot spacing (ft.Travis -> ft. Travis)
    name = re.sub(r'\.(?=[A-Z])', '. ', name)
    
    # Fix double spaces
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name, ext

def update_metadata(filepath, artist, title):
    """
    Clears metadata and sets only specific fields.
    """
    try:
        # Load audio to get duration
        audio = MP3(filepath, ID3=EasyID3)
        duration = audio.info.length
        
        # Clear existing tags
        audio.delete()
        audio.save()
        
        # Create new ID3 tag
        tags = ID3(filepath)
        
        # Set Title and Artist
        tags.add(TIT2(encoding=3, text=title))
        tags.add(TPE1(encoding=3, text=artist))
        
        # Add ID By Rivoli URL as comment or custom tag
        tags.add(COMM(encoding=3, lang='eng', desc='ID By Rivoli', text='https://idbyrivoli.com'))
        
        # Note: BPM and Key detection would need analysis (librosa)
        # Cover art injection would need a source image
        
        tags.save()
    except Exception as e:
        print(f"Error updating metadata for {filepath}: {e}")

def create_edits(vocals_path, inst_path, original_path, base_output_path, base_filename):
    """
    Generates all the requested edits.
    This is a complex function that requires beat grid analysis.
    For this MVP, we will use simplified logic based on duration/segments.
    Real implementation needs beat detection.
    """
    
    # Load audio segments
    print(f"Loading audio for edits: {base_filename}")
    vocals = AudioSegment.from_mp3(vocals_path)
    inst = AudioSegment.from_mp3(inst_path)
    original = AudioSegment.from_mp3(original_path)
    
    # 1 bar at 128 BPM is approx 1.875 seconds. 
    # 32 beats = 8 bars = ~15 seconds.
    # We need to detect BPM to be accurate.
    
    # Use librosa for BPM detection on the original track
    y, sr = librosa.load(original_path, duration=60) # Analyze first 60s
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Handle scalar or array return from beat_track (librosa versions vary)
    # beat_track can return a scalar or an array depending on version/inputs
    if hasattr(tempo, 'item'):
         bpm = round(tempo.item())
    elif isinstance(tempo, np.ndarray):
        bpm = round(float(tempo[0])) if tempo.size > 0 else 120
    else:
        bpm = round(tempo)
        
    print(f"Detected BPM: {bpm}")
    
    # Calculate duration of 1 beat in ms
    beat_ms = (60 / bpm) * 1000
    ms_16_beats = 16 * beat_ms
    ms_32_beats = 32 * beat_ms
    
    edits = []

    # Helper to export
    def export_edit(audio_segment, suffix):
        # Clean filename for output
        clean_name, _ = clean_filename(base_filename)
        # Ensure exact format: "Title Suffix.mp3"
        out_name_mp3 = f"{clean_name} {suffix}.mp3"
        out_name_wav = f"{clean_name} {suffix}.wav"
        
        out_path_mp3 = os.path.join(base_output_path, out_name_mp3)
        out_path_wav = os.path.join(base_output_path, out_name_wav)
        
        # Export MP3
        audio_segment.export(out_path_mp3, format="mp3", bitrate="320k")
        # Export WAV
        audio_segment.export(out_path_wav, format="wav")
        
        # Metadata
        update_metadata(out_path_mp3, "ID By Rivoli", f"{clean_name} {suffix}")
        
        return {
            'name': f"{clean_name} {suffix}",
            'mp3': f"/download_processed/{out_name_mp3}",
            'wav': f"/download_processed/{out_name_wav}"
        }

    # 1. Super Short (Placeholder: Intro + Drop)
    # Logic: Intro (16 beats) + Drop (32 beats)
    super_short = original[:ms_16_beats] + original[ms_32_beats:ms_32_beats*3]
    edits.append(export_edit(super_short, "Super Short"))

    # 2. Short (Intro Inst + 1 Drop)
    # Logic: 32 beats Inst + Drop
    short = inst[:ms_32_beats] + original[ms_32_beats:ms_32_beats*4]
    edits.append(export_edit(short, "Short"))

    # 3. Acap Out (Original + 32 beats Vocals at end)
    # Logic: Full track + Last 32 beats of vocals (simulated)
    # Finding the "drop vocals" is hard without structural analysis.
    # We'll take the last 32 beats of the vocal track.
    acap_out_segment = vocals[-ms_32_beats:]
    acap_out = original + acap_out_segment
    edits.append(export_edit(acap_out, "Acap Out"))

    # 4. Acap In (32 beats Vocals + Drop)
    acap_in = vocals[:ms_32_beats] + original[ms_32_beats:]
    edits.append(export_edit(acap_in, "Acap In"))

    # 5. Slam Intro Short Acap Out (FX + Start + Short + Acap Out)
    # Needs an FX sample. For now, silence or simple cut.
    slam_intro = original[0:500] # Just a blip for now
    slam_edit = slam_intro + short + acap_out_segment
    edits.append(export_edit(slam_edit, "Slam Intro Short Acap Out"))

    # 6. Clap In Short Acap Out (16 beats Claps + Short + Acap Out)
    # Needs Clap sample. We'll simulate with silence for MVP or reuse a beat if we can isolate drums (Demucs 4 stems needed).
    # Since we only have 2 stems (Vocals/Inst), we can't easily get just "Claps".
    # We'll skip adding external samples for now and just structure it.
    clap_in_edit = inst[:ms_16_beats] + short + acap_out_segment # Using inst as placeholder for claps
    edits.append(export_edit(clap_in_edit, "Clap In Short Acap Out"))

    # 7. Short Acap Out
    short_acap_out = short + acap_out_segment
    edits.append(export_edit(short_acap_out, "Short Acap Out"))

    # 8. Slam Dirty Main (Slam Intro + Full Track)
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
    # Same as Short? "Short Main" usually implies Main version but shortened.
    edits.append(export_edit(short, "Short Main"))

    # 13. Clap In Main
    clap_in_main = inst[:ms_16_beats] + original
    edits.append(export_edit(clap_in_main, "Clap In Main"))

    # 14. Original Track Mp3Main (Clean name)
    edits.append(export_edit(original, "Main"))

    # 15. Instrumental Only
    edits.append(export_edit(inst, "Instrumental"))
    
    # 16. Vocals Only
    edits.append(export_edit(vocals, "Acapella"))

    return edits

def run_demucs_thread(filepaths, original_filenames):
    global job_status
    try:
        job_status['state'] = 'processing'
        job_status['total_files'] = len(filepaths)
        job_status['results'] = []
        job_status['progress'] = 0

        # Construct command with ALL files
        # We need 4 stems now to potentially get drums for "Claps"? 
        # User asked for "Clap In", usually requires external sample or Drums stem.
        # But for MVP let's stick to 2 stems (Vocals/Inst) to keep it simpler/faster, 
        # unless we switch to 4 stems (vocals, drums, bass, other).
        # Switching to 4 stems increases processing time but allows better edits.
        # Let's stick to 2 stems for now as requested initially.
        
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
        
        # Use Popen to read stdout/stderr
        # We merge stderr into stdout to capture Demucs progress bars and logs in one stream
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            universal_newlines=True
        )

        current_file_index = 0
        
        # Read output line by line
        for line in process.stdout:
            print(line, end='') # Log to console
            
            # Detect new file start
            if "Separating track" in line:
                current_file_index += 1
                job_status['current_file_idx'] = current_file_index
                job_status['current_filename'] = os.path.basename(filepaths[current_file_index-1])
                # Calculate base progress (start of this file's chunk)
                # Phase 1 (Separation) is 0% to 50% total
                chunk_size = 50 / len(filepaths)
                job_status['progress'] = int((current_file_index - 1) * chunk_size)

            # Try to parse tqdm progress bar (e.g. " 24%|...|")
            # Demucs progress looks like: " 24%|████| 28/284 [00:03<...]"
            elif "%|" in line:
                try:
                    # Extract number before %
                    parts = line.split('%|')
                    if len(parts) > 0:
                        percent_part = parts[0].strip() # "24"
                        # Handle potential garbage before number
                        percent_val = int(re.search(r'(\d+)$', percent_part).group(1))
                        
                        # Add this file's progress to the global progress
                        chunk_size = 50 / len(filepaths)
                        current_file_base = (current_file_index - 1) * chunk_size
                        
                        # Add percentage of the chunk
                        added_val = (percent_val / 100) * chunk_size
                        job_status['progress'] = int(current_file_base + added_val)
                except:
                    pass
        
        process.wait()
        return_code = process.returncode
        
        if return_code != 0:
            job_status['state'] = 'error'
            job_status['error'] = 'Erreur lors du traitement Demucs'
            return

        # Phase 2: Create Edits
        print("Starting Edit Generation Phase...")
        job_status['progress'] = 50
        
        all_results = []
        
        for i, filepath in enumerate(filepaths):
            filename = os.path.basename(filepath)
            original_name = os.path.splitext(filename)[0]
            track_name = original_name 
            
            source_dir = os.path.join(OUTPUT_FOLDER, 'htdemucs', track_name)
            
            # Demucs outputs
            inst_path = os.path.join(source_dir, 'no_vocals.mp3')
            vocals_path = os.path.join(source_dir, 'vocals.mp3')
            
            if os.path.exists(inst_path) and os.path.exists(vocals_path):
                # Create a folder for this track's edits in processed
                clean_name, _ = clean_filename(filename)
                track_output_dir = os.path.join(PROCESSED_FOLDER, clean_name)
                os.makedirs(track_output_dir, exist_ok=True)
                
                # Generate Edits
                edits = create_edits(vocals_path, inst_path, filepath, track_output_dir, filename)
                
                all_results.append({
                    'original': clean_name,
                    'edits': edits
                })
                
                # Update progress
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
    
    # Reset job status
    job_status = {
        'state': 'starting',
        'progress': 0,
        'total_files': len(files),
        'current_file_idx': 0,
        'current_filename': '',
        'results': [],
        'error': None
    }
    
    # Save all files
    for file in files:
        if file.filename:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            saved_filepaths.append(filepath)
            original_filenames.append(filename)
    
    # Start processing
    thread = threading.Thread(target=run_demucs_thread, args=(saved_filepaths, original_filenames))
    thread.start()
    
    return jsonify({'message': 'Traitement démarré', 'total_files': len(files)})

@app.route('/status')
def status():
    return jsonify(job_status)

@app.route('/download_processed/<path:filename>')
def download_processed(filename):
    # Ensure correct MIME types are sent so browser downloads as audio, not HTML/Text
    # Force attachment to ensure download prompt
    return send_from_directory(
        PROCESSED_FOLDER, 
        filename, 
        as_attachment=True,
        mimetype='audio/mpeg' if filename.lower().endswith('.mp3') else 'audio/wav'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
