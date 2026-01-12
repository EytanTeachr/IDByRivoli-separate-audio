import os
import subprocess
import threading
import shutil
import time
import re
from flask import Flask, render_template, request, jsonify, send_from_directory, abort, send_file
from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, TIT2, TPE1, APIC, COMM, TALB, TDRC, TRCK, TCON, TBPM, TSRC, TLEN, TPUB, TMED, WOAR, WXXX
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

import shutil
import zipfile
import io

@app.route('/download_all_zip')
def download_all_zip():
    """
    Creates a ZIP file containing all processed tracks and sends it to the user.
    """
    global job_status
    
    if job_status['state'] != 'completed' or not job_status['results']:
        return jsonify({'error': 'Aucun traitement terminé à télécharger'}), 400

    # Create an in-memory ZIP file
    memory_file = io.BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for result in job_status['results']:
            track_name = result['original']
            track_dir = os.path.join(PROCESSED_FOLDER, track_name)
            
            if os.path.exists(track_dir):
                for root, dirs, files in os.walk(track_dir):
                    for file in files:
                        if file.lower().endswith(('.mp3', '.wav')): 
                            file_path = os.path.join(root, file)
                            arcname = os.path.join(track_name, file)
                            zf.write(file_path, arcname)

    memory_file.seek(0)
    
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='ID_By_Rivoli_Edits_Pack.zip'
    )

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

def update_metadata(filepath, artist, title, original_path, bpm):
    """
    Updates metadata with ONLY the specified fields (clean slate).
    Fields: Title, Artist, Album, Date, Track Number, Genre, BPM, ISRC, Picture, Length, Publisher
    """
    try:
        # Read original file metadata
        try:
            original_audio = MP3(original_path, ID3=ID3)
            original_tags = original_audio.tags
        except:
            original_tags = None
        
        # Clear all existing tags and start fresh
        try:
            audio = MP3(filepath, ID3=ID3)
            audio.delete()  # Remove all tags
            audio.save()
        except:
            pass
        
        # Create new clean ID3 tag
        tags = ID3(filepath)
        
        # Add ONLY specified fields
        
        # 1. Title (from parameter)
        tags.add(TIT2(encoding=3, text=title))
        
        # 2. Artist (from original)
        if original_tags and 'TPE1' in original_tags:
            tags.add(TPE1(encoding=3, text=original_tags['TPE1'].text))
        
        # 3. Album (from original)
        if original_tags and 'TALB' in original_tags:
            tags.add(TALB(encoding=3, text=original_tags['TALB'].text))
        
        # 4. Date (from original, preserve full format)
        if original_tags and 'TDRC' in original_tags:
            tags.add(TDRC(encoding=3, text=original_tags['TDRC'].text))
        
        # 5. Track Number (from original)
        if original_tags and 'TRCK' in original_tags:
            tags.add(TRCK(encoding=3, text=original_tags['TRCK'].text))
        
        # 6. Genre (from original)
        if original_tags and 'TCON' in original_tags:
            tags.add(TCON(encoding=3, text=original_tags['TCON'].text))
        
        # 7. BPM (calculated)
        tags.add(TBPM(encoding=3, text=str(bpm)))
        
        # 8. ISRC (from original)
        if original_tags and 'TSRC' in original_tags:
            tags.add(TSRC(encoding=3, text=original_tags['TSRC'].text))
        
        # 9. Publisher
        tags.add(TPUB(encoding=3, text='ID By Rivoli'))
        
        # 10. Length
        try:
            audio_info = MP3(filepath)
            length_ms = int(audio_info.info.length * 1000)
            tags.add(TLEN(encoding=3, text=str(length_ms)))
        except:
            pass
        
        # 11. Picture - ID By Rivoli Cover as PRIMARY
        cover_path = os.path.join(BASE_DIR, 'assets', 'Cover_Id_by_Rivoli.jpeg')
        if os.path.exists(cover_path):
            with open(cover_path, 'rb') as img:
                tags.add(APIC(
                    encoding=3,
                    mime='image/jpeg',
                    type=3,  # Cover (front)
                    desc='ID By Rivoli',
                    data=img.read()
                ))
        
        # 12. Picture - Original cover as secondary if exists
        if original_tags:
            for apic_key in original_tags.keys():
                if apic_key.startswith('APIC:') and 'ID By Rivoli' not in str(apic_key):
                    try:
                        original_apic = original_tags[apic_key]
                        tags.add(APIC(
                            encoding=original_apic.encoding,
                            mime=original_apic.mime,
                            type=0,  # Other
                            desc='Original',
                            data=original_apic.data
                        ))
                        break
                    except:
                        pass
        
        # Additional fields for ID By Rivoli branding (optional, can be removed if not desired)
        tags.add(TMED(encoding=3, text='ID By Rivoli'))
        tags.add(COMM(encoding=3, lang='eng', desc='Description', text='ID By Rivoli - www.idbyrivoli.com'))
        tags.add(WXXX(encoding=3, desc='ID By Rivoli', url='https://www.idbyrivoli.com'))
        
        tags.save(filepath, v2_version=3)
        
    except Exception as e:
        print(f"Error updating metadata for {filepath}: {e}")

def update_metadata_wav(filepath, artist, title, original_path, bpm):
    """
    Adds ID3v2 tags to WAV file (non-standard but widely supported).
    Primary goal: Add artwork (cover) to WAV files.
    """
    try:
        # Read original file metadata for reference
        try:
            original_audio = MP3(original_path, ID3=ID3)
            original_tags = original_audio.tags
        except:
            original_tags = None
        
        # Create ID3 tags for WAV file
        # WAV files don't natively support ID3, but we can add them
        tags = ID3()
        
        # Add core fields
        tags.add(TIT2(encoding=3, text=title))
        
        if original_tags and 'TPE1' in original_tags:
            tags.add(TPE1(encoding=3, text=original_tags['TPE1'].text))
        
        tags.add(TBPM(encoding=3, text=str(bpm)))
        tags.add(TPUB(encoding=3, text='ID By Rivoli'))
        
        # Add ID By Rivoli Cover
        cover_path = os.path.join(BASE_DIR, 'assets', 'Cover_Id_by_Rivoli.jpeg')
        if os.path.exists(cover_path):
            with open(cover_path, 'rb') as img:
                tags.add(APIC(
                    encoding=3,
                    mime='image/jpeg',
                    type=3,
                    desc='ID By Rivoli',
                    data=img.read()
                ))
        
        # Add original cover as secondary if exists
        if original_tags:
            for apic_key in original_tags.keys():
                if apic_key.startswith('APIC:') and 'ID By Rivoli' not in str(apic_key):
                    try:
                        original_apic = original_tags[apic_key]
                        tags.add(APIC(
                            encoding=original_apic.encoding,
                            mime=original_apic.mime,
                            type=0,
                            desc='Original',
                            data=original_apic.data
                        ))
                        break
                    except:
                        pass
        
        # Save ID3 tags to WAV file
        tags.save(filepath, v2_version=3)
        
    except Exception as e:
        print(f"Warning: Could not add ID3 tags to WAV: {e}")

import requests
from datetime import datetime

# API Endpoint Configuration
API_ENDPOINT = os.environ.get('API_ENDPOINT', 'https://3cd10fc1dff8.ngrok.app')
API_KEY = os.environ.get('API_KEY', '5X#JP5ifkSm?oE6@haMriYG$j!87BEfX@zg3CxcE')

def send_track_info_to_api(track_data):
    """
    Sends track information to external API endpoint with authentication.
    """
    if not API_ENDPOINT:
        print("⚠️  API_ENDPOINT not configured, skipping API call")
        return
    
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }
        
        print(f"\n{'='*60}")
        print(f"📤 ENVOI API → {API_ENDPOINT}")
        print(f"   Titre: {track_data.get('Titre', 'N/A')}")
        print(f"   Type: {track_data.get('Type', 'N/A')}")
        print(f"   Format: {track_data.get('Format', 'N/A')}")
        print(f"   BPM: {track_data.get('BPM', 'N/A')}")
        print(f"{'='*60}")
        
        response = requests.post(API_ENDPOINT, json=track_data, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ API SUCCESS: {track_data['Titre']} ({track_data['Format']})")
        else:
            print(f"❌ API ERROR {response.status_code}: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ API EXCEPTION: {e}")

def prepare_track_metadata(edit_info, original_path, bpm, base_url=""):
    """
    Prepares track metadata for API export.
    """
    try:
        # Read original metadata
        original_audio = MP3(original_path, ID3=ID3)
        original_tags = original_audio.tags if original_audio.tags else {}
        
        # Extract fields
        artist = str(original_tags.get('TPE1', 'Unknown')).strip() if 'TPE1' in original_tags else 'Unknown'
        album = str(original_tags.get('TALB', '')).strip() if 'TALB' in original_tags else ''
        genre = str(original_tags.get('TCON', '')).strip() if 'TCON' in original_tags else ''
        
        # Date handling
        date_str = str(original_tags.get('TDRC', '')).strip() if 'TDRC' in original_tags else ''
        try:
            # Try to parse date to timestamp
            if date_str:
                date_obj = datetime.strptime(date_str[:10], '%Y-%m-%d')
                date_sortie = int(date_obj.timestamp())
            else:
                date_sortie = 0
        except:
            date_sortie = 0
        
        # Publisher/Label
        label = str(original_tags.get('TPUB', 'ID By Rivoli')).strip() if 'TPUB' in original_tags else 'ID By Rivoli'
        
        # Prepare data structure
        track_data = {
            'Type': edit_info.get('type', ''),  # e.g., "Clap In", "Extended", etc.
            'Format': edit_info.get('format', 'MP3'),  # "MP3" or "WAV"
            'Titre': edit_info.get('name', ''),
            'Artiste': artist,
            'Fichiers': edit_info.get('url', ''),  # Download URL
            'Univers': '',  # To be filled if metadata available
            'Mood': '',  # To be filled if metadata available
            'Style': genre,
            'Album': album,
            'Label': 'ID By Rivoli',  # Always ID By Rivoli for processed tracks
            'Sous-label': label if label != 'ID By Rivoli' else '',  # Original label as sub-label
            'Date de sortie': date_sortie,
            'BPM': bpm,
            'Artiste original': artist,  # Same as Artiste for now
            'Url': f"{base_url}/static/covers/Cover_Id_by_Rivoli.jpeg"  # Cover URL
        }
        
        return track_data
        
    except Exception as e:
        print(f"Error preparing track metadata: {e}")
        return None

import audio_processor

def create_edits(vocals_path, inst_path, original_path, base_output_path, base_filename):
    print(f"Loading audio for edits: {base_filename}")
    
    # Detect BPM
    bpm = audio_processor.detect_bpm(original_path)
    print(f"Detected BPM: {bpm}")
    
    # Check genre to determine if we should generate full edits or just preserve original
    try:
        original_audio = MP3(original_path, ID3=ID3)
        original_tags = original_audio.tags
        genre = str(original_tags.get('TCON', '')).lower() if original_tags and 'TCON' in original_tags else ''
    except:
        genre = ''
    
    # Genres that should NOT get edits (just original MP3/WAV)
    simple_genres = ['house', 'electro house', 'dance']
    
    edits = []

    def export_edit(audio_segment, suffix):
        clean_name, _ = clean_filename(base_filename)
        out_name_mp3 = f"{clean_name} {suffix}.mp3"
        out_name_wav = f"{clean_name} {suffix}.wav"
        
        out_path_mp3 = os.path.join(base_output_path, out_name_mp3)
        out_path_wav = os.path.join(base_output_path, out_name_wav)
        
        audio_segment.export(out_path_mp3, format="mp3", bitrate="320k")
        audio_segment.export(out_path_wav, format="wav")
        
        # Update metadata for MP3
        update_metadata(out_path_mp3, "ID By Rivoli", f"{clean_name} {suffix}", original_path, bpm)
        
        # Update metadata for WAV (add ID3 tags to WAV - non-standard but supported by many players)
        try:
            update_metadata_wav(out_path_wav, "ID By Rivoli", f"{clean_name} {suffix}", original_path, bpm)
        except Exception as e:
            print(f"Warning: Could not add metadata to WAV: {e}")
        
        subdir = clean_name
        
        mp3_url = f"/download_processed/{urllib.parse.quote(subdir)}/{urllib.parse.quote(out_name_mp3)}"
        wav_url = f"/download_processed/{urllib.parse.quote(subdir)}/{urllib.parse.quote(out_name_wav)}"
        
        # Prepare and send track info to API (for MP3)
        track_info_mp3 = {
            'type': suffix,
            'format': 'MP3',
            'name': f"{clean_name} {suffix}",
            'url': mp3_url
        }
        track_data_mp3 = prepare_track_metadata(track_info_mp3, original_path, bpm)
        if track_data_mp3:
            send_track_info_to_api(track_data_mp3)
        
        # Prepare and send track info to API (for WAV)
        track_info_wav = {
            'type': suffix,
            'format': 'WAV',
            'name': f"{clean_name} {suffix}",
            'url': wav_url
        }
        track_data_wav = prepare_track_metadata(track_info_wav, original_path, bpm)
        if track_data_wav:
            send_track_info_to_api(track_data_wav)
        
        return {
            'name': f"{clean_name} {suffix}",
            'mp3': mp3_url,
            'wav': wav_url
        }
    
    # Check if genre is in the "simple" list
    if any(simple_genre in genre for simple_genre in simple_genres):
        print(f"Genre '{genre}' detected - Generating original only (no edits)")
        # Just export the original with metadata
        original = AudioSegment.from_mp3(original_path)
        edits.append(export_edit(original, "Original"))
    else:
        print(f"Genre '{genre}' - Generating full edit suite")
        # Generate full edits using the new processor
        generated_edits = audio_processor.process_track(vocals_path, inst_path, original_path, bpm)
        
        # Iterate over generated edits and export them
        for suffix, segment in generated_edits:
            edits.append(export_edit(segment, suffix))

    return edits

def run_demucs_thread(filepaths, original_filenames):
    global job_status
    try:
        job_status['state'] = 'processing'
        job_status['total_files'] = len(filepaths)
        job_status['results'] = []
        job_status['progress'] = 0

        current_file_index = 0

        for i in range(0, len(filepaths), 50):
            chunk = filepaths[i:i + 50]
            
            command = [
                'python3', '-m', 'demucs',
                '--two-stems=vocals',
                '-n', 'htdemucs',
                '--mp3',
                '--mp3-bitrate', '320',
                '-j', '4', 
                '-o', OUTPUT_FOLDER
            ] + chunk

            chunk_num = i // 50 + 1
            total_chunks = (len(filepaths) - 1) // 50 + 1
            print(f"Starting batch processing of chunk {chunk_num}/{total_chunks}...")
            
            # Reset progress for new chunk relative to file count? 
            # Ideally we track global file index.
            
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1, 
                universal_newlines=True
            )

            current_chunk_base = i

            for line in process.stdout:
                print(line, end='')
                
                if "Separating track" in line:
                    # Parse filename from line if possible, or just increment
                    # Demucs output: "Separating track filename.mp3"
                    try:
                        match = re.search(r"Separating track\s+(.+)$", line)
                        if match:
                            job_status['current_filename'] = match.group(1).strip()
                    except:
                        pass

                    current_file_index += 1
                    job_status['current_file_idx'] = current_file_index
                    
                    # Calculate global progress (0-50%)
                    # Phase 1 is separation (0-50%), Phase 2 is editing (50-100%)
                    # Actually, Demucs takes most of the time. Let's say Demucs is 0-90%?
                    # The user prompt implies Edit generation is fast.
                    # But previous code had 0-50 / 50-100.
                    # Let's keep 0-50 for Demucs for now, but update UI to be clearer.
                    
                    percent_per_file = 50 / len(filepaths)
                    base_progress = (current_file_index - 1) * percent_per_file
                    job_status['progress'] = int(base_progress)
                    job_status['current_step'] = f"Séparation IA (Lot {chunk_num}/{total_chunks})"

                elif "%|" in line:
                    # Demucs progress bar " 15%|███      | 20/130 [00:05<00:25,  4.23it/s]"
                    try:
                        # Extract percentage
                        parts = line.split('%|')
                        if len(parts) > 0:
                            percent_part = parts[0].strip()
                            # Use regex to find last number before %
                            p_match = re.search(r'(\d+)$', percent_part)
                            if p_match:
                                track_percent = int(p_match.group(1))
                                
                                # Add fractional progress for current file
                                percent_per_file = 50 / len(filepaths)
                                base_progress = (current_file_index - 1) * percent_per_file
                                added_progress = (track_percent / 100) * percent_per_file
                                job_status['progress'] = int(base_progress + added_progress)
                    except:
                        pass
            
            process.wait()
            
            if process.returncode != 0:
                job_status['state'] = 'error'
                job_status['error'] = 'Erreur lors du traitement Demucs'
                return

        print("Starting Edit Generation Phase...")
        job_status['progress'] = 50
        job_status['current_step'] = "Génération des Edits"
        
        all_results = []
        
        for i, filepath in enumerate(filepaths):
            filename = os.path.basename(filepath)
            
            # Update status for current file
            job_status['current_file_idx'] = i + 1
            job_status['current_filename'] = filename
            job_status['current_step'] = "Création des versions DJ (Edits)"
            
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

def get_git_info():
    try:
        # Get hash
        hash_output = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
        # Get date
        date_output = subprocess.check_output(['git', 'log', '-1', '--format=%cd', '--date=format:%a %b %d %H:%M']).strip().decode('utf-8')
        
        # Get count of commits to simulate version number if needed, or just use hardcoded base
        # Using a simple counter for versioning: v0.20 + (commits since last tag or simple count)
        # For now, let's keep it simple: just show the hash/date dynamically.
        # But user asked for "Version update tout seul".
        # Let's count total commits as a "build number" or similar.
        count = subprocess.check_output(['git', 'rev-list', '--count', 'HEAD']).strip().decode('utf-8')
        
        return f"v0.{count} ({hash_output}) - {date_output}"
    except:
        return "Dev Version"

@app.route('/')
def index():
    version_info = get_git_info()
    return render_template('index.html', version_info=version_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    global job_status
    
    if job_status['state'] == 'processing':
        # Check if the process is actually running/alive?
        # Sometimes state gets stuck if thread died silently.
        # But for now, we assume user is impatient or trying to upload while busy.
        # Let's allow FORCE reset if they reload page?
        # No, that might break current job.
        
        # NOTE FOR USER: If you get this 409 error repeatedly without any active process visible,
        # it might be a stuck state. Restart the server (pkill python; python app.py) to fix.
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
        'current_step': 'Initialisation...', # Added detail
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
    # Use environment variable for port, or default to 5001. 
    # RunPod often expects 8888 for some proxy setups, so we can try to default to that or just be flexible.
    # The user URL suggests port 8888.
    port = int(os.environ.get('PORT', 8888))
    app.run(host='0.0.0.0', port=port, debug=True)
