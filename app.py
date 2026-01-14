import os
import subprocess
import threading
import shutil
import time
import re
from flask import Flask, render_template, request, jsonify, send_from_directory, abort, send_file
from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, TIT2, TPE1, APIC, COMM, TALB, TDRC, TRCK, TCON, TBPM, TSRC, TLEN, TPUB, TMED, WOAR, WXXX, TXXX
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
    'current_step': '',
    'results': [],
    'error': None,
    'logs': []  # Added logs list
}

def log_message(message):
    """Adds a message to the job logs and prints it."""
    print(message)
    timestamp = time.strftime("%H:%M:%S")
    job_status['logs'].append(f"[{timestamp}] {message}")
    # Keep only last 1000 logs to prevent memory issues
    if len(job_status['logs']) > 1000:
        job_status['logs'] = job_status['logs'][-1000:]

import shutil
import zipfile
import io

@app.route('/download_all_zip')
def download_all_zip():
    """
    Creates a ZIP file containing all processed tracks and sends it to the user.
    Can be called at any time to get currently finished tracks.
    """
    global job_status
    
    # Refresh results from disk if needed
    if not job_status['results']:
        # ... (logic to populate from disk, similar to status route)
        processed_dirs = [d for d in os.listdir(PROCESSED_FOLDER) if os.path.isdir(os.path.join(PROCESSED_FOLDER, d))]
        # We need to rebuild job_status['results'] or just iterate dirs directly
        pass 

    # Create an in-memory ZIP file
    memory_file = io.BytesIO()
    
    # We zip everything currently in PROCESSED_FOLDER
    has_files = False
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROCESSED_FOLDER):
             for file in files:
                if file.lower().endswith(('.mp3', '.wav')): 
                    file_path = os.path.join(root, file)
                    # Create relative path inside zip: "Track Name/Track Name Main.mp3"
                    rel_path = os.path.relpath(file_path, PROCESSED_FOLDER)
                    zf.write(file_path, rel_path)
                    has_files = True

    if not has_files:
        return jsonify({'error': 'Aucun fichier traité disponible pour le moment'}), 400

    memory_file.seek(0)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'ID_By_Rivoli_Pack_{timestamp}.zip'
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
        
        # 8. ISRC (from original) - IMPORTANT: Always include
        isrc_value = ''
        if original_tags and 'TSRC' in original_tags:
            isrc_value = str(original_tags['TSRC'].text[0]) if original_tags['TSRC'].text else ''
            tags.add(TSRC(encoding=3, text=isrc_value))
        
        # 9. Publisher
        tags.add(TPUB(encoding=3, text='ID By Rivoli'))
        
        # 10. Custom Track ID: $ISRC_$filename (clean format: no dashes, single underscores only)
        # Extract clean filename (without path and extension)
        filename_base = os.path.splitext(os.path.basename(filepath))[0]
        # Replace dashes with spaces, then normalize spaces, then convert to underscores
        filename_clean = filename_base.replace('-', ' ').replace('_', ' ')
        filename_clean = re.sub(r'\s+', ' ', filename_clean).strip()  # Multiple spaces -> single space
        filename_clean = filename_clean.replace(' ', '_')  # Spaces -> underscores
        filename_clean = re.sub(r'_+', '_', filename_clean)  # Multiple underscores -> single underscore
        
        track_id = f"{isrc_value}_{filename_clean}" if isrc_value else filename_clean
        tags.add(TXXX(encoding=3, desc='TRACK_ID', text=track_id))
        
        # 11. Length
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
API_ENDPOINT = os.environ.get('API_ENDPOINT', 'https://track.idbyrivoli.com/upload')
API_KEY = os.environ.get('API_KEY', '5X#JP5ifkSm?oE6@haMriYG$j!87BEfX@zg3CxcE')

# Dynamic Public URL handling
CURRENT_HOST_URL = os.environ.get('PUBLIC_URL', '')

@app.before_request
def set_public_url():
    """Captures the current public URL from the request headers to support dynamic Pod URLs (RunPod, etc.)."""
    global CURRENT_HOST_URL
    
    # Always try to get the best URL from headers on each request
    # Priority: X-Forwarded-Host > Host header > existing value
    forwarded_host = request.headers.get('X-Forwarded-Host')
    original_host = request.headers.get('Host')
    scheme = request.headers.get('X-Forwarded-Proto', 'https')  # Default to https for pods
    
    # RunPod and similar platforms set X-Forwarded-Host to the public URL
    if forwarded_host:
        new_url = f"{scheme}://{forwarded_host}"
    elif original_host and not original_host.startswith(('10.', '172.', '192.168.', '100.')):
        # Use Host header only if it's not a private IP
        new_url = f"{scheme}://{original_host}"
    else:
        new_url = None
    
    # Update if we found a valid public URL
    if new_url and new_url != CURRENT_HOST_URL:
        CURRENT_HOST_URL = new_url
        print(f"📍 Public URL détectée: {CURRENT_HOST_URL}")

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
        
        # Log the URL being sent
        print(f"📤 API: {track_data['Titre']} ({track_data['Format']}) → {track_data.get('Fichiers', 'N/A')}")
        
        response = requests.post(API_ENDPOINT, json=track_data, headers=headers, timeout=30)
        
        if response.status_code in [200, 202]:
            print(f"✅ API SUCCESS: {track_data['Titre']} ({track_data['Format']})")
            log_message(f"API OK: {track_data['Titre']} ({track_data['Format']}) → {track_data.get('Fichiers', '')}")
        else:
            print(f"❌ API ERROR {response.status_code}: {response.text[:200]}")
            log_message(f"API ERROR {response.status_code} pour {track_data['Titre']}")
            
    except Exception as e:
        print(f"❌ API EXCEPTION: {e}")
        log_message(f"API EXCEPTION: {e}")

def prepare_track_metadata(edit_info, original_path, bpm, base_url=""):
    """
    Prepares track metadata for API export with absolute URLs.
    """
    global CURRENT_HOST_URL
    
    # Fallback if request hasn't set it yet
    base_url = CURRENT_HOST_URL if CURRENT_HOST_URL else "http://localhost:8888"
    
    try:
        # Read original metadata
        original_audio = MP3(original_path, ID3=ID3)
        original_tags = original_audio.tags if original_audio.tags else {}
        
        # Extract fields
        artist = str(original_tags.get('TPE1', 'Unknown')).strip() if 'TPE1' in original_tags else 'Unknown'
        album = str(original_tags.get('TALB', '')).strip() if 'TALB' in original_tags else ''
        genre = str(original_tags.get('TCON', '')).strip() if 'TCON' in original_tags else ''
        
        # ISRC extraction
        isrc = ''
        if 'TSRC' in original_tags:
            isrc = str(original_tags['TSRC'].text[0]).strip() if original_tags['TSRC'].text else ''
        
        # Date handling
        date_str = str(original_tags.get('TDRC', '')).strip() if 'TDRC' in original_tags else ''
        try:
            if date_str:
                date_obj = datetime.strptime(date_str[:10], '%Y-%m-%d')
                date_sortie = int(date_obj.timestamp())
            else:
                date_sortie = 0
        except:
            date_sortie = 0
        
        # Publisher/Label
        label = str(original_tags.get('TPUB', 'ID By Rivoli')).strip() if 'TPUB' in original_tags else 'ID By Rivoli'
        
        # Construct ABSOLUTE URLs using DYNAMIC BASE URL
        relative_url = edit_info.get('url', '')
        absolute_url = f"{base_url}{relative_url}" if relative_url else ''
        
        # Extract original cover (Cover 2) and save it, then use that URL
        cover_url = f"{base_url}/static/covers/Cover_Id_by_Rivoli.jpeg"  # Fallback
        
        # Try to extract original cover from source file
        if original_tags:
            for apic_key in original_tags.keys():
                if apic_key.startswith('APIC:'):
                    try:
                        original_apic = original_tags[apic_key]
                        # Generate unique filename based on track
                        track_name_clean = re.sub(r'[^\w\s-]', '', os.path.splitext(os.path.basename(original_path))[0])
                        track_name_clean = track_name_clean.replace(' ', '_')[:50]
                        
                        # Determine extension from mime type
                        ext = 'jpg' if 'jpeg' in original_apic.mime else 'png'
                        cover_filename = f"cover_{track_name_clean}.{ext}"
                        cover_save_path = os.path.join(BASE_DIR, 'static', 'covers', cover_filename)
                        
                        # Save the original cover
                        with open(cover_save_path, 'wb') as f:
                            f.write(original_apic.data)
                        
                        # Use the original cover URL
                        cover_url = f"{base_url}/static/covers/{cover_filename}"
                        break
                    except Exception as e:
                        print(f"Could not extract original cover: {e}")
        
        # Generate Track ID (clean format: no dashes, single underscores only)
        filename_raw = edit_info.get('name', '')
        filename_clean = filename_raw.replace('-', ' ').replace('_', ' ')
        filename_clean = re.sub(r'\s+', ' ', filename_clean).strip()
        filename_clean = filename_clean.replace(' ', '_')
        filename_clean = re.sub(r'_+', '_', filename_clean)
        
        track_id = f"{isrc}_{filename_clean}" if isrc else filename_clean
        
        # Prepare data structure
        track_data = {
            'Type': edit_info.get('type', ''),
            'Format': edit_info.get('format', 'MP3'),
            'Titre': edit_info.get('name', ''),
            'Artiste': artist,
            'Fichiers': absolute_url,
            'Univers': '',
            'Mood': '',
            'Style': genre,
            'Album': album,
            'Label': 'ID By Rivoli',
            'Sous-label': label if label != 'ID By Rivoli' else '',
            'Date de sortie': date_sortie,
            'BPM': bpm,
            'Artiste original': artist,
            'Url': cover_url,
            'ISRC': isrc,
            'TRACK_ID': track_id
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
    log_message(f"BPM détecté pour {base_filename}: {bpm}")
    
    # FORCE MAIN ONLY MODE FOR ALL GENRES (TEMPORARY OVERRIDE)
    # Check genre to determine if we should generate full edits or just preserve original
    try:
        original_audio = MP3(original_path, ID3=ID3)
        original_tags = original_audio.tags
        genre = str(original_tags.get('TCON', '')).lower() if original_tags and 'TCON' in original_tags else ''
    except:
        genre = ''
    
    # Genres that should NOT get edits (just original MP3/WAV)
    # simple_genres = ['house', 'electro house', 'dance']
    
    edits = []

    def export_edit(audio_segment, suffix):
        clean_name, _ = clean_filename(base_filename)
        out_name_mp3 = f"{clean_name} {suffix}.mp3"
        out_name_wav = f"{clean_name} {suffix}.wav"
        
        out_path_mp3 = os.path.join(base_output_path, out_name_mp3)
        out_path_wav = os.path.join(base_output_path, out_name_wav)
        
        audio_segment.export(out_path_mp3, format="mp3", bitrate="320k")
        audio_segment.export(out_path_wav, format="wav")
        
        # Update metadata for MP3 (full metadata + cover)
        update_metadata(out_path_mp3, "ID By Rivoli", f"{clean_name} {suffix}", original_path, bpm)
        
        # Update metadata for WAV (cover art included)
        update_metadata_wav(out_path_wav, "ID By Rivoli", f"{clean_name} {suffix}", original_path, bpm)
        
        subdir = clean_name
        
        # New robust URL format using query parameter
        # Path relative to PROCESSED_FOLDER: "Subdir/Filename.mp3"
        rel_path_mp3 = f"{subdir}/{out_name_mp3}"
        rel_path_wav = f"{subdir}/{out_name_wav}"
        
        # IMPORTANT: safe='/' to NOT encode the slash!
        mp3_url = f"/download_file?path={urllib.parse.quote(rel_path_mp3, safe='/')}"
        wav_url = f"/download_file?path={urllib.parse.quote(rel_path_wav, safe='/')}"
        
        # VERIFICATION: Check if files actually exist where we expect them
        expected_mp3_path = os.path.join(PROCESSED_FOLDER, rel_path_mp3)
        expected_wav_path = os.path.join(PROCESSED_FOLDER, rel_path_wav)
        
        print(f"\n{'='*60}")
        print(f"📁 FILE GENERATION CHECK:")
        print(f"   Subdir (clean_name): '{subdir}'")
        print(f"   MP3 filename: '{out_name_mp3}'")
        print(f"   WAV filename: '{out_name_wav}'")
        print(f"   ")
        print(f"   Expected MP3 path: {expected_mp3_path}")
        print(f"   MP3 EXISTS: {os.path.exists(expected_mp3_path)}")
        print(f"   ")
        print(f"   Expected WAV path: {expected_wav_path}")
        print(f"   WAV EXISTS: {os.path.exists(expected_wav_path)}")
        print(f"   ")
        # Get the full URL with base
        base_url = CURRENT_HOST_URL if CURRENT_HOST_URL else "http://localhost:8888"
        full_mp3_url = f"{base_url}{mp3_url}"
        full_wav_url = f"{base_url}{wav_url}"
        
        print(f"   Generated MP3 URL: {full_mp3_url}")
        print(f"   Generated WAV URL: {full_wav_url}")
        print(f"{'='*60}\n")
        
        # Log to UI as well - FULL URLs
        log_message(f"📥 URL MP3: {full_mp3_url}")
        log_message(f"📥 URL WAV: {full_wav_url}")
        
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
    
    # Export 3 versions: Main, Acapella, Instrumental
    log_message(f"Génération des 3 versions pour : {base_filename}")
    
    # 1. Main (Original)
    original = AudioSegment.from_mp3(original_path)
    edits.append(export_edit(original, "Main"))
    
    # 2. Acapella (Vocals only) - if available
    if vocals_path and os.path.exists(vocals_path):
        vocals = AudioSegment.from_mp3(vocals_path)
        edits.append(export_edit(vocals, "Acapella"))
        log_message(f"✓ Version Acapella créée")
    else:
        log_message(f"⚠️ Pas de fichier vocals pour Acapella")
    
    # 3. Instrumental (No vocals) - if available
    if inst_path and os.path.exists(inst_path):
        instrumental = AudioSegment.from_mp3(inst_path)
        edits.append(export_edit(instrumental, "Instrumental"))
        log_message(f"✓ Version Instrumentale créée")
    else:
        log_message(f"⚠️ Pas de fichier instrumental")

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
            log_message(f"Démarrage de la séparation IA (Lot {chunk_num}/{total_chunks})...")
            
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
                            filename_found = match.group(1).strip()
                            job_status['current_filename'] = filename_found
                            log_message(f"Séparation en cours : {filename_found}")
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
        log_message("Fin de la séparation IA. Début de la génération des Edits DJ...")
        job_status['progress'] = 50
        job_status['current_step'] = "Génération des Edits"
        
        all_results = []
        
        for i, filepath in enumerate(filepaths):
            filename = os.path.basename(filepath)
            
            # Update status for current file
            job_status['current_file_idx'] = i + 1
            job_status['current_filename'] = filename
            job_status['current_step'] = "Création des versions DJ (Edits)"
            log_message(f"Création des edits pour : {filename}")
            
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

import queue

import json

# Persistence File
HISTORY_FILE = os.path.join(BASE_DIR, 'processed_history.json')

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_to_history(filename):
    history = load_history()
    if filename not in history:
        history.append(filename)
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f)

# Load initial history
processed_history = load_history()

# Global Queue for processing tracks
track_queue = queue.Queue()

# Worker thread function
def worker():
    global job_status
    while True:
        try:
            filename = track_queue.get()
            if filename is None:
                break
            
            # Check if already processed
            if filename in load_history():
                log_message(f"⏩ Déjà traité (ignoré) : {filename}")
                track_queue.task_done()
                # Reset state if queue is empty
                if track_queue.empty():
                    job_status['state'] = 'idle'
                    job_status['current_step'] = ''
                    job_status['current_filename'] = ''
                continue
            
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Check if file exists (might have been deleted)
            if not os.path.exists(filepath):
                 log_message(f"⚠️ Fichier introuvable (ignoré) : {filename}")
                 track_queue.task_done()
                 # Reset state if queue is empty
                 if track_queue.empty():
                     job_status['state'] = 'idle'
                     job_status['current_step'] = ''
                     job_status['current_filename'] = ''
                 continue

            process_single_track(filepath, filename)
            
            # Mark as done in history
            save_to_history(filename)
            
            track_queue.task_done()
            
            # Reset state to idle if queue is empty (ready for new files)
            if track_queue.empty():
                job_status['state'] = 'idle'
                job_status['current_step'] = 'Prêt pour de nouveaux fichiers'
                job_status['current_filename'] = ''
                log_message("✅ File d'attente terminée - Prêt pour de nouveaux fichiers")
                
        except Exception as e:
            print(f"Worker Error: {e}")
            log_message(f"Erreur Worker: {e}")
            job_status['state'] = 'idle'  # Reset on error too

# Start worker thread
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

# Restore pending files on startup
def restore_queue():
    """Scans upload folder and re-queues files that haven't been processed yet."""
    log_message("🔄 Vérification des fichiers en attente...")
    upload_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith('.mp3')]
    history = load_history()
    
    count = 0
    for f in upload_files:
        if f not in history:
            track_queue.put(f)
            count += 1
            
    if count > 0:
        log_message(f"♻️ Restauration de {count} fichiers dans la file d'attente.")

# Call restore on startup
restore_queue()

# Modified process function for SINGLE track
def process_single_track(filepath, filename):
    global job_status
    
    try:
        job_status['state'] = 'processing'
        job_status['current_filename'] = filename
        job_status['current_step'] = "Séparation IA (Demucs)..."
        log_message(f"🚀 Début traitement : {filename}")
        
        track_name = os.path.splitext(filename)[0]
        
        # 1. Run Demucs separation
        log_message(f"🎵 Séparation vocale/instrumentale en cours...")
        
        command = [
            'python3', '-m', 'demucs',
            '--two-stems=vocals',
            '-n', 'htdemucs',
            '--mp3',
            '--mp3-bitrate', '320',
            '-j', '4',
            '-o', OUTPUT_FOLDER,
            filepath
        ]
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        for line in process.stdout:
            print(line, end='')
            # Update progress from demucs output
            if "%|" in line:
                try:
                    parts = line.split('%|')
                    if len(parts) > 0:
                        percent_part = parts[0].strip()
                        p_match = re.search(r'(\d+)$', percent_part)
                        if p_match:
                            track_percent = int(p_match.group(1))
                            job_status['progress'] = int(track_percent * 0.7)  # Demucs = 70%
                except:
                    pass
        
        process.wait()
        
        if process.returncode != 0:
            log_message(f"❌ Erreur Demucs pour {filename}")
            return
        
        # 2. Generate edits (Main, Acapella, Instrumental)
        job_status['current_step'] = "Génération des versions..."
        job_status['progress'] = 70
        
        clean_name, _ = clean_filename(filename)
        track_output_dir = os.path.join(PROCESSED_FOLDER, clean_name)
        os.makedirs(track_output_dir, exist_ok=True)
        
        # Get separated files
        source_dir = os.path.join(OUTPUT_FOLDER, 'htdemucs', track_name)
        inst_path = os.path.join(source_dir, 'no_vocals.mp3')
        vocals_path = os.path.join(source_dir, 'vocals.mp3')
        
        if os.path.exists(inst_path) and os.path.exists(vocals_path):
            edits = create_edits(vocals_path, inst_path, filepath, track_output_dir, filename)
            
            # Add to results
            job_status['results'].append({
                'original': clean_name,
                'edits': edits
            })
            log_message(f"✅ Terminé : {clean_name}")
        else:
            log_message(f"⚠️ Fichiers séparés non trouvés pour {filename}")
        
        job_status['progress'] = 100

    except Exception as e:
        log_message(f"❌ Erreur critique {filename}: {e}")

@app.route('/enqueue_file', methods=['POST'])
def enqueue_file():
    data = request.json
    filename = data.get('filename')
    
    if filename:
        track_queue.put(filename)
        q_size = track_queue.qsize()
        log_message(f"📥 Ajouté à la file : {filename} (File d'attente: {q_size})")
        return jsonify({'message': 'Queued', 'queue_size': q_size})
    
    return jsonify({'error': 'No filename'}), 400

@app.route('/upload_chunk', methods=['POST'])
def upload_chunk():
    """
    Receives a single file upload and saves it to the upload folder.
    This allows the frontend to sequence uploads 1 by 1.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'message': f'File {filename} uploaded successfully'})

@app.route('/start_processing', methods=['POST'])
def start_processing():
    """
    Triggered after all uploads are done.
    Scans the uploads folder and starts the processing thread.
    """
    global job_status
    
    if job_status['state'] == 'processing':
        return jsonify({'error': 'Un traitement est déjà en cours. Veuillez patienter.'}), 409

    # Scan upload folder for MP3s
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith('.mp3')]
    
    if not files:
        return jsonify({'error': 'Aucun fichier trouvé dans le dossier uploads'}), 400

    saved_filepaths = [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in files]
    original_filenames = files # filenames are just the basenames
    
    job_status = {
        'state': 'starting',
        'progress': 0,
        'total_files': len(files),
        'current_file_idx': 0,
        'current_filename': '',
        'current_step': 'Initialisation...',
        'results': [],
        'error': None,
        'logs': []
    }
    
    log_message(f"Traitement démarré pour {len(files)} fichier(s) (Mode Batch)")
    
    thread = threading.Thread(target=run_demucs_thread, args=(saved_filepaths, original_filenames))
    thread.start()
    
    return jsonify({'message': 'Traitement démarré', 'total_files': len(files)})

@app.route('/upload', methods=['POST'])
def upload_file():
    # Keep legacy endpoint for backward compatibility if needed, 
    # but strictly we should move to the new flow.
    # ... (redirecting to new logic ideally, but let's keep it simple)
    return jsonify({'error': 'Please use the new sequential upload flow'}), 400

@app.route('/status')
def status():
    # Update queue info in status
    job_status['queue_size'] = track_queue.qsize()
    
    # If results are empty (e.g. after restart), populate them from disk
    if not job_status['results']:
        processed_dirs = [d for d in os.listdir(PROCESSED_FOLDER) if os.path.isdir(os.path.join(PROCESSED_FOLDER, d))]
        for d in processed_dirs:
            # Reconstruct result object
            track_dir = os.path.join(PROCESSED_FOLDER, d)
            files = [f for f in os.listdir(track_dir) if f.endswith(('.mp3', '.wav'))]
            
            # Simple reconstruction of edits list
            edits = []
            for f in files:
                ext = 'mp3' if f.endswith('.mp3') else 'wav'
                # Try to guess edit type from filename suffix? 
                # Not strictly necessary for display, just name and URL needed.
                # Filename: "Track Name Suffix.mp3"
                
                subdir = d
                url = f"/download_processed/{urllib.parse.quote(subdir)}/{urllib.parse.quote(f)}"
                
                # We only want one entry per edit type (MP3/WAV pair ideally), 
                # but for simple display list, we can group them in UI or just send raw.
                # The UI expects objects with {name, mp3, wav}.
                pass 
            
            # Better approach: Group by name (without extension)
            grouped = {}
            for f in files:
                name_no_ext = os.path.splitext(f)[0]
                if name_no_ext not in grouped:
                    grouped[name_no_ext] = {'name': name_no_ext, 'mp3': '#', 'wav': '#'}
                
                subdir = d
                # New robust URL format - safe='/' to keep slashes!
                rel_path = f"{subdir}/{f}"
                url = f"/download_file?path={urllib.parse.quote(rel_path, safe='/')}"
                
                if f.endswith('.mp3'):
                    grouped[name_no_ext]['mp3'] = url
                else:
                    grouped[name_no_ext]['wav'] = url
            
            job_status['results'].append({
                'original': d,
                'edits': list(grouped.values())
            })
            
    return jsonify(job_status)

@app.route('/download_file')
def download_file():
    """
    Robust download route using query parameter.
    Usage: /download_file?path=SubDir/File.mp3
    """
    relative_path = request.args.get('path')
    
    print(f"📥 DOWNLOAD REQUEST")
    print(f"   Raw path param: {relative_path}")
    
    if not relative_path:
        print("   ❌ No path provided")
        abort(400)
    
    # Security: prevent directory traversal
    if '..' in relative_path:
        print("   ❌ Directory traversal attempt")
        abort(403)
        
    # Construct full path - path should already be decoded by Flask
    filepath = os.path.join(PROCESSED_FOLDER, relative_path)
    
    print(f"   Looking for: {filepath}")
    print(f"   File exists: {os.path.exists(filepath)}")
    
    if not os.path.exists(filepath):
        # Debug: list what's actually in the processed folder
        print(f"   ❌ FILE NOT FOUND!")
        print(f"   Contents of PROCESSED_FOLDER:")
        for item in os.listdir(PROCESSED_FOLDER):
            item_path = os.path.join(PROCESSED_FOLDER, item)
            if os.path.isdir(item_path):
                print(f"      📁 {item}/")
                for subitem in os.listdir(item_path)[:5]:
                    print(f"         - {subitem}")
            else:
                print(f"      📄 {item}")
        abort(404)
    
    # Use send_file with absolute path (most reliable)
    print(f"   ✅ Sending file: {filepath}")
    return send_file(
        filepath,
        as_attachment=True,
        download_name=os.path.basename(filepath)
    )

# Serve static files from processed folder directly
@app.route('/processed/<path:filepath>')
def serve_processed_file(filepath):
    """Alternative route: serve files directly from processed folder."""
    full_path = os.path.join(PROCESSED_FOLDER, filepath)
    print(f"📥 SERVE PROCESSED: {filepath}")
    print(f"   Full path: {full_path}")
    print(f"   Exists: {os.path.exists(full_path)}")
    
    if not os.path.exists(full_path):
        abort(404)
    
    return send_file(full_path, as_attachment=True)

# Debug route to list all processed files
@app.route('/list_files')
def list_files():
    """Debug route to see what files are available."""
    result = {}
    for subdir in os.listdir(PROCESSED_FOLDER):
        subdir_path = os.path.join(PROCESSED_FOLDER, subdir)
        if os.path.isdir(subdir_path):
            result[subdir] = os.listdir(subdir_path)
    return jsonify(result)

# Debug route to check detected public URL
@app.route('/debug_url')
def debug_url():
    """Debug route to see the detected public URL and request headers."""
    return jsonify({
        'CURRENT_HOST_URL': CURRENT_HOST_URL,
        'PUBLIC_URL_ENV': os.environ.get('PUBLIC_URL', ''),
        'headers': {
            'Host': request.headers.get('Host'),
            'X-Forwarded-Host': request.headers.get('X-Forwarded-Host'),
            'X-Forwarded-Proto': request.headers.get('X-Forwarded-Proto'),
            'X-Real-IP': request.headers.get('X-Real-IP'),
        },
        'request_host': request.host,
        'request_url': request.url,
    })

# Test route to check URL generation
@app.route('/test_download')
def test_download():
    """Test route that lists all files with their download URLs and tests them."""
    results = []
    
    for subdir in os.listdir(PROCESSED_FOLDER):
        subdir_path = os.path.join(PROCESSED_FOLDER, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                rel_path = f"{subdir}/{filename}"
                url = f"/download_file?path={urllib.parse.quote(rel_path, safe='/')}"
                
                # Test if the path would work
                test_path = os.path.join(PROCESSED_FOLDER, rel_path)
                
                results.append({
                    'subdir': subdir,
                    'filename': filename,
                    'rel_path': rel_path,
                    'url': url,
                    'file_exists_at_original': os.path.exists(file_path),
                    'file_exists_at_test_path': os.path.exists(test_path),
                    'paths_match': file_path == test_path
                })
    
    return jsonify({
        'PROCESSED_FOLDER': PROCESSED_FOLDER,
        'total_files': len(results),
        'files': results
    })

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    """
    Deletes all files in uploads, output, and processed directories to free up disk space.
    """
    global job_status
    
    try:
        # Clear directories
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, PROCESSED_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

        # Reset Job Status
        job_status = {
            'state': 'idle', 
            'progress': 0,
            'total_files': 0,
            'current_file_idx': 0,
            'current_filename': '',
            'current_step': '',
            'results': [],
            'error': None,
            'logs': [],
            'queue_size': 0
        }
        
        # Clear Queue (drain it)
        with track_queue.mutex:
            track_queue.queue.clear()
            
        # Clear History
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            
        log_message("🧹 Espace disque et historique nettoyés avec succès.")
        return jsonify({'message': 'Cleanup successful'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ID By Rivoli Audio Processor')
    parser.add_argument('-p', '--port', type=int, default=int(os.environ.get('PORT', 8888)),
                        help='Port to run the server on (default: 8888)')
    args = parser.parse_args()
    
    print(f"🚀 Starting ID By Rivoli on port {args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=True)
