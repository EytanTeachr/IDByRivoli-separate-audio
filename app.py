import os
import subprocess
import threading
import shutil
import time
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to track progress
# In a real multi-user app, this should be a database or Redis
job_status = {
    'state': 'idle', # idle, processing, completed, error
    'progress': 0,
    'total_files': 0,
    'current_file_idx': 0,
    'current_filename': '',
    'results': [],
    'error': None
}

def run_demucs_thread(filepaths, original_filenames):
    global job_status
    try:
        job_status['state'] = 'processing'
        job_status['total_files'] = len(filepaths)
        job_status['results'] = []
        job_status['progress'] = 0

        # We process files in batches to update progress per file
        # Running one big command is faster for model loading, but harder for individual file progress tracking
        # Compromise: Pass all files to ONE command to load model once, but parsing progress is harder.
        # Actually, Demucs prints progress bars for each file. We can parse that.
        
        # Construct command with ALL files
        command = [
            'python3', '-m', 'demucs',
            '--two-stems=vocals',
            '-n', 'htdemucs',
            '--mp3',
            '--mp3-bitrate', '320',
            '-j', '4', # Multithreading
            '-o', OUTPUT_FOLDER
        ] + filepaths

        print(f"Starting batch processing of {len(filepaths)} files...")
        
        # Use Popen to read stdout/stderr
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            bufsize=1, 
            universal_newlines=True
        )

        # We need to read output to track progress. 
        # Demucs outputs "Separating track filename..."
        
        # Since reading both stdout and stderr can deadlock, and Demucs puts progress on stderr, we focus on stderr.
        # But we also want to catch errors.
        
        current_file_index = 0
        
        while True:
            line = process.stderr.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                # Print to console for debugging
                print(line, end='')
                
                # Try to parse progress
                # Demucs output example: "Separating track path/to/file.mp3"
                if "Separating track" in line:
                    current_file_index += 1
                    job_status['current_file_idx'] = current_file_index
                    job_status['current_filename'] = os.path.basename(filepaths[current_file_index-1])
                    # Update progress % based on file count
                    # We can't easily get % within a file without complex parsing of the progress bar chars
                    job_status['progress'] = int((current_file_index - 1) / len(filepaths) * 100)
                
                # Approximate progress within file (Demucs uses tqdm which outputs 10%... etc)
                # This is tricky to parse reliably from stderr due to carriage returns \r
        
        return_code = process.poll()
        
        if return_code != 0:
            job_status['state'] = 'error'
            job_status['error'] = 'Erreur lors du traitement Demucs'
            return

        # Post-processing: Rename and organize files
        processed_files = []
        
        for i, filepath in enumerate(filepaths):
            filename = os.path.basename(filepath)
            original_name = os.path.splitext(filename)[0]
            track_name = original_name # Demucs usually uses the filename without ext as folder name
            
            # Demucs output structure: output/htdemucs/{track_name}/vocals.mp3
            source_dir = os.path.join(OUTPUT_FOLDER, 'htdemucs', track_name)
            
            # Target filenames
            # "Fichier A instrumental uniquement.mp3"
            # "Fichier A voix uniquement.mp3"
            
            inst_filename = f"{original_name} instrumental uniquement.mp3"
            vocals_filename = f"{original_name} voix uniquement.mp3"
            
            inst_path_src = os.path.join(source_dir, 'no_vocals.mp3')
            vocals_path_src = os.path.join(source_dir, 'vocals.mp3')
            
            inst_path_dest = os.path.join(PROCESSED_FOLDER, inst_filename)
            vocals_path_dest = os.path.join(PROCESSED_FOLDER, vocals_filename)
            
            if os.path.exists(inst_path_src) and os.path.exists(vocals_path_src):
                shutil.copy2(inst_path_src, inst_path_dest)
                shutil.copy2(vocals_path_src, vocals_path_dest)
                
                processed_files.append({
                    'original': filename,
                    'instrumental': f'/download_processed/{inst_filename}',
                    'vocals': f'/download_processed/{vocals_filename}'
                })
            else:
                print(f"Warning: Output files not found for {track_name}")

        job_status['progress'] = 100
        job_status['results'] = processed_files
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
    
    # Start processing in background thread
    thread = threading.Thread(target=run_demucs_thread, args=(saved_filepaths, original_filenames))
    thread.start()
    
    return jsonify({'message': 'Traitement démarré', 'total_files': len(files)})

@app.route('/status')
def status():
    return jsonify(job_status)

@app.route('/download_processed/<path:filename>')
def download_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    # Port 5001 pour éviter les conflits avec AirPlay sur Mac
    app.run(host='0.0.0.0', port=5001, debug=True)
