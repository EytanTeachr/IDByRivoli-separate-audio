import os
import subprocess
from flask import Flask, render_template, request, jsonify, send_from_directory
import shutil

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier envoyé'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Clean up output folder for this file if it exists
        track_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_FOLDER, 'htdemucs', track_name)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        # Run demucs
        # Options optimized for Quality and Speed on local machine:
        # -n htdemucs: High quality Hybrid Transformer model (industry standard)
        # --mp3-bitrate 320: Max MP3 quality to avoid compression artifacts
        # -j 4: Use 4 parallel threads for faster processing (CPU optimization)
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
        
        try:
            print(f"Running command: {' '.join(command)}")
            # On utilise Popen pour afficher la sortie en temps réel dans le terminal
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True) as p:
                # Afficher la sortie standard et d'erreur en temps réel
                for line in p.stderr:
                    print(line, end='') 
                
                p.wait()
                if p.returncode != 0:
                     print(f"Demucs failed with return code {p.returncode}")
                     return jsonify({'error': 'Erreur lors du traitement'}), 500
            
            # The output files should be in output/htdemucs/{track_name}/
            # vocals.mp3 and no_vocals.mp3
            
            # Check if files exist
            vocals_path = os.path.join(OUTPUT_FOLDER, 'htdemucs', track_name, 'vocals.mp3')
            no_vocals_path = os.path.join(OUTPUT_FOLDER, 'htdemucs', track_name, 'no_vocals.mp3')
            
            if not os.path.exists(vocals_path) or not os.path.exists(no_vocals_path):
                 return jsonify({'error': 'Les fichiers de sortie sont introuvables.'}), 500

            return jsonify({
                'vocals': f'/download/{track_name}/vocals.mp3',
                'no_vocals': f'/download/{track_name}/no_vocals.mp3'
            })
            
        except Exception as e:
            print(f"Exception: {str(e)}")
            return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    # filename will be like "track_name/vocals.wav"
    # We serve from OUTPUT_FOLDER/htdemucs/
    return send_from_directory(os.path.join(OUTPUT_FOLDER, 'htdemucs'), filename, as_attachment=True)

if __name__ == '__main__':
    # Port 5001 pour éviter les conflits avec AirPlay sur Mac (qui utilise souvent le port 5000)
    app.run(host='0.0.0.0', port=5001, debug=True)
