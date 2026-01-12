import numpy as np
import librosa
from pydub import AudioSegment
from pydub.generators import WhiteNoise, Sine
import random
import os

def detect_bpm(file_path):
    try:
        y, sr = librosa.load(file_path, duration=120)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, 'item'):
            return round(tempo.item())
        elif isinstance(tempo, np.ndarray):
            return round(float(tempo[0])) if tempo.size > 0 else 120
        return round(tempo)
    except Exception as e:
        print(f"Error detecting BPM: {e}")
        return 120

def find_drop_start(inst_segment, beat_ms, sr=44100):
    """
    Finds the start of the drop (loudest 32-beat section).
    Skips the first 15 seconds to avoid loud intros.
    """
    ms_32_beats = 32 * beat_ms
    
    # Convert to mono for analysis
    mono_inst = inst_segment.set_channels(1)
    
    # Skip first 15 seconds if track is long enough
    start_offset = 15000 if len(mono_inst) > 45000 else 0
    
    analysis_audio = mono_inst[start_offset:]
    
    if len(analysis_audio) < ms_32_beats:
        return 0
        
    # We will slide a window of 32 beats and find max RMS
    # To save time, we can sample every 1 beat (approx)
    step = int(beat_ms)
    max_rms = 0
    best_start = 0
    
    # Limit search to first 2 minutes + offset to save time? 
    # Or search whole track. Drops are usually in 0:45 - 2:00 range.
    search_duration = min(len(analysis_audio), 4 * 60 * 1000) # Search up to 4 mins
    
    # Iterate
    # Pydub RMS calculation can be slow on long segments in a loop.
    # Faster approach: use array of samples.
    samples = np.array(analysis_audio.get_array_of_samples())
    if analysis_audio.sample_width == 2:
        samples = samples.astype(np.int16)
    elif analysis_audio.sample_width == 4:
        samples = samples.astype(np.int32)
    
    # samples is 1D array (mono)
    # Calculate window size in samples
    # beat_ms is ms. sample_rate is samples/sec -> samples/ms = sr/1000
    samples_per_ms = analysis_audio.frame_rate / 1000
    window_samples = int(ms_32_beats * samples_per_ms)
    step_samples = int(step * samples_per_ms)
    
    # Square the samples for RMS
    # Use float to avoid overflow
    sq_samples = samples.astype(np.float64) ** 2
    
    # Use convolution for sliding window sum of squares (faster than loop)
    # But for very large arrays, convolution can be heavy.
    # Let's use a simplified stride loop
    
    num_steps = (len(samples) - window_samples) // step_samples
    
    best_idx = 0
    max_energy = -1
    
    for i in range(0, len(samples) - window_samples, step_samples):
        # Calculate sum of squares in this window
        # To optimize, we could use integral image / prefix sum, but python loop with numpy slice is okay for audio < 5min
        # Actually, numpy slice is fast.
        current_energy = np.sum(sq_samples[i : i+window_samples])
        if current_energy > max_energy:
            max_energy = current_energy
            best_idx = i
            
    # Convert best_idx back to ms
    best_start_ms = int(best_idx / samples_per_ms) + start_offset
    
    # Align to nearest beat? 
    # Ideally yes, but "exact drop" might not be exactly on our beat grid if bpm varies slightly.
    # We will return the found time.
    
    return best_start_ms

def generate_clap(duration_ms=200):
    # Check if we have a custom clap sample
    sample_path = os.path.join(os.path.dirname(__file__), 'assets', 'clap.wav')
    
    if os.path.exists(sample_path):
        try:
            # Load the custom sample
            clap = AudioSegment.from_wav(sample_path)
            
            # If the sample is longer than the beat duration, we might want to trim it, 
            # but usually for a clap sample we want the full tail unless it's huge.
            # Let's just ensure it's not excessively long (e.g. > 1 sec)
            if len(clap) > 1000:
                clap = clap[:1000].fade_out(50)
                
            return clap
        except Exception as e:
            print(f"Error loading custom clap sample: {e}. Falling back to synthesis.")
    
    # Fallback: White noise with exponential decay
    noise = WhiteNoise().to_audio_segment(duration=duration_ms)
    # Simple envelope
    # We want a sharp attack and fast decay
    # Pydub fade_out is linear. We can simulate exp decay by multiple fades or just linear short.
    clap = noise.fade_out(duration_ms - 10)
    # High pass filter to make it crisp (approximate)
    clap = clap.high_pass_filter(800)
    return clap

def generate_fx_hit(duration_ms=1000):
    # A "Boom" or "Impact"
    # Sine wave sweep 100Hz -> 40Hz
    # Pydub doesn't have sweeps easily.
    # Let's just make a low sine ping + noise burst
    
    sine = Sine(60).to_audio_segment(duration=duration_ms).fade_out(duration_ms)
    noise = WhiteNoise().to_audio_segment(duration=duration_ms//2).fade_out(duration_ms//2).low_pass_filter(500)
    
    fx = sine.overlay(noise)
    return fx

def create_clap_loop(bpm, beats=16):
    beat_ms = 60000 / bpm
    total_duration = beats * beat_ms
    
    clap_sample = generate_clap(duration_ms=min(200, int(beat_ms/2)))
    
    # Create silence container
    loop = AudioSegment.silent(duration=int(total_duration))
    
    # Place claps on beats (every beat)
    # "Clap In ... 16 beats of claps only" -> usually means claps on the beat (1, 2, 3, 4)
    for i in range(beats):
        pos = int(i * beat_ms)
        loop = loop.overlay(clap_sample, position=pos)
        
    return loop

def process_track(vocals_path, inst_path, original_path, bpm):
    vocals = AudioSegment.from_mp3(vocals_path)
    inst = AudioSegment.from_mp3(inst_path)
    original = AudioSegment.from_mp3(original_path)
    
    beat_ms = 60000 / bpm
    ms_32_beats = 32 * beat_ms
    ms_16_beats = 16 * beat_ms
    
    # Find Drop
    drop_start = find_drop_start(inst, beat_ms)
    
    # Define Segments
    drop_voc = vocals[drop_start : drop_start + ms_32_beats]
    drop_inst = inst[drop_start : drop_start + ms_32_beats]
    outro_inst = drop_inst # Re-use drop instrumental as DJ outro
    
    clap_loop_16 = create_clap_loop(bpm, 16)
    fx_hit = generate_fx_hit()
    
    edits = []
    
    # Helper to clean up segments (fade in/out slightly to avoid clicks?)
    # For DJ edits, hard cuts on grid are preferred if quantized.
    
    # 1. Clap In (Updated Logic for 15h19 Request)
    # Goal: Intro 16 beats (Claps + Melody/Drums from Drop).
    # Then: Cut claps immediately when vocals start (anacrouse)
    # And continue with original track perfectly synced.
    
    # Step 1: Create the Intro Bed (Melody + Drums from Drop Inst)
    # We take the first 16 beats of the Drop Instrumental.
    intro_bed = drop_inst[:ms_16_beats]
    
    # Step 2: Overlay Claps on this bed
    intro_with_claps = intro_bed.overlay(clap_loop_16)
    
    # Step 3: Handle the transition to the full track.
    # The user says "Coupe les claps des le debut de l'anacrouse".
    # This implies we switch to the Original Track slightly *before* the theoretical drop point if there's a vocal pickup.
    # Since we can't easily detect the exact millisecond of the vocal pickup automatically without complex VAD,
    # we will use a safe crossfade window centered around the beat.
    # However, to simulate "cutting claps", we effectively switch from (Inst+Clap) to (Original)
    # The Original track at the drop point contains (Inst + Vocal).
    # If the vocal starts early, it is in the Original track before the drop point.
    
    # Strategy: 
    # Take Original track starting from (DropStart - 1 Beat).
    # Take IntroWithClaps (first 15 beats).
    # Crossfade them over the last beat?
    # Or just hard cut if perfectly quantized? Crossfade is safer for "anacrouse".
    
    # Let's try: Intro (16 beats) -> but we crossfade into Original starting at (DropStart - 500ms).
    # This effectively fades out the claps while fading in the vocal pickup.
    
    # Crossfade duration = 1 beat
    xfade_ms = int(beat_ms)
    
    # We shorten the intro by the crossfade amount so the total timing remains correct on the grid
    intro_part = intro_with_claps
    
    # We start the original track 'xfade_ms' BEFORE the drop start to catch the vocal pickup
    original_start_point = max(0, drop_start - xfade_ms)
    original_part = original[original_start_point:]
    
    # Now we blend. 
    # intro_part is 16 beats long.
    # We want the drop of original_part to align with end of intro_part.
    # original_part starts 1 beat before drop.
    # So we should append original_part to (intro_part - 1 beat).
    
    # Cut intro at 15 beats
    intro_cut = intro_part[:(ms_16_beats - xfade_ms)]
    
    # Append with crossfade? No, if we cut, we can just concat if we trust the grid.
    # But a crossfade smooths it.
    # Let's crossfade the last beat of Intro (with claps) against the pre-drop beat of Original (with vocals).
    # Actually, if we crossfade, we hear claps fading out + vocals fading in.
    # User said "Coupe les claps". Maybe a fast fade out of claps?
    
    # Revised approach for "Niquel":
    # 1. Intro = 16 beats of (DropInst + Claps).
    # 2. But at beat 16 (or slightly before), we switch to Original.
    # If we switch at Beat 15, we lose the last clap?
    # Let's switch at Beat 16 exactly, but using the Original's audio from Beat 16 (which is DropStart).
    # Wait, if vocal is BEFORE Beat 16 (anacrouse of drop), we need to hear it.
    # So we must switch to Original at Beat 15.5 or 15.
    
    # Let's switch at Beat 15 (last beat of the intro bar).
    # Intro: 15 beats of (DropInst + Claps).
    # Then: Original starting from (DropStart - 1 beat).
    # This preserves the vocal pickup (anacrouse) that happens in that last beat.
    # And since Original has the instrumental, the music continuity is kept.
    # The claps will stop at beat 15.
    
    intro_15 = intro_with_claps[:(ms_16_beats - int(beat_ms))]
    pickup_segment = original[max(0, drop_start - int(beat_ms)):]
    
    # Smooth transition (short crossfade 50ms to avoid click, but keep timing tight)
    clap_in = intro_15.append(pickup_segment, crossfade=50)
    
    # Add Outro
    clap_in = clap_in + outro_inst
    
    edits.append(("Clap In", clap_in))
    
    # 2. Acap In (Drop Vocal Only)
    # 16 beats acapella (first 16 beats of drop vocal) -> Drop -> End -> Outro
    # "strictly the lead vocal taken from the 32-beat drop section"
    # "starts with 16 beats of acapella only"
    acap_intro = drop_voc[:ms_16_beats]
    acap_in = acap_intro + original[drop_start:] + outro_inst
    edits.append(("Acap In", acap_in))
    
    # 3. Acap Out
    # "after the final drop... acapella outro must last 32 beats... no instrumental outro"
    # We will use the drop vocal as the outro acapella.
    # We need to append this to the FULL track (or from drop?).
    # "Create a version where... all instrumental elements are removed... leaving only the lead vocal"
    # This implies playing the track until the end, then ensuring the outro is Acapella.
    # But usually "Acap Out" replaces the original outro.
    # "After the final drop" -> Identify final drop? Or just append to the main part?
    # Let's assume we take the whole Original track, try to detect the end, and append the Acapella?
    # Or cut the original outro and replace with Acapella?
    # Safer: Original Track + Drop Acapella (32 beats).
    # "Acap Out... after the final drop... leaving only the lead vocal taken from the 32-beat drop section."
    # This phrasing suggests replacing the existing outro with the Drop Vocal.
    # But we don't know where the "final drop" ends in the original.
    # Strategy: Original Track + Drop Vocal (32 beats). (Effectively an Acapella Outro).
    acap_out_edit = original + drop_voc
    edits.append(("Acap Out", acap_out_edit))

    # 4. Intro (Instrumental Intro)
    # "32-beat instrumental intro only, no vocals."
    # "After the intro, introduce the vocal together with the instrumental, preserving the original drop structure."
    # "At the end... add 32-beat inst outro"
    # Logic: Instrumental version of first 32 beats -> Then Original (starting at 32 beats) -> ... -> Outro
    # This effectively removes vocals from the first 32 beats.
    intro_inst_32 = inst[:ms_32_beats]
    intro_edit = intro_inst_32 + original[ms_32_beats:] + outro_inst
    edits.append(("Intro", intro_edit))
    
    # 5. Short
    # "Short instrumental intro" -> Let's use 16 beats of Intro Inst.
    # "Only one break and one drop".
    # Structure: Intro(16) -> Break(before drop, 16) -> Drop(32) -> Outro(32)
    # Break logic: 16 beats before drop_start.
    break_start = max(0, drop_start - ms_16_beats)
    break_segment = original[break_start : drop_start] # Use original for break (might have build-up vocals)
    # Wait, "Short instrumental intro".
    intro_short = inst[:ms_16_beats]
    
    # What if break_start < 16 beats? 
    # If drop is early, we might overlap.
    # Assuming drop is at least 32 beats in.
    
    short_edit = intro_short + break_segment + original[drop_start : drop_start + ms_32_beats] + outro_inst
    edits.append(("Short", short_edit))
    
    # 6. Short Acap In
    # "16 beats of acapella only... using lead vocal from 32-beat drop"
    # "After acapella intro, include only one break and one drop"
    # "Add 32-beat inst outro"
    acap_intro_short = drop_voc[:ms_16_beats]
    short_acap_in = acap_intro_short + break_segment + original[drop_start : drop_start + ms_32_beats] + outro_inst
    edits.append(("Short Acap In", short_acap_in))
    
    # 7. Short Clap In
    # "16 beats of claps only" -> Modified: Claps + Melody
    # "After clap intro, include only one break and one drop"
    # Reuse intro_15 (which is effectively the Clap In section up to beat 15)
    # BUT we need to transition into the BREAK segment now, not the Drop directly.
    # The Short edit structure is: Intro -> Break -> Drop.
    # We should use the Clap Intro logic but transition into the Break.
    
    # However, 'intro_15' was built using Drop Instrumental.
    # If we transition to Break, it might be jarring if the break is quiet.
    # But usually Short Clap In means: Intro (DJ Tool) -> Breakdown -> Drop.
    # So we use the same Clap In block.
    
    # We need to construct it carefully to match the "Clap In" logic (cut claps at end).
    # Break usually starts quiet.
    # Let's transition from intro_15 to Break Segment.
    
    # Crossfade into break?
    short_clap_in = intro_15.append(break_segment, crossfade=50) + original[drop_start : drop_start + ms_32_beats] + outro_inst
    
    edits.append(("Short Clap In", short_clap_in))
    
    # 8. Acap In / Acap Out
    # "starts with 16 beats of acapella only"
    # "main body of the track remains unchanged"
    # "end of final drop... keep only the same drop vocal for 32 beats"
    # Logic: Acap Intro -> Original -> Drop Vocal Outro
    acap_in_out = drop_voc[:ms_16_beats] + original + drop_voc
    edits.append(("Acap In Acap Out", acap_in_out))
    
    # 9. Slam
    # "Single FX hit"
    # "Immediately after FX, start track directly from drop"
    # "Add 32-beat inst outro"
    slam_edit = fx_hit + original[drop_start:] + outro_inst
    edits.append(("Slam", slam_edit))
    
    # 10. Short Acap Out
    # "Shortened version with only one drop (32 beats)"
    # "After the drop, remove all inst... keep only lead vocal... for 32 beats"
    # Structure: (Intro/Break?) -> Drop -> Acapella Outro.
    # Spec says "Shortened version with only one drop". Doesn't explicitly mention intro/break.
    # But usually a track needs *some* buildup.
    # "Short" (v5) had intro/break.
    # Let's assume minimal structure: Break -> Drop -> Acapella Outro.
    short_acap_out = break_segment + original[drop_start : drop_start + ms_32_beats] + drop_voc
    edits.append(("Short Acap Out", short_acap_out))
    
    return edits

