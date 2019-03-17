
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import pandas as pd
import numpy as np
import pickle
import pprint
import os
import glob

from gensim.models import Word2Vec
from music21 import converter, instrument, note, chord, stream, pitch, roman
from multiprocessing import Pool


# In[2]:


def extract_notes(song_part):
    parent_element = []
    ret = []
    for nt in song_part.flat.notes:
        if isinstance(nt, note.Note):
            ret.append(max(0.0, nt.pitch.ps))
            parent_element.append(nt)
        elif isinstance(nt, chord.Chord):
            for pitch in nt.pitches:
                ret.append(max(0.0, pitch.ps))
                parent_element.append(nt)
    
    return ret, parent_element


# In[3]:


def print_song(song):
    fig = plt.figure(figsize = (12, 5))
    ax = fig.add_subplot(1, 1, 1)
    minPitch = pitch.Pitch('C10').ps
    maxPitch = 0
    xMax = 0
    
    for i in range(len(song.parts)):
        top = song.parts[i].flat.notes
        y, parent_element = extract_notes(top)
        if (len(y) < 1): 
            continue
            
        x = [n.offset for n in parent_element]
        ax.scatter(x, y, alpha = 0.6, s = 7)
        
        aux = min(y)
        if (aux > minPitch):
            minPitch = aux           
        aux = max(y)
        if (aux > maxPitch):
            maxPitch = aux           
        aux = max(x)
        if (aux > xMax):
            xMax = aux
            
    for i in range(1, 10):
        linePitch = pitch.Pitch('C{0}'.format(i)).ps
        if (linePitch > minPitch and linePitch < maxPitch):
            ax.add_line(mLine2D([0, xMax], [linePitch, linePitch], color = 'red', alpha = 0.1))
                
    plt.ylabel("Note index (each octave has 12 notes)")
    plt.xlabel("Number of quarter notes (beats)")
    plt.title('Piano song reprezentation')
    plt.show()


# In[4]:


def create_song(prediction_output, output_song_name):
    offset = 0
    output_notes = []
    
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
            
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp = output_song_name)


# In[5]:


def vectorize_harmony(model, song):
    word_vecs = []
    for word in song:
        try:
            vec = model[word]
            word_vecs.append(vec)
        except KeyError:
            pass
        
    return np.mean(word_vecs, axis = 0)


# In[6]:


def cosine_similarity(vecA, vecB):
    csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
    if np.isnan(np.sum(csim)):
        return 0
    return csim


# In[7]:


def calculate_similarity_aux(df, model, source_name, target_names=[], threshold=0):
    source_harmo = df[df["midi_name"] == source_name]["harmonic_reduction"].values[0]
    source_vec = vectorize_harmony(model, source_harmo)  
    results = []
    for name in target_names:
        target_harmo = df[df["midi_name"] == name]["harmonic_reduction"].values[0]
        if (len(target_harmo) == 0):
            continue
            
        target_vec = vectorize_harmony(model, target_harmo)  
        sim_score = cosine_similarity(source_vec, target_vec)
        if sim_score > threshold:
            results.append({
                'score' : sim_score,
                'name' : name
            })
                
    # Sort results by score in desc order
    results.sort(key=lambda k : k['score'] , reverse=True)
    return results

def calculate_similarity(df, model, source_name, target_prefix, threshold=0):
    source_midi_names = df[df["midi_name"] == source_name]["midi_name"].values
    if (len(source_midi_names) == 0):
        print("Invalid source name")
        return
    
    source_midi_name = source_midi_names[0]
    
    target_midi_names = df[df["midi_name"].str.startswith(target_prefix)]["midi_name"].values  
    if (len(target_midi_names) == 0):
        print("Invalid target prefix")
        return
    
    return calculate_similarity_aux(df, model, source_midi_name, target_midi_names, threshold)


# In[8]:


def note_count(measure, count_dict):
    bass_note = None
    for chord in measure.recurse().getElementsByClass('Chord'):
        # All notes have the same length of its chord parent.
        note_length = chord.quarterLength
        for note in chord.pitches:          
            # If note is "C5", note.name is "C". We use "C5"
            # style to be able to detect more precise inversions.
            note_name = str(note) 
            if (bass_note is None or bass_note.ps > note.ps):
                bass_note = note
                
            if note_name in count_dict:
                count_dict[note_name] += note_length
            else:
                count_dict[note_name] = note_length
        
    return bass_note
                
def simplify_roman_name(roman_numeral):
    # Chords can get nasty names as "bII#86#6#5",
    # in this method we try to simplify names, even if it ends in
    # a different chord to reduce the chord vocabulary and display
    # chord function clearer.
    ret = roman_numeral.romanNumeral
    inversion_name = None
    inversion = roman_numeral.inversion()
    
    # Checking valid inversions.
    if ((roman_numeral.isTriad() and inversion < 3) or
            (inversion < 4 and
                 (roman_numeral.seventh is not None or roman_numeral.isSeventh()))):
        inversion_name = roman_numeral.inversionName()
        
    if (inversion_name is not None):
        ret = ret + str(inversion_name)
        
    elif (roman_numeral.isDominantSeventh()): ret = ret + "M7"
    elif (roman_numeral.isDiminishedSeventh()): ret = ret + "o7"
    return ret
                
def harmonic_reduction(midi_file):
    ret = []
    temp_midi = stream.Score()
    temp_midi_chords = midi_file.chordify()
    temp_midi.insert(0, temp_midi_chords)    
    music_key = temp_midi.analyze('key')
    max_notes_per_chord = 4   
    for m in temp_midi_chords.measures(0, None): # None = get all measures.
        if (type(m) != stream.Measure):
            continue
        
        # Here we count all notes length in each measure,
        # get the most frequent ones and try to create a chord with them.
        count_dict = dict()
        bass_note = note_count(m, count_dict)
        if (len(count_dict) < 1):
            ret.append("-") # Empty measure
            continue
        
        sorted_items = sorted(count_dict.items(), key=lambda x:x[1])
        sorted_notes = [item[0] for item in sorted_items[-max_notes_per_chord:]]
        measure_chord = chord.Chord(sorted_notes)
        
        # Convert the chord to the functional roman representation
        # to make its information independent of the music key.
        roman_numeral = roman.romanNumeralFromChord(measure_chord, music_key)
        ret.append(simplify_roman_name(roman_numeral))
        
    return ret


# In[9]:


def get_file_name(link):
    filename = link.split('/')[::-1][0]
    return filename


# In[10]:


def open_midi(midi_path):
    # There is an one-line method to read MIDIs
    # but to remove the drums we need to manipulate some
    # low level MIDI events.
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
           

    return midi.translate.midiFileToStream(mf)


# In[11]:


def process_single_file(midi_param):
    try:
        game_name = midi_param
        midi_path = midi_param
        midi_name = get_file_name(midi_path)
        midi = converter.parse(midi_path)
        return (
            midi.analyze('key'),
            game_name,
            harmonic_reduction(midi),
            midi_name)
    except Exception as e:
        print("Error on {0}".format(midi_name))
        print(e)
        return None


# In[12]:


def create_results(folder_path, output_song_path):
    results = []
    for midi_name in os.listdir(folder_path):
        res = process_single_file(folder_path + midi_name)
        results.append(res) 
    
    resz = process_single_file(output_song_path)
    results.append(resz) 
    
    return results


# In[13]:


def create_midi_dataframe(results):
    key_signature_column = []
    game_name_column = []
    harmonic_reduction_column = []
    midi_name_column = []
    for result in results:
        if (result is None):
            continue
            
        key_signature_column.append(result[0])
        game_name_column.append(result[1])
        harmonic_reduction_column.append(result[2])
        midi_name_column.append(result[3])
    
    d = {'midi_name': midi_name_column,
         'game_name': game_name_column,
         'key_signature' : key_signature_column,
         'harmonic_reduction': harmonic_reduction_column}
    return pd.DataFrame(data=d)


# In[14]:


def compare_generated_song_with_sources(output_song_path, training_sourche_path):
    results = create_results(training_sourche_path, output_song_path)
    dataframe = create_midi_dataframe(results)
    model = Word2Vec(dataframe["harmonic_reduction"], min_count=2, window=4)
    
    pp = pprint.PrettyPrinter(width=41, compact=True)
    pp.pprint(calculate_similarity(dataframe, model, output_song_path, training_sourche_path))

