
# coding: utf-8

# In[1]:


from music21 import converter, instrument, note, chord, stream, pitch
import glob
import numpy as np
import pickle
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

import MusicDataExtractor as musicData


# In[2]:


def getNotes(songs_path):
    notes = []

    for file in glob.glob(songs_path):
        midi = converter.parse(file)
        try: 
            instruments = instrument.partitionByInstrument(midi)        
            for element in instruments.parts[0].recurse():
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except:
            print('No instrumental partition. Skipping...')
    return notes


# In[3]:


def prepare_input(notes, pitch_names, number_of_notes):  
    song_input = []
    
    # dictionary to map pitches to integers because NN perform better on integer-based numerical data
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))
    
    for i in range(0, len(notes) - generated_song_length, 1):
        sequence_in = notes[i:i + generated_song_length]
        song_input.append([note_to_int[char] for char in sequence_in])
    
    patterns_number = len(song_input)
    
    # reshape the input into a format compatible with LSTM layers
    normalized_input = np.reshape(song_input, (patterns_number, generated_song_length, 1))
    # normalize input
    normalized_input = normalized_input / float(number_of_notes)
    
    return (song_input, normalized_input)


# In[98]:


def create_network(song_input, number_of_notes):
    model = Sequential()
    model.add(LSTM(512, input_shape = (song_input.shape[1], song_input.shape[2]), return_sequences = True))
    model.add(Dropout(0.9))
    model.add(LSTM(512, return_sequences = True))
    model.add(Dropout(0.9))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.9))
    model.add(Dense(number_of_notes))
    model.add(Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop')
    
    model.load_weights('weights.hdf5')
    
    return model


# In[99]:


def generate_notes(model, song_input, pitch_names, number_of_notes):
    start_note = np.random.randint(0, len(song_input) - 1)
    int_to_note = dict ((number, note) for number, note in enumerate(pitch_names))
    
    pattern = song_input[start_note]
    predicted_song_output = []
    
    for note_index in range(100):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(number_of_notes)
        
        prediction = model.predict(prediction_input, verbose = 0)
       
        index = np.argmax(prediction) 
        
        result = int_to_note[index]
        predicted_song_output.append(result)
        
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        
    return predicted_song_output


# In[100]:


def create_song(predicted_song_output, song_name):
    offset = 0
    output_notes = []
    
    for pattern in predicted_song_output:
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
    midi_stream.write('midi', fp = "output.mid")


# In[101]:


target_songs_path = "midi_songs/*.mid"

notes = []
#taking the notes from file in stead of using getNotes(songs_path)
with open('data/notes', 'rb') as filepath:
    notes = pickle.load(filepath)


# In[102]:


# pitch names
pitch_names = sorted(set(item for item in notes))

# get amount of pitch names
number_of_notes = len(set(notes))

#lenght of generated song
generated_song_length = 100


# In[103]:


song_input, normalized_input = prepare_input(notes, pitch_names, number_of_notes)


# In[104]:


model = create_network(normalized_input, number_of_notes)


# In[105]:


predicted_song_output = generate_notes(model, song_input, pitch_names, number_of_notes)


# In[106]:


output_song_name = "output.mid"
musicData.create_song(predicted_song_output, output_song_name)
output_song_generated = converter.parse(output_song_name) 
musicData.print_song(output_song_generated)


# In[70]:


folder_path = os.getcwd()
output_song_path = folder_path + output_song_name

training_sourche_path = folder_path + "\\midi_songs\\"


# In[109]:


target_songs_names = glob.glob(target_songs_path)


# In[110]:


target_songs_names = dict([(x,0) for x in target_songs_names])


# In[ ]:


musicData.compare_generated_song_with_sources(output_song_path, training_sourche_path)

