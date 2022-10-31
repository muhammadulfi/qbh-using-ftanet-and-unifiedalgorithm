import os
import pickle
import numpy as np
import pandas as pd
import pretty_midi


ioacas_midilist  = './database/IOACAS_QBH/querymidi_list.txt'
mirqbsh_midilist = './database/MIR-QBSH/querymidi_list.txt'

# with open(mirqbsh_midilist,'r') as f:
#     midilist = f.readlines()

#     for midi in midilist:
#         # prepare path for midi and output file
#         midipath = midi.replace('\n', '')
#         splitted_path = midipath.split('\\')

#         filename = splitted_path[-1]
#         output_path = os.path.join('E:\Kuliah\S2\Thesis\source code thesis\database\MIR-QBSH\midi_note', filename +'.csv')
#         output_path = output_path.replace('.mid','')

#         end_time = -1
#         time = []
#         pitch = []
#         midi_data = pretty_midi.PrettyMIDI(midipath)
#         for instrument in midi_data.instruments:
#             for note in instrument.notes:
#                 if note.start != end_time and end_time != -1:
#                     for t in np.arange(end_time, round(note.start,2), 0.01):
#                         if t == time[-1]:
#                             time[-1] = round(t,2)
#                             pitch[-1] = 0
#                         else :
#                             time.append(round(t,2))
#                             pitch.append(0)
#                 for t in np.arange(round(note.start,2), round(note.end,2), 0.01):
#                     if end_time != -1 and t == time[-1]:
#                         time[-1] = round(t,2)
#                         pitch[-1] = 0
#                     else :
#                         time.append(round(t,2))
#                         pitch.append(note.pitch)
#                 end_time = round(note.end,2)

#         pd.DataFrame({'time':time, 'semitone':pitch}).to_csv(output_path, index=False)

#         # midi_file = open(output_path, 'wb')
#         # pickle.dump(notes, midi_file)
#         # midi_file.close()
#         # print(notes)
#         # break
# f.close()

with open(ioacas_midilist,'r') as f:
    midilist = f.readlines()
    
    for midi in midilist:
        # prepare path for midi and output file
        midipath = midi.replace('\n', '')
        splitted_path = midipath.split('\\')
        
        filename = splitted_path[-1]
        output_path = os.path.join('E:\Kuliah\S2\Thesis\source code thesis\database\IOACAS_QBH\midi_note', filename +'.csv')
        output_path = output_path.replace('.mid','')
        
        end_time = -1
        time = []
        pitch = []
        midi_data = pretty_midi.PrettyMIDI(midipath)
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                if note.start != end_time and end_time != -1:
                    for t in np.arange(end_time, round(note.start,2), 0.01):
                        if t == time[-1]:
                            time[-1] = round(t,2)
                            pitch[-1] = 0
                        else :
                            time.append(round(t,2))
                            pitch.append(0)
                for t in np.arange(round(note.start,2), round(note.end,2), 0.01):
                    if end_time != -1 and t == time[-1]:
                        time[-1] = round(t,2)
                        pitch[-1] = 0
                    else :
                        time.append(round(t,2))
                        pitch.append(note.pitch)
                end_time = round(note.end,2)
        
        pd.DataFrame({'time':time, 'semitone':pitch}).to_csv(output_path, index=False)

        # midi_file = open(output_path, 'wb')
        # pickle.dump(notes, midi_file)
        # midi_file.close()
f.close()