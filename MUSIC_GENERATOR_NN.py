
import torch
import torch.nn as nn
import numpy as np
from music21 import note, chord, tempo, duration
from music21 import converter, instrument, stream
from random import shuffle

def LoadMusicComponents ( file_paths ) :
    global NOTES, DURATIONS, OFFSETS, LENGTHS
    NOTES = list()
    DURATIONS = list()
    OFFSETS = list()
    LENGTHS = list()
    
    for file in file_paths :
        if not file.endswith('.mid') : continue
        try : music_file = converter.parse(file)
        except : continue
            
        last_offset = 0.0
        itertable_notes_and_chords = None
        parts = instrument.partitionByInstrument(music_file)
        length = 0
        
        if parts : itertable_notes_and_chords = parts.parts[0].recurse()
        else : itertable_notes_and_chords = music_file.flat.notes
            
        for element in itertable_notes_and_chords :
            if isinstance(element, note.Note) :
                DURATIONS.append( round(eval(str(element.duration.quarterLength)), 4) )
                OFFSETS.append( round(element.offset - last_offset, 4) )
                last_offset = element.offset
                NOTES.append( str(element.pitch) )
                length += 1
            elif isinstance(element, chord.Chord) :
                DURATIONS.append( round(eval(str(element.duration.quarterLength)), 4) )
                OFFSETS.append( round(element.offset - last_offset, 4) )
                last_offset = element.offset
                NOTES.append('$'.join(str(n) for n in element.normalOrder))
                length += 1
            elif isinstance(element, note.Rest) :
                DURATIONS.append( round(eval(str(element.duration.quarterLength)), 4) )
                OFFSETS.append( round(element.offset - last_offset, 4) )
                last_offset = element.offset
                NOTES.append('rest')
                length += 1
            elif isinstance(element, tempo.MetronomeMark) :
                DURATIONS.append( round(eval(str(element.referent.quarterLength)), 4) )
                OFFSETS.append( round(element.offset - last_offset, 4) )
                last_offset = element.offset
                info = [ str(element.text), str(element.number), str(element.parentheses) ]
                NOTES.append('$'.join(v for v in info))
                length += 1
        
        LENGTHS.append(length)

def CreateLookupTables ( ) :
    global ID_TO_NOTE, NOTE_TO_ID, ID_TO_DURATION, DURATION_TO_ID, ID_TO_OFFSET, OFFSET_TO_ID
    ID_TO_NOTE = list(set(NOTES))
    ID_TO_NOTE.sort()
    NOTE_TO_ID = { n : idd for idd, n in enumerate(ID_TO_NOTE) }
    
    ID_TO_DURATION = list(set(DURATIONS))
    ID_TO_DURATION.sort()
    DURATION_TO_ID = { dur : idd for idd, dur in enumerate(ID_TO_DURATION) }
    
    ID_TO_OFFSET = list(set(OFFSETS))
    ID_TO_OFFSET.sort()
    OFFSET_TO_ID = { off : idd for idd, off in enumerate(ID_TO_OFFSET) }

def ConstructDataset ( seq_length ) :
    global SEQ_LENGTH, NOTES_ENC, DURATIONS_ENC, OFFSETS_ENC
    SEQ_LENGTH = seq_length
    
    NOTES_ENC = list()
    DURATIONS_ENC = list()
    OFFSETS_ENC = list()
    for note in NOTES : NOTES_ENC.append(NOTE_TO_ID[note])
    for dur in DURATIONS : DURATIONS_ENC.append(DURATION_TO_ID[dur])
    for off in OFFSETS : OFFSETS_ENC.append(OFFSET_TO_ID[off])
    
    inputs = list()
    output_notes = list()
    output_durations = list()
    output_offsets = list()
    
    start = 0
    for length in LENGTHS :
        end = start + length
        data_notes = NOTES_ENC[start:end]
        data_offsets = OFFSETS_ENC[start:end]
        data_durations = DURATIONS_ENC[start:end]
        data = list(zip(data_notes, data_offsets, data_durations))
        for first in range(length-seq_length) :
            last = first + seq_length
            input_sequence = data[first:last]
            output_comp = data[last]
            inputs.append(input_sequence)
            output_notes.append(output_comp[0])
            output_offsets.append(output_comp[1])
            output_durations.append(output_comp[2])
        start += length
    
    global DATASET
    DATASET = list(zip(inputs, output_notes, output_offsets, output_durations))

def MakeBatches ( batch_size ) :
    shuffle(DATASET)
    batches = list()
    for start in range(0, len(DATASET), batch_size) :
        end = start + batch_size
        batch, notes, offsets, durations = list(zip(*DATASET[start:end]))
        notes, offsets, durations = torch.Tensor(notes), torch.Tensor(offsets), torch.Tensor(durations)
        notes, offsets, durations = notes.long(), offsets.long(), durations.long()
        labels = (notes, offsets, durations)
        batch = torch.Tensor(batch)
        batches.append((batch, labels))
    return batches

class MusicGenerator ( nn.Module ) :
    def __init__ ( self , hidden_size ) :
        super(MusicGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 3
        self.lstm = nn.LSTM(input_size = 3, hidden_size = hidden_size, 
                            num_layers = self.num_layers, dropout = 0.5, batch_first = True)
        self.dense_notes = nn.Linear(hidden_size, len(ID_TO_NOTE))
        self.dense_offsets = nn.Linear(hidden_size, len(ID_TO_OFFSET))
        self.dense_durations = nn.Linear(hidden_size, len(ID_TO_DURATION))
    
    def forward ( self , inputs ) :
        h0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size)
        output, _ = self.lstm(inputs, (h0, c0))
        output = output[:, -1, :]
        notes_dist = self.dense_notes(output)
        offsets_dist = self.dense_offsets(output)
        durations_dist = self.dense_durations(output)
        return notes_dist, offsets_dist, durations_dist

def MakeModel ( learning_rate ) :
    global model, optimizer, criterion_notes, criterion_offsets, criterion_durations
    model = torch.load('MUSIC_GENERATOR_1')
    criterion_notes = nn.CrossEntropyLoss()
    criterion_offsets = nn.CrossEntropyLoss()
    criterion_durations = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

def TrainModel ( batch_size, total_epochs ) :
    global model, optimizer, criterion_notes, criterion_offsets, criterion_durations
    L = len(DATASET)
    for epoch in range(total_epochs) :
        print('\n EPOCH {} STARTED '.format(epoch+1))
        total_loss = 0.0
        total_n, total_o, total_d = 0.0, 0.0, 0.0
        batches = MakeBatches(batch_size)
        
        for step, (batch, labels) in enumerate(batches) :
            true_note, true_offset, true_duration = labels
            notes_dist, offsets_dist, durations_dist = model(batch)
            loss_notes = criterion_notes(notes_dist, true_note)
            loss_offsets = criterion_offsets(offsets_dist, true_offset)
            loss_durations = criterion_durations(durations_dist, true_duration)
            loss = loss_notes + loss_offsets + loss_durations
            
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # The following is an alternative loss function where the one classification out of the three that
            # performs the worst (with maximum loss) is given the consideration. This proved to be effective because
            # the number of classes of types is much more than the number of classes of durations or relative offsets.
            # Keeping loss equal to the sum of the three classification losses in this case initially causes a lot of
            # bias by just improving the classification for the attributes that have lesser number of target classes.
            # After few epochs, change the loss function and then once the accuracies of all the three classifications
            # reaches say 95%, you may change the loss function back to the "sum of three losses".
            
            # Check the accuracy progresses in README.md to observed the impact of changing loss functions.
            
            # if ( loss_notes > loss_offsets and loss_notes > loss_durations ) :
            #     optimizer.zero_grad()
            #     loss_notes.backward()
            #     optimizer.step()
            # elif ( loss_offsets > loss_durations ) :
            #     optimizer.zero_grad()
            #     loss_offsets.backward()
            #     optimizer.step()
            # else :
            #     optimizer.zero_grad()
            #     loss_durations.backward()
            #     optimizer.step()
            
            notes_pred = torch.argmax(notes_dist, dim = 1)
            offsets_pred = torch.argmax(offsets_dist, dim = 1)
            durations_pred = torch.argmax(durations_dist, dim = 1)
            
            t_n = (notes_pred == true_note).sum()
            total_n += t_n
            acc_n = t_n / batch.shape[0]
            
            t_o = (offsets_pred == true_offset).sum()
            total_o += t_o
            acc_o = t_o / batch.shape[0]
            
            t_d = (durations_pred == true_duration).sum()
            total_d += t_d
            acc_d = t_d / batch.shape[0]
            
            print('    STEP : {:2d} | LOSS : {:.6f} | ACC_N : {:.4f} | ACC_O : {:.4f} | ACC_D : {:.4f} '.format(step+1, loss, 
                                                                                            acc_n, acc_o, acc_d))
        acc_n, acc_o, acc_d = total_n / L, total_o / L, total_d / L
        loss = total_loss / len(batches)
        print(' EPOCH LOSS : {:.6f} | ACC_N : {:.4f} | ACC_O : {:.4f} | ACC_D : {:.4f} '.format(loss, acc_n, acc_o, acc_d))

def SaveModel ( destination ) :
    global model
    torch.save(model, destination)

def PredictMusicComponents ( count ) :
    global PREDICTED_MUSIC_COMPS
    PREDICTED_MUSIC_COMPS = list(zip(NOTES[:SEQ_LENGTH], OFFSETS[:SEQ_LENGTH], DURATIONS[:SEQ_LENGTH]))
    input_seq = list(zip(NOTES_ENC[:SEQ_LENGTH], OFFSETS_ENC[:SEQ_LENGTH], DURATIONS_ENC[:SEQ_LENGTH]))
    L = count - SEQ_LENGTH
    for i in range(L) :
        pred = model(torch.Tensor([input_seq]))
        pred_note_id, pred_off_id, pred_dura_id = torch.argmax(pred[0]), torch.argmax(pred[1]), torch.argmax(pred[2])
        next_note, next_off, next_dura = ID_TO_NOTE[pred_note_id], ID_TO_OFFSET[pred_off_id], ID_TO_DURATION[pred_dura_id]
        PREDICTED_MUSIC_COMPS.append((next_note, next_off, next_dura))
        input_seq = input_seq[1:]
        input_seq.append((pred_note_id, pred_off_id, pred_dura_id))

def GenerateAndSaveMusic ( file_loc ) :
    curr_time = 0.0
    global GENERATED_MUSIC_COMPS, PREDICTED_MUSIC_COMPS
    GENERATED_MUSIC_COMPS = []
    for i, (pred_note, pred_off, pred_dura) in enumerate(PREDICTED_MUSIC_COMPS) :
        if pred_note.isdigit() :
            new_note = note.Note(int(pred_note))
            new_note.storedInstrument = instrument.Piano()
            new_chord = chord.Chord([new_note])
            new_chord.offset = curr_time + pred_off
            new_chord.duration = duration.Duration(pred_dura)
            GENERATED_MUSIC_COMPS.append(new_chord)
            curr_time += pred_off
        
        elif '$' in pred_note and not 'True' in pred_note and not 'False' in pred_note :
            notes_in_chord = pred_note.split('$')
            notes = []
            for current_note in notes_in_chord :
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = curr_time + pred_off
            new_chord.duration = duration.Duration(pred_dura)
            GENERATED_MUSIC_COMPS.append(new_chord)
            curr_time += pred_off
        
        elif '$' in pred_note :
            info1, info2, info3 = pred_note.split('$')
            text, number, paren = info1, eval(info2), bool(info3)
            if text == 'None' : text = None
            new_tempo = tempo.MetronomeMark()
            new_tempo.text = text
            new_tempo.number = number
            new_tempo.parentheses = paren
            new_tempo.offset = curr_time + pred_off
            new_tempo.referent = duration.Duration(pred_dura)
            GENERATED_MUSIC_COMPS.append(new_tempo)
            curr_time += pred_off
        
        elif pred_note == 'rest' :
            new_rest = note.Rest()
            new_rest.storedInstrument = instrument.Piano()
            new_rest.offset = curr_time + pred_off
            new_rest.duration = duration.Duration(pred_dura)
            GENERATED_MUSIC_COMPS.append(new_rest)
            curr_time += pred_off
        
        else :
            new_note = note.Note(pred_note)
            new_note.storedInstrument = instrument.Piano()
            new_note.offset = curr_time + pred_off
            new_note.duration = duration.Duration(pred_dura)
            GENERATED_MUSIC_COMPS.append(new_note)
            curr_time += pred_off
            GENERATED_MUSIC_COMPS.append(new_note)

    midi_stream = stream.Stream(GENERATED_MUSIC_COMPS)
    midi_stream.write('midi', fp = file_loc)

if __name__ == '__main__' :
    LoadMusicComponents(['MUSIC_ORIGINAL/1.mid'])
    CreateLookupTables()
    ConstructDataset(seq_length = 100)
    MakeModel(learning_rate = 0.001)
    TrainModel(batch_size = 64, total_epochs = 100)
    SaveModel('MODELS/MUSIC_GENERATOR_1')
    PredictMusicComponents(count = 2000)
    GenerateAndSaveMusic('MUSIC_GENERATED/1.mid')
