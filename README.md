## About
This project trains a *recurrent neural network* to learn the various components and temporal attributes of a list of music pieces, including the various ***notes/chords/tempos/rests*** that occur, the ***duration*** for which they last and their ***offset*** relative to the last note/chord/tempo/rest. Therefore, given an initial set of these attributes, the trained model will be able to compose a new music piece by predicting the exact set of notes/chords/tempos/rests, the instances at which they initiate and the durations for which they last.

The project uses <a href = "https://web.mit.edu/music21/"> *Music21* </a> library which is an easy-to-use Python toolkit used for computer-aided musicology. It allows us to teach the fundamentals of music theory, generate music examples and study music. The toolkit provides a simple interface to acquire the musical notation of *MIDI* files. Additionally, it allows us to create *Note*, *Chord*, *Tempo* etc objects so that we can make our own *MIDI* files easily.

## Training
Though a *music generation model* could be trained on a collection of music pieces, yet for a more precise evaluation it was trained on single music pieces and then the generated music was compared with the original music. One model was trained on the music *Highwind Takes To The Skies* (*MUSIC_ORIGINAL/2.mid*) for *95 epochs* at a learning rate of *0.001*. *1599* musical components were extracted from *Highwind Takes To The Skies*, that with a sequence length of *100* caused a training dataset of size *1499*. Training took almost *1 hour*.

&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120914975-e6036480-c6be-11eb-938f-d440b13a7265.png" width = '400' height = '320'>
&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120914983-ea2f8200-c6be-11eb-841e-4d69ee9c3629.png" width = '400' height = '320'>
&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120914984-eac81880-c6be-11eb-9718-29497bd005fb.png" width = '400' height = '320'>
&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120914981-e996eb80-c6be-11eb-956a-c14d75916b6d.png" width = '400' height = '320'>

( Another model was trained on the music *Final Fantasy VIII 0fithos*, *MUSIC_ORIGINAL/1.mid* )

