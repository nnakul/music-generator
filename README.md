## About
This project trains a *recurrent neural network* to learn the various components and temporal attributes of a list of music pieces, including the various ***notes/chords/tempos/rests*** that occur, the ***duration*** for which they last and their ***offset*** relative to the last note/chord/tempo/rest. Therefore, given an initial set of these attributes, the trained model will be able to compose a new music piece by predicting the exact set of notes/chords/tempos/rests, the instances at which they initiate and the durations for which they last.

The project uses <a href = "https://web.mit.edu/music21/"> *Music21* </a> library which is an easy-to-use Python toolkit used for computer-aided musicology. It allows us to teach the fundamentals of music theory, generate music examples and study music. The toolkit provides a simple interface to acquire the musical notation of *MIDI* files. Additionally, it allows us to create *Note*, *Chord*, *Tempo* etc objects so that we can make our own *MIDI* files easily.

## Training



