A simulation model that I've used for my master's thesis titled "A broadcast model of spread of digital music composition among artificial audience".
Thesis is written for graduation from Cognitive Science Department of Middle East Technical University.

Python libraries required for the execution of the simulation can be found at libraries.py file in this repository.

To execute simulation with broadcasters:

  do_broadcast = True

To execute simulation without broadcasters:

  do_broadcast = False

​You should modify the following parameters to experiment with the model:
  
  num_composers = Number of composers
  num_listeners = Number of listeners
  num_broadcasters = Number of broadcasters
  num_iterations = Number of iterations
  playlist_size = Playlist size
  num_semitones = Number of musical pitches
  melody_length = Length of compositions
  success_threshold = Success threshold
  learning_rate = Learning rate
  decay_rate = Decay rate

You should note that when running the simulation without broadcasters, "num_broadcasters", "playlist_size" and "decay_rate" parameters should be set to "None" since these parameters are only related to broadcasters.
