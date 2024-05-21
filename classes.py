@dataclasses.dataclass
class Parameters:
  num_composers: int
  num_listeners: int
  num_broadcasters: int
  num_iterations: int
  playlist_size: int
  num_semitones: int
  melody_length: int
  success_threshold: float
  learning_rate: float
  decay_rate: float


@dataclasses.dataclass
class Rating:
  listener_id: str
  composition_id: str
  success: bool
  iteration: int


class Composition:

  def __init__(self, composer_id: str, melody: tuple[int, ...]) -> None:
    self.id = uuid.uuid4()
    self.composer_id = composer_id
    self.melody = melody
    self.ratings = []
    self.is_forgotten = False

  def __str__(self) -> str:
    return (
        f"Composition {self.id}:"
        f" Melody: {self.melody},"
        f" Ratings: {self.ratings}"
        f" Forgotten: {self.is_forgotten}"
    )

  def add(self, rating: Rating) -> None:
    self.ratings.append(rating)

  def note_transitions(self) -> tuple[tuple[int, int]]:
    return tuple(nltk.ngrams(self.melody, 2))

  def forget(self) -> None:
    self.is_forgotten = True


class CompositionPool:

  def __init__(self) -> None:
    self.pool = {}

  def __str__(self) -> str:
    return "\n".join(str(c) for c in self.pool.values())

  def add(self, composition: Composition) -> None:
    self.pool[composition.id] = composition

  def get(self, composition_id: str) -> Composition:
    return self.pool[composition_id]

  def get_unforgotten(self) -> tuple[int, ...]:
    return tuple(k for k, c in self.pool.items() if not c.is_forgotten)

  def sample(self, sample_size: int) -> list[Composition]:
    unforgotten = self.get_unforgotten()
    return random.sample(unforgotten, sample_size)

  def ratings_of(self, iteration: int) -> list[Rating]:
    sought_ratings = []

    for composition in self.pool.values():
      ratings = (r for r in composition.ratings if r.iteration == iteration)
      sought_ratings.extend(ratings)

    return sought_ratings

  def decay(self, decay_rate: float) -> None:
    unforgotten = self.get_unforgotten()
    decay_count = int(len(unforgotten) * decay_rate)
    to_forget = set(random.sample(unforgotten, decay_count))

    for composition_id in to_forget:
      self.get(composition_id).forget()


class TransitionTable:

  def __init__(self, num_semitones: int) -> None:
    self._values = self._populate(num_semitones)

  def _populate(self, num_semitones: int) -> dict[int, dict[int]]:
    # 0 is a 'Rest'. From 1 to 12 are notes from C to B, each next integer
    # representing each next semitone.
    semitones = list(range(num_semitones))
    table = dict()

    for prev_note in semitones:
      # Initialize preference scores for each combination of previous and next
      # note pairs as 0.0.
      table[prev_note] = dict(((k, 0.0) for k in semitones))

    return table

  def semitones(self) -> tuple[int, ...]:
    return tuple(self._values.keys())

  def preferred_next_note(self, prev_note: int) -> int:
    next_notes_scores = self._values[prev_note]
    max_score = max(next_notes_scores.values())
    candidates = [n for n, s in next_notes_scores.items() if s == max_score]
    return random.choice(candidates)

  def preference_score(self, prev_note: int, next_note: int) -> float:
    return self._values[prev_note][next_note]

  def add_to_preference_score(
      self, prev_note: int, next_note: int, addition: float
  ) -> None:
    self._values[prev_note][next_note] += addition

  def normalize_preference_scores(self, factor: float) -> None:
    transitions = itertools.product(self.semitones(), self.semitones())

    for prev_note, next_note in transitions:
      normalized = self.preference_score(prev_note, next_note) / factor
      self._values[prev_note][next_note] = normalized

  def update_preferences(
      self,
      composition: Composition,
      success: bool,
      learning_rate: float,
  ) -> None:
    for prev_note, next_note in composition.note_transitions():
      gradient = learning_rate if success else (-1 * learning_rate)
      self._values[prev_note][next_note] += gradient

  def to_csv(self) -> None:
    semitones = self.semitones()
    header = "," + ",".join(str(s) for s in semitones)
    print(header)

    for prev_note in semitones:
      scores = (f"{s:.4f}" for s in self._values[prev_note].values())
      row = f"{prev_note}," + ",".join(scores)
      print(row)


class Composer:

  def __init__(self, num_semitones: int, learning_rate: float) -> None:
    self.id = uuid.uuid4()
    self.transition_table = TransitionTable(num_semitones)
    self.learning_rate = learning_rate

  def create_composition(self, length: int) -> Composition:
    # Assing a random first note.
    melody = [random.choice(self.transition_table.semitones())]

    # Iteratively add most probable next notes until we reach the desired melody
    # length.
    while len(melody) < length:
      melody.append(self.transition_table.preferred_next_note(melody[-1]))

    return Composition(self.id, melody)

  def update_preferences(self, composition: Composition, success: bool) -> None:
    self.transition_table.update_preferences(
        composition,
        success,
        self.learning_rate,
    )


class Broadcaster:

  def __init__(self) -> None:
    self.id = uuid.uuid4()
    self.playlist = []

  def update_playlist(
      self,
      composition_pool: CompositionPool,
      playlist_size: int,
  ) -> None:
    self.playlist = composition_pool.sample(playlist_size)


class Listener:

  def __init__(
      self,
      num_semitones: int,
      success_threshold: float,
      learning_rate: float
  ) -> None:
    self.id = uuid.uuid4()
    self.transition_table = TransitionTable(num_semitones)
    self.success_threshold = success_threshold
    self.learning_rate = learning_rate

  def rate(self, composition: Composition, iteration: int) -> Rating:
    transitions = composition.note_transitions()
    overall_score = sum(
        self.transition_table.preference_score(p, n) for p, n in transitions
    )
    success = True if overall_score >= self.success_threshold else False
    rating = Rating(self.id, composition.id, success, iteration)
    composition.add(rating)
    return rating

  def update_preferences(self, composition: Composition) -> None:
    self.transition_table.update_preferences(
        composition,
        True,
        self.learning_rate,
    )
