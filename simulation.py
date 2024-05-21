def simulate_society(params: Parameters, broadcast: bool) -> CompositionPool:
  composers, listeners = {}, {}

  for _ in range(params.num_composers):
    composer = Composer(params.num_semitones, params.learning_rate)
    composers[composer.id] = composer

  for _ in range(params.num_listeners):
    listener = Listener(
        params.num_semitones,
        params.success_threshold,
        params.learning_rate,
    )
    listeners[listener.id] = listener

  # The composition pool is not cleared out in between iterations. Hence, a
  # composition that is generated in previous iterations might be broadcasted
  # at a given time in the society.
  composition_pool = CompositionPool()

  if broadcast:
    broadcasters = [Broadcaster() for _ in range(params.num_broadcasters)]

    for iteration in range(params.num_iterations):
      # Each iteration composers contribute towards the pool with a new
      # composition.
      for composer in composers.values():
        composition = composer.create_composition(params.melody_length)
        composition_pool.add(composition)

      # Sample a subset of the compositions from the pool for each broadcaster
      # and use that sampled set of compositions as the playlist of the
      # broadcaster for this iteration.
      for broadcaster in broadcasters:
        broadcaster.update_playlist(composition_pool, params.playlist_size)

      for listener in listeners.values():
        # Pair the listener with a random broadcaster.
        broadcaster = random.choice(broadcasters)

        # Listener listens the compositions in the playlist of broadcaster and
        # rates each of them.
        for composition_id in broadcaster.playlist:
          listener.rate(composition_pool.get(composition_id), iteration)

      # Now, update the transition scores of each composer and listener that
      # contributed in the iteration. First, gather the ratings provided by the
      # listeners in this iteration.
      iteration_ratings = composition_pool.ratings_of(iteration)

      for rating in iteration_ratings:
        composition = composition_pool.get(rating.composition_id)
        listener = listeners[rating.listener_id]
        composer = composers[composition.composer_id]

        # Listeners always increase the preferences for the note transitions that
        # are observed in the compositions that they have heard in the iteration,
        # modelling the increase in familiarity with those bi-gram transitions.
        listener.update_preferences(composition)

        # Composers either increase or descrease the preferences for the note
        # transitions that are observed in the compositions that they composed in
        # the iteration based on the rating success/failure rating they get back
        # from listeners.
        composer.update_preferences(composition, rating.success)

      # Finally, randomly forget/decay n% of the compositions from the
      # composition pool.
      composition_pool.decay(params.decay_rate)
  else:
    for iteration in range(params.num_iterations):
      # Pick a random composer and listener from the population to engage in a
      # pairwise interaction.
      composer = random.choice(list(composers.values()))
      listener = random.choice(list(listeners.values()))

      # Composer creates a new melody and listener rates it.
      composition = composer.create_composition(params.melody_length)
      composition_pool.add(composition)
      rating = listener.rate(composition, iteration)

      # Listener always positively updates its transition table to represent
      # the increase in familiarity to note transitions. Composer updates the
      # transition table based on the success of the interaction.
      listener.update_preferences(composition)
      composer.update_preferences(composition, rating.success)

  return composition_pool, composers, listeners


def run_experiment(
    num_runs: int, do_broadcast: bool, params: Parameters
) -> list[float]:
  aggregate = []

  for _ in range(num_runs):
    composition_pool, _, _ = simulate_society(params, do_broadcast)
    success_rates = []
    total_success = 0

    if do_broadcast:
      for iteration in range(params.num_iterations):
        iteration_ratings = composition_pool.ratings_of(iteration)
        num_success = sum(1 for r in iteration_ratings if r.success)
        num_ratings = len(iteration_ratings)
        success_rates.append(num_success / num_ratings)
    else:
      for iteration in range(params.num_iterations):
        iteration_ratings = composition_pool.ratings_of(iteration)
        total_success += 1 if iteration_ratings[0].success else 0
        success_rates.append(total_success / float(iteration + 1))

    aggregate.append(success_rates)

  return [sum(i) / float(len(i)) for i in zip(*aggregate)]
