# @title Run simulation 1 times and plot global transition table

def compute_global_transition_table(
    composers: Iterable[Composer], listeners: Iterable[Listener]
) -> TransitionTable:
  agent_count = len(composers) + len(listeners)

  composer_tables = [c.transition_table for c in composers]
  listener_tables = [l.transition_table for l in listeners]
  table_pool = composer_tables + listener_tables

  semitones = table_pool[0].semitones()
  global_table = TransitionTable(len(semitones))
  transitions = tuple(itertools.product(semitones, semitones))

  for table in table_pool:
    for prev_note, next_note in transitions:
      addition = table.preference_score(prev_note, next_note)
      global_table.add_to_preference_score(prev_note, next_note, addition)

  global_table.normalize_preference_scores(float(agent_count))
  return global_table


do_broadcast = True
params = params_with_broadcast if do_broadcast else params_wo_broadcast
composition_pool, composers, listeners = simulate_society(params, do_broadcast)
global_table = compute_global_transition_table(
    composers.values(),
    listeners.values(),
)
global_table.to_csv()
