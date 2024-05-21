# @title Run simulation n times and plot success rate curve

do_broadcast = True
num_runs = 30
params = params_with_broadcast if do_broadcast else params_wo_broadcast
success_rates = run_experiment(num_runs, do_broadcast, params)

_, ax = plt.subplots()
ax.plot(tuple(range(params.num_iterations)), success_rates)
ax.set(xlabel='Iteration', ylabel='Success rate')
ax.grid()
plt.show()
