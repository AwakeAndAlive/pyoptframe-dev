
# register model components (evaluation function, constructive, ...)
pKP.engine.setup(pKP)
ev_idx = 0
c_idx = 0
is_idx = 0
# is_idx = pKP.engine.create_initial_search(ev_idx, c_idx)

# make engine silent (loglevel 0)
pKP.engine.experimental_set_parameter("ENGINE_LOG_LEVEL", "0")

# test each component

fev = pKP.engine.get_evaluator(ev_idx)
pKP.engine.print_component(fev)

fc = pKP.engine.get_constructive(c_idx)
pKP.engine.print_component(fc)

solxx = pKP.engine.fconstructive_gensolution(fc)
print("solxx:", solxx)

z1 = pKP.engine.fevaluator_evaluate(fev, False, solxx)
print("evaluation:", z1)
