# have to select parameters and generate FB inputs apparently. supporting files
# are taken from valence-fitting/Sage 2.1.0 (should be the same):
# explicit_ring_torsions, smiles-to-exclude, smarts-to-exclude, and the data
# sets

init := full.offxml
opt := opt.json
td := td.json

# select optimization parameters to optimize
opt-smirks.json: $(opt) $(init)
	select_parameters.py opt	\
	--dataset $<			\
	--forcefield $(init)		\
	--output-smirks $@

# select torsiondrive parameters to optimize
td-smirks.json: $(td) $(init)
	select_parameters.py td \
	--dataset $<		\
	--forcefield $(init)	\
	--output-smirks $@	\
	--ring-torsions explicit_ring_torsions.dat

# generate the ForceBalance input. this creates a bunch of stuff, so use a dummy
# fb-ready file to indicate that it's finished
fb-deps := smiles-to-exclude.dat smarts-to-exclude.dat $(opt) $(td)	\
	   opt-smirks.json td-smirks.json $(init)

fb-ready fb-fit/optimize.in fb-fit/forcefield/force-field.offxml: $(fb-deps)
	-rm -r fb-fit/targets
	mkdir -p fb-fit/targets
	create_fb_inputs.py                                     \
	--tag                       "fb-fit"                    \
	--optimization-dataset      $(opt)			\
	--torsion-dataset           $(td)			\
	--valence-to-optimize       opt-smirks.json		\
	--torsions-to-optimize      td-smirks.json		\
	--forcefield                $(init)			\
	--smiles-to-exclude         smiles-to-exclude.dat	\
	--smarts-to-exclude         smarts-to-exclude.dat	\
	--max-iterations            100                         \
	--port                      55387                       \
	--output-directory          "output"                    \
	--verbose
	date > $@

# pack up the generated files for copying to a supercomputer
fb-fit/targets.tar.gz: fb-ready
	-rm $@
	cd fb-fit; tar cfz targets.tar.gz targets

deps := $(addprefix fb-fit/,forcefield/force-field.offxml optimize.in	\
targets.tar.gz)
out.tar.gz: fb-ready $(deps)
	-rm $@
	tar cfz $@ $(deps)

