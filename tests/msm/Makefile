# this is an experiment to see how the MSM values perform alone without any
# re-fitting on them. from the histograms I've produced, the average MSM values
# look quite similar to the optimized Sage values anyway, and this would save
# time in the fitting procedure

msm.offxml:
	create_msm.py --initial-force-field openff-2.1.0.offxml \
		      --optimization-dataset ../../datasets/filtered-opt.json \
		      --output $@
