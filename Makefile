.PHONY: industry

query_flags =

ifdef PLOT
    query_flags += -p
endif

parse_query = \
    python parse_query.py -i data/industry/$1_dedup.dat \
			  -o output/$1/industry \
		          $(query_flags)

industry:
	$(call parse_query,bonds)
	$(call parse_query,angles)
	$(call parse_query,torsions)

json := $(addprefix data/industry/,bonds_dedup.json angles_dedup.json	\
				   torsions_dedup.json)
$(json): query.py
	python query.py --dataset ../benchmarking/datasets/industry.json \
			--out-dir data/industry
