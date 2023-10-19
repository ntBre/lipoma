.PHONY: industry industry-eq

query_flags =

ifdef PLOT
    query_flags += -p
endif

parse_query = \
    python parse_query.py -i data/industry/$1_dedup.dat \
			  -o output/$1/industry \
		          $(query_flags)

parse:
	$(call parse_query,bonds)
	$(call parse_query,angles)
	$(call parse_query,torsions)

# industry force constants
json := $(addprefix data/industry/,bonds_dedup.json angles_dedup.json	\
				   torsions_dedup.json impropers_dedup.json)
$(json): query.py
	python query.py --dataset ../benchmarking/datasets/industry.json \
			--out-dir data/industry
industry: $(json)

# industry equilibrium values
eq := $(addprefix data/industry/,bonds_eq.json angles_eq.json)
$(eq): query.py
	python query.py --dataset ../benchmarking/datasets/industry.json \
			--out-dir data/industry --force-constants

industry-eq: $(eq)
