query_flags =

ifdef PLOT
    query_flags += -p
endif

parse_query = \
    python parse_query.py -i data/$1/$2.dat \
			  -o output/$2/$1 \
		          $(query_flags)

parse_query_i = $(call parse_query,industry,$1)
parse_query_m = $(call parse_query,msm,$1)

all: msm industry industry-eq

parse:
	$(call parse_query_i,bonds_dedup)
	$(call parse_query_i,angles_dedup)
	$(call parse_query_i,torsions_dedup)
	$(call parse_query_i,impropers_dedup)

	$(call parse_query_i,bonds_eq)
	$(call parse_query_i,angles_eq)

	$(call parse_query_m,bonds_dedup)
	$(call parse_query_m,angles_dedup)
	$(call parse_query_m,bonds_eq)
	$(call parse_query_m,angles_eq)

# industry force constants
json := $(addprefix data/industry/,bonds_dedup.json angles_dedup.json	\
				   torsions_dedup.json impropers_dedup.json)
$(json): query.py
	python query.py --dataset datasets/industry.json \
			--out-dir data/industry
industry: $(json)

# industry equilibrium values
eq := $(addprefix data/industry/,bonds_eq.json angles_eq.json)
$(eq): query.py
	python query.py --dataset datasets/industry.json \
			--out-dir data/industry --force-constants

industry-eq: $(eq)

msm := $(addprefix data/msm/,angles_dedup.json angles_eq.json bonds_dedup.json	\
			     bonds_eq.json)
$(msm): msm.py
	python msm.py -o data/msm

msm: $(msm)

forcefields/full.offxml: apply.py
	python apply.py							\
		--output $@						\
		--angles-eq output/angles_eq/industry/output.dat	\
		--bonds-eq output/bonds_eq/industry/output.dat		\
		--torsions output/torsions_dedup/industry/output.dat	\
		--angles output/angles_dedup/industry/output.dat	\
		--bonds output/bonds_dedup/industry/output.dat
