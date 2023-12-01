query_flags =

ifdef PLOT
    query_flags += -p
endif

all: msm industry industry-eq

output_dats := $(addsuffix /output.dat,\
       output/bonds_dedup/industry output/angles_dedup/industry		\
       output/torsions_dedup/industry output/impropers_dedup/industry	\
       output/bonds_eq/industry output/angles_eq/industry		\
       output/bonds_dedup/msm output/angles_dedup/msm			\
       output/bonds_eq/msm output/angles_eq/msm)

parse: $(output_dats)

output/%/industry/output.dat: data/industry/%.dat
	python parse_query.py -i $< -o $(dir $@) $(query_flags)

output/%/msm/output.dat: data/msm/%.dat
	python parse_query.py -i $< -o $(dir $@) $(query_flags)


# industry force constants
json := $(addprefix data/industry/,bonds_dedup.json angles_dedup.json	\
				   torsions_dedup.json impropers_dedup.json)
dats := $(subst .json,.dat,$(json))

$(json) $(dats): query.py
	python query.py --dataset datasets/industry.json \
			--out-dir data/industry
industry: $(json)

# industry equilibrium values
eq := $(addprefix data/industry/,bonds_eq.json angles_eq.json)
$(eq) $(subst .json,.dat,$(eq)): query.py
	python query.py --dataset datasets/industry.json \
			--out-dir data/industry --force-constants

industry-eq: $(eq)

# espaloma force constants on opt data
esp_fc := $(addprefix data/esp/,bonds_dedup.json angles_dedup.json	\
				   torsions_dedup.json impropers_dedup.json)
$(esp_fc) $(subst .json,.dat,$(esp_fc)): query.py
	python query.py --dataset datasets/filtered-opt.json \
			--out-dir data/esp
esp: $(esp_fc)

# espaloma equilibrium values on opt data
esp_eq := $(addprefix data/esp/,bonds_eq.json angles_eq.json)
$(esp_eq): query.py
	python query.py --dataset datasets/filtered-opt.json \
			--out-dir data/esp --force-constants
esp-eq: $(esp_eq)

# all msm values for opt data

msm := $(addprefix data/msm/,angles_dedup.json angles_eq.json bonds_dedup.json	\
			     bonds_eq.json)
$(msm): msm.py
	python msm.py -o data/msm

msm: $(msm)

forcefields/full.offxml: apply.py $(output_dats)
	python apply.py							\
		--output $@						\
		--angles-eq output/angles_eq/industry/output.dat	\
		--bonds-eq output/bonds_eq/industry/output.dat		\
		--torsions output/torsions_dedup/industry/output.dat	\
		--angles output/angles_dedup/industry/output.dat	\
		--bonds output/bonds_dedup/industry/output.dat

data := $(json) $(eq) $(msm)
src := proxy/src proxy/Cargo.toml proxy/Cargo.lock proxy/index.html
py := board.py utils.py query.py cluster.py main.py wrapper.py twod.py
deploy:
	rsync -Ravz $(src) env.yaml $(py) $(data) 'vomsf:server/.'
