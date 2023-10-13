.PHONY: industry

parse_query = \
    python parse_query.py -i data/industry/$1_dedup.dat -o output/$1/industry

industry:
	$(call parse_query,bonds)
	$(call parse_query,angles)
	$(call parse_query,torsions)
