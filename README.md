# lipoma
Literally Interpreting esPalOMA parameters

# Usage

``` shell
mamba env create -f env.yaml
```

If you want to run the dashboard (`python board.py`), you also need to generate
the data files. This can be done simply with `make`:

``` shell
make
```

This runs the default `all` recipe, which in turn runs the following commands to
generate the JSON files read by the dashboard.

``` shell
python query.py -d datasets/industry.json -o data/industry
python query.py -d datasets/industry.json -o data/industry -f
python msm.py -d datasets/filtered-opt.json -o data/msm
```
