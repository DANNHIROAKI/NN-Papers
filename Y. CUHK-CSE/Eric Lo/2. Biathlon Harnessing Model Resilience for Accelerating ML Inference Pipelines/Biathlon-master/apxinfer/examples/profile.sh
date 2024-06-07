pip install gprof2dot

python -m cProfile -s cumtime -o ~/.cache/apxinf/xip/online.pstats run.py --stage online --task test/trips --model lgbm --nparts 10 --nreqs 20 --offline_nreqs 20 --pest_constraint error --max_error 0.5 --min_conf 0.95
gprof2dot -f pstats ~/.cache/apxinf/xip/online.pstats | dot -Tsvg -o ~/.cache/apxinf/xip/online.svg

python -m cProfile -s cumtime -o ~/.cache/apxinf/xip/exact.pstats run.py --stage online --task test/trips --model lgbm --nparts 10 --nreqs 20 --offline_nreqs 20 --exact
gprof2dot -f pstats ~/.cache/apxinf/xip/exact.pstats | dot -Tsvg -o ~/.cache/apxinf/xip/exact.svg
