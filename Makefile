.PHONY: setup smoke train prep clean

setup:
	bash scripts/setup_env.sh

smoke:
	bash scripts/run_smoke_test.sh

train:
	bash scripts/run_full_train.sh

prep:
	python data/data_prep.py

clean:
	rm -rf __pycache__ .pytest_cache
	rm -f metadata_clean.csv modalities.hdf5 extinction_model.pth
