PY_INC_PATH := $(shell python -c "import sysconfig; print(sysconfig.get_paths()['include'])")

CUDA_HOME := $(shell which nvcc | xargs dirname | xargs dirname)

test: build_extension
	# PYTHONPATH=.. time python test/test_sum_loop.py
	# PYTHONPATH=.. time python test/test_layout.py
	PYTHONPATH=.. time python test/test_cat_sum.py

tutor: build_extension
	PYTHONPATH=.. python tutor.py

build_extension:
	time g++ extension.cpp -I$(PY_INC_PATH) -I$(CUDA_HOME)/include -O3 -shared -fPIC -lcuda -o _C.so
