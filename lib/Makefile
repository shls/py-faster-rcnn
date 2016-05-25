python3:
	python3 setup.py build_ext --inplace
	rm -rf build

python2:
	python setup.py build_ext --inplace
	rm -rf build

clean:
	rm utils/bbox.c
	rm nms/cpu_nms.c
	rm nms/gpu_nms.cpp