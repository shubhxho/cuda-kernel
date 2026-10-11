.PHONY: host-test build gpu-test bench model report gpu-validate sweep
host-test:
	cmake -S . -B build-host -DENABLE_CUDA=OFF
	cmake --build build-host -j2
	ctest --test-dir build-host --output-on-failure
	python3 -m unittest discover -s tests -v
build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j2
gpu-test: build
	ctest --test-dir build --output-on-failure
bench: gpu-test
	./build/decode_bench --suite decode --output results/kernels.json
	python3 tools/report.py results/kernels.json --output results/kernels.md
model:
	python3 tools/model_bench.py --output results/model.json
	python3 tools/model_report.py results/model.json --output results/model.md
report:
	python3 tools/report.py results/kernels.json --output results/kernels.md
	python3 tools/model_report.py results/model.json --output results/model.md

gpu-validate:
	python3 tools/gpu_validate.py
sweep: gpu-test
	python3 tools/sweep.py --model
