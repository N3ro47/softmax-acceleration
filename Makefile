BUILD_DIR := build
BENCH_EXEC := $(BUILD_DIR)/bench
PYTHON := python3

TEST_SIZES := 1024 4096 16384 65536 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456

.DEFAULT_GOAL := help

.PHONY: all build configure benchmark clean help bench

# Targets
help:
	@echo "Available commands:"
	@echo "  make configure   - Configures the project using CMake and Ninja."
	@echo "  make build       - Builds all targets (requires configuration first)."
	@echo "  make benchmark   - Generates data and runs performance benchmarks."
	@echo "  make clean       - Removes generated build files and test data."

configure:
	@echo "--- Configuring project with CMake ---"
	@cmake -S . -B $(BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=Release

build: configure
	@echo "--- Building project with Ninja ---"
	@cmake --build $(BUILD_DIR)

generate-data:
	@echo "--- Generating test data for sizes: $(TEST_SIZES) ---"
	@for size in $(TEST_SIZES); do \
$(PYTHON) scripts/generate_data.py --size $$size --output data/vector_$$size.bin; \
done

benchmark bench: build generate-data
	@echo "--- Running benchmarks ---"
	@$(BENCH_EXEC) # --benchmark_repetitions=5

test: build
	@echo "--- Running unit tests ---"
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	@echo "--- Cleaning project ---"
	@rm -rf $(BUILD_DIR) data/*
