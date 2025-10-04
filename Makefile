# --- User-friendly Makefile Orchestrator ---
# This Makefile wraps CMake commands for a simpler user experience.

# Configuration
BUILD_DIR := build
BENCH_EXEC := $(BUILD_DIR)/bench
PYTHON := python3

# Test data sizes (powers of 2)
TEST_SIZES := 1024 4096 16384 65536 262144

# Default target
.DEFAULT_GOAL := help

# Phony targets do not represent files
.PHONY: all build configure benchmark test clean help

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

# Generate all required data files
generate-data:
	@echo "--- Generating test data for sizes: $(TEST_SIZES) ---"
	@for size in $(TEST_SIZES); do \
$(PYTHON) scripts/generate_data.py --size $$size --output data/vector_$$size.bin; \
done

benchmark: build generate-data
	@echo "--- Running benchmarks ---"
	@$(BENCH_EXEC)

clean:
	@echo "--- Cleaning project ---"
	@rm -rf $(BUILD_DIR) data/*
