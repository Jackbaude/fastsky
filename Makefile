
help:
	@echo "FastSky - GPU-Accelerated Satellite Tracker"
	@echo ""
	@echo "Available targets:"
	@echo "  make install         - Install dependencies with Poetry"
	@echo "  make lock           - Update Poetry lock file"
	@echo "  make run            - Run the satellite tracker"
	@echo "  make download-tle   - Download fresh TLE data from Celestrak"
	@echo "  make clean           - Clean generated files"
	@echo "  make container-build - Build Podman container image"
	@echo "  make container-run   - Run in Podman container (tries GPU)"
	@echo "  make container-run-cpu - Run in Podman container (CPU only)"

install:
	@echo "Installing dependencies..."
	poetry install
	@echo "✓ Installation complete!"

lock:
	@echo "Updating lock file..."
	poetry lock
	@echo "✓ Lock file updated!"

run:
	@echo "Starting satellite tracker..."
	@if ! ls *.tle 1> /dev/null 2>&1; then \
		echo "No TLE file found. Downloading..."; \
		$(MAKE) download-tle; \
	fi
	poetry run fastsky

download-tle:
	@echo "Downloading latest Starlink TLE data..."
	wget -O starlink-tle-latest.tle "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
	@echo "✓ Downloaded to starlink-tle-latest.tle"
	@echo "Run with: fastsky --tle starlink-tle-latest.tle"

clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned!"

