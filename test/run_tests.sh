#!/bin/bash

# Script to run all test_*.py files except those marked for manual execution
# Usage: ./run_tests.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}=== Automated Test Runner ===${NC}"
echo "Scanning directory: $SCRIPT_DIR"
echo

# Initialize counters
total_tests=0
skipped_tests=0
failed_tests=0
passed_tests=0

# Arrays to track test results
declare -a failed_test_files=()
declare -a skipped_test_files=()

# Find all test_*.py files in the script directory
for test_file in "$SCRIPT_DIR"/test_*.py; do
    # Check if any files match the pattern
    if [ ! -f "$test_file" ]; then
        echo -e "${YELLOW}No test files found matching pattern 'test_*.py'${NC}"
        exit 0
    fi

    total_tests=$((total_tests + 1))
    filename=$(basename "$test_file")

    # Check if file contains "# RUN MANUALLY ONLY" comment
    if grep -q "RUN MANUALLY ONLY" "$test_file"; then
        echo -e "${YELLOW}SKIPPING${NC} $filename (marked for manual execution)"
        skipped_tests=$((skipped_tests + 1))
        skipped_test_files+=("$filename")
        continue
    fi

    # Run the test
    echo -e "${BLUE}RUNNING${NC} $filename..."

    # Capture all output and exit code
    test_output=$(python3 "$test_file" 2>&1)
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}PASSED${NC} $filename"
        passed_tests=$((passed_tests + 1))
    else
        echo -e "${RED}FAILED${NC} $filename"
        failed_tests=$((failed_tests + 1))
        failed_test_files+=("$filename")

        # Show all output (stdout + stderr) for failed tests
        echo -e "${RED}Test output:${NC}"
        echo "$test_output" | sed 's/^/  /'
        echo
    fi
done

# Print summary
echo
echo -e "${BLUE}=== Test Summary ===${NC}"
echo "Total test files found: $total_tests"
echo -e "${GREEN}Passed: $passed_tests${NC}"
echo -e "${RED}Failed: $failed_tests${NC}"
echo -e "${YELLOW}Skipped: $skipped_tests${NC}"

# List failed tests if any
if [ ${#failed_test_files[@]} -gt 0 ]; then
    echo
    echo -e "${RED}Failed tests:${NC}"
    for failed_test in "${failed_test_files[@]}"; do
        echo "  - $failed_test"
    done
fi

# List skipped tests if any
if [ ${#skipped_test_files[@]} -gt 0 ]; then
    echo
    echo -e "${YELLOW}Skipped tests:${NC}"
    for skipped_test in "${skipped_test_files[@]}"; do
        echo "  - $skipped_test"
    done
fi

echo

# Exit with appropriate code
if [ $failed_tests -gt 0 ]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi