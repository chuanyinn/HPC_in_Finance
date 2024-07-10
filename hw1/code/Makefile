# Compiler
CC = icc

# Compiler flags
CFLAGS = -std=c++11 -g -Wall -qopt-report=0

# Source file
ASSIGNMENT_1 = assignment_1.cpp

# Target executable
all: assignment_1

# Rule to build the executable
assignment_1: $(ASSIGNMENT_1)
	$(CC) $(CFLAGS) $< -o $@

# Clean rule to remove generated files
clean:
	rm -rf assignment_1 *.optrpt