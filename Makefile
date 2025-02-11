# Compiler commands
NVCC    := nvcc
CC      := gcc

# Target executable name
TARGET  := sim

# Use sdl2-config to get SDL2 flags.
CFLAGS  := -Wall -O2 $(shell sdl2-config --cflags)
LDFLAGS := $(shell sdl2-config --libs) -L/usr/local/cuda-12.8/lib64 -lcudart -lm

# Source and object files in the src folder.
SOURCES := src/main.c src/engine.cu
OBJS    := main.o engine.o

# Default target
all: $(TARGET)

# Compile main.c using gcc.
main.o: src/main.c src/engine.h
	$(CC) $(CFLAGS) -c src/main.c -o main.o

# Compile engine.cu using nvcc.
engine.o: src/engine.cu src/engine.h
	$(NVCC) -c src/engine.cu -o engine.o

# Link the object files using gcc.
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Clean up generated files.
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean