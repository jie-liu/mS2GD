CC = g++
CFLAGS = -O3
MAIN = build/MB_SVRG



all: $(MAIN)

$(MAIN):  src/command_line.h src/SGDs.h src/read_binary.h src/logistic_L2.h src/MB_SVRG.cpp
	$(CC) $(CFLAGS)  src/MB_SVRG.cpp -o $@

clean:
	rm -rf ./build/*

