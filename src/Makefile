proj3: main.o bitmap.o baysiannetwork.o
	g++ -o proj3 main.o bitmap.o baysiannetwork.o
main.o: main.cpp
	g++ -std=c++11 -c main.cpp
bitmap.o: bitmap.cpp
	g++ -std=c++11 -c bitmap.cpp
baysiannetwork.o: baysiannetwork.cpp
	g++ -std=c++11 -c baysiannetwork.cpp
clean:
	rm proj3 main.o bitmap.o baysiannetwork.o
