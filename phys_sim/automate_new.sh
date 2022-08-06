#!/bin/sh

var=0   # Seed
i=0     # Timestep
while [ $var -le 5 ]
do
	GSL_RNG_SEED=${var} ./tern.exe
	i=0
	while [ $i -le 5 ]
	do 
		./bin2asc 
		i=$((i + 2))
	done
	var=$((var + 1))
done
