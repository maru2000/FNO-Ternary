#!/bin/sh

var=1111     #this is for seed
i=1000    #this is for timestep
ind=0
while [ $ind -le 1 ]
do
	GSL_RNG_SEED=${var} ./tern.exe

	./cpbin
	for file in prof_gp*; do
		mv "$file" "$ind$file"; done;
		mv *prof_gp* bintemp
	var=$((var+1))
	ind=$((ind+1))
	echo "BATCH $ind DONE !!"
done
rm -rf comp*
