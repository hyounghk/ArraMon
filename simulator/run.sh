#!/bin/sh
for i in `seq 1 $1`
do
	docker run --rm --network host --name sim"$i" arramonsim:latest --port $3 --batch-size $2 --scaled-sized 224 --quality 5 &	
done
