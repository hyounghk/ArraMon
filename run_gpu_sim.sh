for i in `seq 1 $1`
do
	sudo xinit ./build.x86_64 --batch-size $2 --port $3 -logFile ./sim_log.log --scaled-size 224 --quality 5 -sharevts -- :"$i" &
	sleep 5
done
