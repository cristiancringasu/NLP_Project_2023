kill -9 `cat save_pid.txt`
rm save_pid.txt
kill $(ps aux | grep 'StanfordCoreNLPServe' | awk '{print $2}')
printf -v datelog '%(%Y-%m-%d--%H-%M-%S)T' -1 
cp server.log logs/"server--$datelog".log