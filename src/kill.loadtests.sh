for pr in `ps ax | grep loadtest | sed s/" s00.*"//g`
do
    echo $pr
    kill $pr
done