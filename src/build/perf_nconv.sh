#/bin/sh
# take perfomance metrics for nconv
# we reset times in files before each performance measurement

# cpu vary width
>cvw.txt
for i in {3..49..2}
do
  /usr/bin/time -f "%e" ./Nconv ../Black-Star-hen.jpg 0 1 $i 2>> cvw.txt
done

#gpu vary width
>gvw.txt
for i in {3..49..2}
do
  /usr/bin/time -f "%e" ./Nconv ../Black-Star-hen.jpg 1 1 $i 2>> gvw.txt
done

#gpu vary number of filters
>gvf.txt
for i in {1..49..4}
do
  /usr/bin/time -f "%e" ./Nconv ../Black-Star-hen.jpg 1 $i 49 2>> gvf.txt
done
