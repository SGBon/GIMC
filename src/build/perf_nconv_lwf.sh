#/bin/sh
# take perfomance metrics for nconv
# we reset times in files before each performance measurement

#gpu vary width
>gvw_lwf.txt
for i in {3..49..2}
do
  /usr/bin/time -f "%e" ./Nconv_lwf ../Black-Star-hen.jpg 1 1 $i 2>> gvw_lwf.txt
done

#gpu vary number of filters
>gvf_lwf.txt
for i in {1..49..4}
do
  /usr/bin/time -f "%e" ./Nconv_lwf ../Black-Star-hen.jpg 1 $i 49 2>> gvf_lwf.txt
done
