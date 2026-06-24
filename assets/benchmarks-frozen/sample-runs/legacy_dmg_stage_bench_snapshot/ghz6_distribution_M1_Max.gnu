set terminal pngcairo size 1200,600 enhanced font 'Arial,12'
set output 'bench/ghz6_distribution_M1_Max.png'
set title 'GHZ-6 Measurement Distribution (10000 shots on M1 Max)'
set xlabel 'Measurement Outcome'
set ylabel 'Probability'
set yrange [0:0.6]
set grid
set style fill solid 0.7
set boxwidth 0.8
set xtics rotate by 45 right
set datafile separator ','
plot 'bench/ghz6_distribution_M1_Max.csv' using 0:3:xtic(1) with boxes lc rgb '#00AA00' title 'Measured', \
     0.5 with lines lc rgb 'red' lw 2 dt 2 title 'Theoretical (0.5)'
