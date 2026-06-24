set terminal png size 800,600
set output 'ghz_M1_Max_scaling.png'
set datafile separator ','
set key autotitle columnhead
set title 'ghz Performance (mlx-Quantum on M1 Max)'
set xlabel 'Qubits'
set ylabel 'Execution Time (ms)'
set logscale y
set grid
plot 'ghz_M1_Max_data.csv' using 1:2 with linespoints linewidth 2 pointtype 7
