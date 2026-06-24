set terminal png size 800,600
set output 'long_range_ising_M1_Max_scaling.png'
set datafile separator ','
set key autotitle columnhead
set title 'long_range_ising Performance (mlx-Quantum on M1 Max)'
set xlabel 'Qubits'
set ylabel 'Execution Time (ms)'
set logscale y
set grid
plot 'long_range_ising_M1_Max_data.csv' using 1:2 with linespoints linewidth 2 pointtype 7
