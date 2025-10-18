import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
import math


flight_path_10k = pd.read_csv('MRT Porthos official Flight data 2023.csv')
flight_path_60k = pd.read_csv('FlightData60k.csv')

def plot_altitude_vs_time(files, labels=None, output='altitude_vs_time.png'):
	"""Plot altitude vs time for a list of CSV file paths."""
	plt.figure(figsize=(10, 6))
	for i, f in enumerate(files):
		df = pd.read_csv(f)
		if 'Time' not in df.columns or 'Altitude' not in df.columns:
			raise ValueError(f"File {f} does not contain 'Time' and 'Altitude' columns")
		t = df['Time'].astype(float)
		alt = df['Altitude'].astype(float)
		label = labels[i] if labels and i < len(labels) else f
		plt.plot(t, alt, label=label)

	plt.xlabel('Time (s)')
	plt.ylabel('Altitude (m)')
	plt.title('Altitude vs Time')
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output)
	print(f"Saved plot to {output}")
	plt.show()


if __name__ == '__main__':
	files = ['MRT Porthos official Flight data 2023.csv',
			 'FlightData60k.csv']
	labels = ['Porthos flight (10k)', 'FlightData 60k']
	plot_altitude_vs_time(files, labels)
