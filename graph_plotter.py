#plots the graph of the best global chromosome over time
import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import time
from itertools import count

plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []
count1 = 0
count2 = 0

city_x = []
city_y = []
travel_order = []
city_order_x = []
city_order_y = []

#Change this parameter according to the number of
#cities in the data set
n = 127

print("HI")

fitness = []
total_distance = []

with open('best_fitness_global.txt','r') as csvfile:
	plots = csv.reader(csvfile, delimiter=' ')
	for row in plots:
		fitness.append(float(row[0]))

with open('best_total_distance_global.txt','r') as csvfile:
	plots = csv.reader(csvfile, delimiter=' ')
	for row in plots:
		total_distance.append(float(row[0]))

with open('city_coord_data.txt','r') as csvfile:
	plots = csv.reader(csvfile, delimiter=' ')
	for row in plots:
		city_x.append(int(row[0]))
		city_y.append(int(row[1]))
		
def animate1(i):
	global count1
	with open('best_chromosome_global.txt','r') as csvfile:
		plots = csv.reader(csvfile, delimiter=' ')
		for i in range(count1):
			next(plots)
		row = next(plots)
		
		count=0
		start=0
		for i in range(n):
			if count==0:
				start=int(row[i])
			travel_order.append(int(row[i]))
			count=count+1
		travel_order.append(start)
		
		for i in range(n+1):
			city_order_x.append(city_x[travel_order[i]])
			city_order_y.append(city_y[travel_order[i]])
			
		plt.cla()
		plt.plot(city_order_x,city_order_y, label='Chromosomes_generation', linewidth=1.0, alpha=0.4)
		travel_order.clear()
		city_order_x.clear()
		city_order_y.clear()
		count1=count1+1
		print(count1, ": ", total_distance[count1])
	plt.scatter(city_x,city_y,label='Cities',color='red',alpha=1,s=100,marker='o')	

ani1 = FuncAnimation(plt.gcf(), animate1, interval=1)
#ani2 = FuncAnimation(plt.gcf(), animate2, interval=0.1)
plt.tight_layout()
plt.show()

