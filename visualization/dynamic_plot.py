import matplotlib.pyplot as plt
import time
import random
import numpy as np
 
class dynamic_plot:
	def __init__(self,
				 xlim = 100,
				 ylim = 50,
				 xdata = [],
				 ydata = [],
				 refresh_window = 10,
				 refresh_complete = False):
		
		self.xdata = xdata
		self.ydata = ydata
		self.refresh_window = refresh_window
		self.xlim, self.ylim = xlim, ylim
		self.refresh_complete = refresh_complete
		plt.show()
	 	
		self.axes = plt.gca()
		self.axes.set_xlim(0, xlim)
		self.axes.set_ylim(-ylim, +ylim)
		self.line, = self.axes.plot(self.xdata, self.ydata, 'r-')
 	

	def add_point(self, x=[0],y=[0]):
		if type(x) != list:
			x = [x]
			y = [y]
		for idx, i in enumerate(x):
			self.xdata.append(i)
			self.ydata.append(y[idx]*0.3)
			self.line.set_xdata(self.xdata)
			self.line.set_ydata(self.ydata)
			plt.draw()
			plt.pause(1e-17)
			time.sleep(0.1)
		if len(self.xdata) == self.axes.get_xlim()[1]: #limit for graph reached
			start = 0 if self.refresh_complete == False else  self.axes.get_xlim()[0] + self.refresh_window
			self.axes.set_xlim(start, self.axes.get_xlim()[1]+self.refresh_window)
		if len

if __name__ == "__main__":
	o = dynamic_plot(xlim=100, refresh_window=10, refresh_complete=True)
	# o.add_point(x=list(range(0,100)),y=list(range(0,100)))
	for i in range(1000):
		y = np.random.random(1)
		o.add_point(i,y[0]*100)
	plt.show()
	# plt.show()