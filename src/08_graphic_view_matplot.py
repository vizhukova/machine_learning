import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10)
y = 2 * x

plt.plot(x, y)
plt.title("Some Title")
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.xlim(0, 6)
plt.ylim(0, 15)

# plt.show()
# plt.savefig('test.jpg')

# Creates empty frame
fig = plt.figure(dpi=200)

# the position and the size of the canvas:
# Lower left corner, right top corner
axes1 = fig.add_axes([0, 0, 1, 1]) # Large figure
axes2 = fig.add_axes([0.2, 0.2, 0.5, 0.5]) # Smaller figure

a = np.linspace(0,10,11)
b = a ** 4

axes1.plot(a, b)

axes1.set_xlabel('X Label')
axes1.set_ylabel('Y Label')
axes1.set_title('Big Figure')

# Insert Figure Axes 2
axes2.plot(a,b)
axes2.set_xlim(8,10)
axes2.set_ylim(4000,10000)
axes2.set_xlabel('X')
axes2.set_ylabel('Y')
axes2.set_title('Zoomed In')

# plt.show()

print(type(fig))

# Empty canvas of 2 by 2 subplots
fig, axes = plt.subplots(nrows=2, ncols=2)
fig,axes = plt.subplots(nrows=1,ncols=2)

for axe in axes:
    axe.plot(x,y)

plt.show()