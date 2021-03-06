import numpy as np
import sys

from kemp.fdtd3d.cpu import QueueTask, Fields, Core, Pbc, IncidentDirect


tmax = 150
tfunc = lambda tstep: np.sin(0.05 * tstep)

# plot
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.ioff()
mpl.rc('image', interpolation='nearest', origin='lower')
fig = plt.figure(figsize=(14,8))


# z-axis
nx, ny, nz = 180, 160, 4
fields = Fields(QueueTask(), nx, ny, nz)
Pbc(fields, 'xyz')
Core(fields)
IncidentDirect(fields, 'ey', (20, 0, 0), (20, -1, -1), tfunc) 
IncidentDirect(fields, 'ex', (0, 20, 0), (-1, 20, -1), tfunc) 

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()
fields.enqueue_barrier()

ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(fields.get('ey')[:,:,nz/2].T, vmin=-1.1, vmax=1.1)
ax1.set_title('%s, ey[20,:,:]' % repr(fields.ns))
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2 = fig.add_subplot(2, 3, 4)
ax2.imshow(fields.get('ex')[:,:,nz/2].T, vmin=-1.1, vmax=1.1)
ax2.set_title('%s, ex[:,20,:]' % repr(fields.ns))
ax2.set_xlabel('x')
ax2.set_ylabel('y')

# y-axis 
nx, ny, nz = 180, 2, 160
fields = Fields(QueueTask(), nx, ny, nz)
Pbc(fields, 'xyz')
Core(fields)
IncidentDirect(fields, 'ez', (20, 0, 0), (20, -1, -1), tfunc) 
IncidentDirect(fields, 'ex', (0, 0, 20), (-1, -1, 20), tfunc) 

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()
fields.enqueue_barrier()

ax1 = fig.add_subplot(2, 3, 2)
ax1.imshow(fields.get('ez')[:,ny/2,:].T, vmin=-1.1, vmax=1.1)
ax1.set_title('%s, ez[20,:,:]' % repr(fields.ns))
ax1.set_xlabel('x')
ax1.set_ylabel('z')

ax2 = fig.add_subplot(2, 3, 5)
ax2.imshow(fields.get('ex')[:,ny/2,:].T, vmin=-1.1, vmax=1.1)
ax2.set_title('%s, ex[:,:,20]' % repr(fields.ns))
ax2.set_xlabel('x')
ax2.set_ylabel('z')

# x-axis 
nx, ny, nz = 2, 180, 160
fields = Fields(QueueTask(), nx, ny, nz)
Pbc(fields, 'xyz')
Core(fields)
IncidentDirect(fields, 'ez', (0, 20, 0), (-1, 20, -1), tfunc) 
IncidentDirect(fields, 'ey', (0, 0, 20), (-1, -1, 20), tfunc) 

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()
fields.enqueue_barrier()

ax1 = fig.add_subplot(2, 3, 3)
ax1.imshow(fields.get('ez')[nx/2,:,:].T, vmin=-1.1, vmax=1.1)
ax1.set_title('%s, ez[:,20,:]' % repr(fields.ns))
ax1.set_xlabel('y')
ax1.set_ylabel('z')

ax2 = fig.add_subplot(2, 3, 6)
ax2.imshow(fields.get('ey')[nx/2,:,:].T, vmin=-1.1, vmax=1.1)
ax2.set_title('%s, ey[:,:,20]' % repr(fields.ns))
ax2.set_xlabel('y')
ax2.set_ylabel('z')

print('save as png')
plt.savefig('plot.png')
#plt.show()
