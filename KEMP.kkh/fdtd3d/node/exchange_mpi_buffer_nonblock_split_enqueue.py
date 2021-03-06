import numpy as np
from mpi4py import MPI

from kemp.fdtd3d.util import common
from kemp.fdtd3d import gpu, cpu


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class ExchangeMpiBufferNonBlockSplitEnqueue():
    def __init__(self, gpuf, direction, tmax):
        common.check_type('gpuf', gpuf, gpu.Fields)
        common.check_value('direction', direction, ('+', '-', '+-'))

        qtask = cpu.QueueTask()

        if '+' in direction:
            self.cpuf_p = cpuf_p = cpu.Fields(qtask, 3, gpuf.ny, gpuf.nz, gpuf.coeff_use, gpuf.precision_float, use_cpu_core=1)

            self.gf_p_h = gpu.GetFields(gpuf, ['hy', 'hz'], (-2, 0, 0), (-2, -1, -1)) 
            self.sf_p_h = cpu.SetFields(cpuf_p, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            self.gf_p_e = cpu.GetFields(cpuf_p, ['ey', 'ez'], (1, 0, 0), (1, -1, -1)) 
            self.sf_p_e = gpu.SetFields(gpuf, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True)

            self.gf_h = gf_h = cpu.GetFields(cpuf_p, ['hy', 'hz'], (1, 0, 0), (1, -1, -1)) 
            self.sf_e = cpu.SetFields(cpuf_p, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True) 

            self.req_send_h = comm.Send_init(gf_h.host_array, rank+1, tag=0)
            self.tmp_recv_e_list = [np.zeros(gf_h.host_array.shape, gpuf.dtype) for i in range(2)]
            self.req_recv_e_list = [comm.Recv_init(tmp_recv_e, rank+1, tag=1) for tmp_recv_e in self.tmp_recv_e_list]
            self.switch_e = 0

        if '-' in direction:
            self.cpuf_m = cpuf_m = cpu.Fields(qtask, 3, gpuf.ny, gpuf.nz, gpuf.coeff_use, gpuf.precision_float, use_cpu_core=1)
            self.gf_m_e = gpu.GetFields(gpuf, ['ey', 'ez'], (1, 0, 0), (1, -1, -1)) 
            self.sf_m_e = cpu.SetFields(cpuf_m, ['ey', 'ez'], (-1, 0, 0), (-1, -1, -1), True)

            self.gf_m_h = cpu.GetFields(cpuf_m, ['hy', 'hz'], (1, 0, 0), (1, -1, -1)) 
            self.sf_m_h = gpu.SetFields(gpuf, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True)

            self.gf_e = gf_e = cpu.GetFields(cpuf_m, ['ey', 'ez'], (1, 0, 0), (1, -1, -1)) 
            self.sf_h = cpu.SetFields(cpuf_m, ['hy', 'hz'], (0, 0, 0), (0, -1, -1), True) 

            self.req_send_e = comm.Send_init(gf_e.host_array, rank-1, tag=1)
            self.tmp_recv_h_list = [np.zeros(gf_e.host_array.shape, gpuf.dtype) for i in range(2)]
            self.req_recv_h_list = [comm.Recv_init(tmp_recv_h, rank-1, tag=0) for tmp_recv_h in self.tmp_recv_h_list]
            self.switch_h = 0

        # global variables
        self.direction = direction
        self.qtask = qtask

        self.tmax = tmax
        self.tstep = 1



    def update_e(self):
        if '+' in self.direction:
            # update e
            for instance in self.cpuf_p.instance_list:
                instance.update_e()

            # internal send e
            self.sf_p_e.set_fields(self.gf_p_e.get_fields(), [self.gf_p_e.get_event()])

        if '-' in self.direction:
            # update e
            for instance in self.cpuf_m.instance_list:
                instance.update_e('pre')

            # mpi send e
            #print 'req_send_e', self.tstep, rank, self.req_send_e.Test()
            if self.tstep > 1: self.qtask.enqueue(self.req_send_e.Wait)
            self.qtask.enqueue(self.req_send_e.Start, wait_for=[self.gf_e.get_event()])
            if self.tstep == self.tmax: self.qtask.enqueue(self.req_send_e.Wait)

            # mpi recv h
            if self.tstep > 1: 
                self.qtask.enqueue(self.req_recv_h_list[self.switch_h].Wait)
                self.sf_h.set_fields(self.tmp_recv_h_list[self.switch_h])
                self.switch_h = 1 if self.switch_h == 0 else 0
            if self.tstep < self.tmax: self.qtask.enqueue(self.req_recv_h_list[self.switch_h].Start)

            # update e
            for instance in self.cpuf_m.instance_list:
                instance.update_e('post')

            # internal recv e
            self.sf_m_e.set_fields(self.gf_m_e.get_fields(), [self.gf_m_e.get_event()])




    def update_h(self):
        if '-' in self.direction:
            # update h
            for instance in self.cpuf_m.instance_list:
                instance.update_h()

            # internal send h
            self.sf_m_h.set_fields(self.gf_m_h.get_fields(), [self.gf_m_h.get_event()])

        if '+' in self.direction:
            # mpi recv e
            if self.tstep == 1: self.qtask.enqueue(self.req_recv_e_list[self.switch_e].Start)
            self.qtask.enqueue(self.req_recv_e_list[self.switch_e].Wait)
            self.sf_e.set_fields(self.tmp_recv_e_list[self.switch_e])
            self.switch_e = 1 if self.switch_e == 0 else 0
            if self.tstep < self.tmax: self.qtask.enqueue(self.req_recv_e_list[self.switch_e].Start)

            # update h
            for instance in self.cpuf_p.instance_list:
                instance.update_h('pre')

            # mpi send h
            #print 'req_send_h', rank, self.req_send_h.Test()
            if self.tstep > 1: self.qtask.enqueue(self.req_send_h.Wait)
            if self.tstep < self.tmax: 
                self.qtask.enqueue(self.req_send_h.Start, wait_for=[self.gf_h.get_event()])

            # update h
            for instance in self.cpuf_p.instance_list:
                instance.update_h('post')

            # internal recv h
            self.sf_p_h.set_fields(self.gf_p_h.get_fields(), [self.gf_p_h.get_event()])

        self.tstep += 1
