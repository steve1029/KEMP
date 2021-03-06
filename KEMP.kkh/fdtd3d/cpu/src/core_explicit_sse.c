/* 
Author  : Ki-Hwan Kim
Purpose : C functions to update the core of FDTD
Target  : CPU
Created : 2012-02-09
Modified: 
*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <MM_HEADER>
#define LOADU _mm_loadu_PSD // not aligned to 16 bytes
#define LOAD _mm_load_PSD	
#define STORE _mm_store_PSD
#define ADD _mm_add_PSD
#define SUB _mm_sub_PSD
#define MUL _mm_mul_PSD
#define SET1 _mm_set_PSD1


static PyObject *update_e(PyObject *self, PyObject *args) {
	int nx, ny, nz;
	float *ex, *ey, *ez, *hx, *hy, *hz;
	PyArrayObject *Ex, *Ey, *Ez, *Hx, *Hy, *Hz;
	COEFF_E float *cx, *cy, *cz;
	COEFF_E PyArrayObject *Cx, *Cy, *Cz;
	
	if(!PyArg_ParseTuple(args, "PARSE_ARGS_CE", &Ex, &Ey, &Ez, &Hx, &Hy, &Hz PYARRAYOBJECT_CE)) return NULL;
	nx = (int)(Ex->dimensions)[0];
	ny = (int)(Ex->dimensions)[1];
	nz = (int)(Ex->dimensions)[2];
	ex = (DTYPE*)(Ex->data);
	ey = (DTYPE*)(Ey->data);
	ez = (DTYPE*)(Ez->data);
	hx = (DTYPE*)(Hx->data);
	hy = (DTYPE*)(Hy->data);
	hz = (DTYPE*)(Hz->data);
	COEFF_E cx = (DTYPE*)(Cx->data);
	COEFF_E cy = (DTYPE*)(Cy->data);
	COEFF_E cz = (DTYPE*)(Cz->data);
	
	int idx;
	TYPE128 fx0, fy0, fz0, f1, f2;

	SET_NUM_THREADS
    OMPPy_BEGIN_ALLOW_THREADS
    OMP#pragma omp parallel for private(idx, fx0, fy0, fz0, f1, f2) shared(ex, ey, ez, hx, hy, hz)
	for( idx=0; idx<(nx*ny-1)*nz; idx+=STEP ) {
        fx0 = LOAD(hx+idx);
        fy0 = LOAD(hy+idx);
        fz0 = LOAD(hz+idx);

		f1 = LOAD(hz+idx+nz);
		f2 = LOADU(hy+idx+1);
		STORE(ex+idx, ADD(LOAD(ex+idx), MUL(CEX, SUB(SUB(f1, fz0), SUB(f2, fy0)))) ); 

        if( idx < (nx-1)*ny*nz ) {
            f1 = LOADU(hx+idx+1);
            f2 = LOAD(hz+idx+ny*nz);
            STORE(ey+idx, ADD(LOAD(ey+idx), MUL(CEY, SUB(SUB(f1, fx0), SUB(f2, fz0)))) );
            
            f1 = LOAD(hy+idx+ny*nz);
            f2 = LOAD(hx+idx+nz);
            STORE(ez+idx, ADD(LOAD(ez+idx), MUL(CEZ, SUB(SUB(f1, fy0), SUB(f2, fx0)))) );
        }
	}
	OMPPy_END_ALLOW_THREADS
	
	// zero-fixed boundaries
	int i, j, k;
	for(i=0; i<nx; i++) {
		for(j=0; j<ny; j++) {
			idx = i*ny*nz + j*nz + (nz-1);
			ex[idx] = 0;
			ey[idx] = 0;
		}
		for(k=0; k<nz; k++) {
			idx = i*ny*nz + (ny-1)*nz + k;
			ex[idx] = 0;
			ez[idx] = 0;
		}
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}



static PyObject *update_h(PyObject *self, PyObject *args) {
	int nx, ny, nz;
	float *ex, *ey, *ez, *hx, *hy, *hz;
	PyArrayObject *Ex, *Ey, *Ez, *Hx, *Hy, *Hz;
	COEFF_H float *cx, *cy, *cz;
	COEFF_H PyArrayObject *Cx, *Hy, *Cz;
	
	if(!PyArg_ParseTuple(args, "PARSE_ARGS_CH", &Ex, &Ey, &Ez, &Hx, &Hy, &Hz PYARRAYOBJECT_CH)) return NULL;
	nx = (int)(Ex->dimensions)[0];
	ny = (int)(Ex->dimensions)[1];
	nz = (int)(Ex->dimensions)[2];
	ex = (DTYPE*)(Ex->data);
	ey = (DTYPE*)(Ey->data);
	ez = (DTYPE*)(Ez->data);
	hx = (DTYPE*)(Hx->data);
	hy = (DTYPE*)(Hy->data);
	hz = (DTYPE*)(Hz->data);
	COEFF_H cx = (DTYPE*)(Cx->data);
	COEFF_H cy = (DTYPE*)(Cy->data);
	COEFF_H cz = (DTYPE*)(Cz->data);
	
	int idx;
	TYPE128 fx0, fy0, fz0, f1, f2;

	SET_NUM_THREADS
    OMPPy_BEGIN_ALLOW_THREADS
    OMP#pragma omp parallel for private(idx, fx0, fy0, fz0, f1, f2) shared(ex, ey, ez, hx, hy, hz)
	for( idx=nz; idx<nx*ny*nz; idx+=STEP ) {
		cf = SET1(0.5);
        fx0 = LOAD(ex+idx);
        fy0 = LOAD(ey+idx);
        fz0 = LOAD(ez+idx);

		f1 = LOAD(ez+idx-nz);
		f2 = LOADU(ey+idx-1);
		STORE(hx+idx, SUB(LOAD(hx+idx), MUL(CHX, SUB(SUB(fz0, f1), SUB(fy0, f2)))) );

        if( idx > ny*nz ) {
            f1 = LOADU(ex+idx-1);
            f2 = LOAD(ez+idx-ny*nz);
            STORE(hy+idx, SUB(LOAD(hy+idx), MUL(CHY, SUB(SUB(fx0, f1), SUB(fz0, f2)))) );
            
            f1 = LOAD(ey+idx-ny*nz);
            f2 = LOAD(ex+idx-nz);
            STORE(hz+idx, SUB(LOAD(hz+idx), MUL(CHZ, SUB(SUB(fy0, f1), SUB(fx0, f2)))) );
        }
	}	OMPPy_END_ALLOW_THREADS

	// zero-fixed boundaries
	int i, j, k;
	for(i=0; i<nx; i++) {
		for(j=0; j<ny; j++) {
			idx = i*ny*nz + j*nz;
			hx[idx] = 0;
			hy[idx] = 0;
		}
		for(k=0; k<nz; k++) {
			idx = i*ny*nz + k;
			hx[idx] = 0;
			hz[idx] = 0;
		}
	}

	Py_INCREF(Py_None);
	return Py_None;
}



static PyMethodDef ufunc_methods[] = {
	{"update_e", update_e, METH_VARARGS, ""},
	{"update_h", update_h, METH_VARARGS, ""},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcore() {
	Py_InitModule("core", ufunc_methods);
	import_array();
}