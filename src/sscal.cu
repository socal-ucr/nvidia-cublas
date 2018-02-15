/*
 * Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  
 *
 * This software and the information contained herein is being provided 
 * under the terms and conditions of a Source Code License Agreement.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* This file contains the implementation of the BLAS-1 function sscal */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

texture<float> texX;

__global__ void sscal_gld_main (struct cublasSscalParams parms);
__global__ void sscal_tex_main (struct cublasSscalParams parms);

/*
 * void
 * cublasSscal (int n, float alpha, float *x, int incx)
 *
 * replaces single precision vector x with single precision alpha * x. For i 
 * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx], 
 * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  single precision scalar multiplier
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * x      single precision result (unchanged if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/sscal.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 * 
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasSscal (int n, float alpha, float *x, int incx)
{
    struct cublasContext *ctx = CUBLAS_GET_CTX();
    struct cublasSscalParams params;
    cudaError_t cudaStat;
    int nbrCtas;
    int elemsPerCta;
    int threadsPerCta;
    int sizeX = n * (imax (1, abs(incx)));
    size_t texXOfs = 0;
    int useTexture;

    if (!cublasInitialized (ctx)) {
        cublasSetError (ctx, CUBLAS_STATUS_NOT_INITIALIZED);
        return;
    }
    /* early out if nothing to do */
    if ((n <= 0) || (incx <= 0)) {
        return;
    }
    
    useTexture = sizeX < CUBLAS_MAX_1DBUF_SIZE;

    /* Currently, the overhead for using textures is high. Do not use texture
     * for vectors that are short, or those that are aligned and have unit
     * stride and thus have nicely coalescing GLDs.
     */
    if ((n < 100000) || /* experimental bound */
        ((sizeX == n) && (!(((uintptr_t) x) % CUBLAS_WORD_ALIGN)))) {
        useTexture = 0;
    }

    if (useTexture) {
        if ((cudaStat=cudaBindTexture (&texXOfs,texX,x,sizeX*sizeof(x[0]))) != 
            cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_MAPPING_ERROR);
            return;
        }
        texXOfs /= sizeof(x[0]);
    }

    memset (&params, 0, sizeof(params));
    params.n  = n;
    params.sx = x;
    params.sa = alpha;
    params.incx = incx;
    params.texXOfs = (int)texXOfs;

    cublasVectorSplay (n, CUBLAS_SSCAL_THREAD_MIN, CUBLAS_SSCAL_THREAD_MAX,
                       CUBLAS_SSCAL_CTAS_MAX, &nbrCtas, &elemsPerCta, 
                       &threadsPerCta);

    cudaStat = cudaGetLastError(); /* clear error status */
    if (useTexture) {
        sscal_tex_main<<<nbrCtas,threadsPerCta>>>(params);
    } else {
        sscal_gld_main<<<nbrCtas,threadsPerCta>>>(params);
    }
    cudaStat = cudaGetLastError(); /* check for launch error */
    if (cudaStat != cudaSuccess) {
        cublasSetError (ctx, CUBLAS_STATUS_EXECUTION_FAILED);
    }

    if (useTexture) {
        if ((cudaStat = cudaUnbindTexture (texX)) != cudaSuccess) {
            cublasSetError (ctx, CUBLAS_STATUS_INTERNAL_ERROR);
        }
    }
}

__global__ void sscal_gld_main (struct cublasSscalParams parms)
{
#undef  USE_TEX
#define USE_TEX 0
#include "sscal.h"
}

__global__ void sscal_tex_main (struct cublasSscalParams parms)
{
#undef  USE_TEX
#define USE_TEX 1
#include "sscal.h"
}
