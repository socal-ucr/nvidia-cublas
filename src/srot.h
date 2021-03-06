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

#if (USE_TEX==1)
#undef fetchx
#undef fetchy
#define fetchx(i)  tex1Dfetch(texX,parms.texXOfs+(i))
#define fetchy(i)  tex1Dfetch(texY,parms.texYOfs+(i))
#else
#undef fetchx
#undef fetchy
#define fetchx(i)  sx[i]
#define fetchy(i)  sy[i]
#endif /* USE_TEX */

    int i, n, tid, totalThreads, ctaStart;
    float w, z;
    float *sx;
    float *sy;

    /* NOTE: wrapper must ensure that parms.n > 0  */

    tid = threadIdx.x;
    n = parms.n;
    sx = parms.sx; 
    sy = parms.sy;
    totalThreads = gridDim.x * blockDim.x;
    ctaStart = blockDim.x * blockIdx.x;
   
    if ((parms.incx == 0) || (parms.incy == 0)) {
        if ((blockIdx.x == 0) && (tid == 0)) {
            if ((parms.incx == 0) && (parms.incy == 0)) {
                float tw, tz;
                w = sx[0];
                z = sy[0];
                for (i = 0; i < parms.n; i++) {
                    tw = parms.sc * w + parms.ss * z;
                    tz = parms.sc * z - parms.ss * w;
                    w = tw;
                    z = tz;
                }
                sx[0] = w;
                sy[0] = z;
            } else if (parms.incx == 0) {
                int ky = (parms.incy < 0) ? ((1 - parms.n) * parms.incy) : 0;
                float temp = sx[0];
                for (i = 0; i < parms.n; i++) {
                    w = temp;
                    z = sy[ky];
                    temp   = parms.sc * w + parms.ss * z;
                    sy[ky] = parms.sc * z - parms.ss * w;
                    ky += parms.incy;
                }
                sx[0] = temp;
            } else {
                int kx = (parms.incx < 0) ? ((1 - parms.n) * parms.incx) : 0;
                float temp = sy[0];
                for (i = 0; i < parms.n; i++) {
                    w = sx[kx];
                    z = temp;
                    sx[kx] = parms.sc * w + parms.ss * z;
                    temp   = parms.sc * z - parms.ss * w;
                    kx += parms.incx;
                }
                parms.sy[0] = temp;
            }
        }
    } else if ((parms.incx == parms.incy) && (parms.incx > 0)) {
        /* equal, positive, increments */
        if (parms.incx == 1) {
            /* both increments equal to 1 */
            for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
                w = fetchx(i);
                z = fetchy(i);
                sx[i] = parms.sc * w + parms.ss * z;
                sy[i] = parms.sc * z - parms.ss * w;
            }
        } else {
            /* equal, positive, non-unit increments. */
            for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
                w = fetchx(i*parms.incx);
                z = fetchy(i*parms.incx);
                sx[i*parms.incx] = parms.sc * w + parms.ss * z;
                sy[i*parms.incx] = parms.sc * z - parms.ss * w;
            }
        }
    } else {
        /* unequal or nonpositive increments */
        int ix = ((parms.incx < 0) ? ((1 - n) * parms.incx) : 0);
        int iy = ((parms.incy < 0) ? ((1 - n) * parms.incy) : 0);
        for (i = ctaStart + tid; i < parms.n; i += totalThreads) {
            w = fetchx(ix+i*parms.incx);
            z = fetchy(iy+i*parms.incy);
            sx[ix+i*parms.incx] = parms.sc * w + parms.ss * z;
            sy[iy+i*parms.incy] = parms.sc * z - parms.ss * w;
        }
    }
