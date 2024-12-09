// this file contains training code for GPT-2 model
// runs on CPU
// it utilizes few OpenMP pragmas to induce large speedup at low cost


#include <stdio.h>
#include <stdlib.h>     //working with malloc(), free(), exit(), system()
#include <ctype.h>      // char operations
#include <stdint.h>     // for integer and float precision
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>     //file manipulation and system info

#ifndef OMP
#include <omp.h>
#endif

//previously defined utility

#include "headers/utils.h"      //define fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck

#include "headers/tokenizer.h"  //defines tokenizer_init, tokenizer_decode, tokenizer_free

#include "headers/dataloader.h"     // dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free


// B = batch_size, T = sequence length, C = channels, V = vocab_size

void encoder_forward(float* out, int* inp, float* wte, float* wpe,
                        int B, int T, int C) {
                            // out is (B, T, C) where for each T there is a corresponding C-dimensional vector summarizing its position
                            // in  is (B, T)  just batches raw input sequence
                            // wte is (V, C) a single embedding for each vocab
                            // wpe is (maxT, C) all possible position have an embedding
                            for (int b=0; b<B; b++){
                                for (int t=0; t<T; t++){  // for each vocab in sequence

                                    //find the output position in out [b,t, :]
                                    float* out_bt = out + b * T * C + t * C;
                                    //get the index of the token at inp[b, t]
                                    int ix = inp[b * T + t];
                                    //seek to the position of wte corresponding to token t
                                    float* wte_ix = wte + ix * C;
                                    //seek to the position of wpe corresponding to position
                                    float* wpe_ix = wpe + t * C;
                                    // add the two vectors and store result
                                    for (int i=0; i<C; i++){
                                        out_bt[i] = wte_ix[i] + wpe_t[i]
                                    }


                                }
                            }
                        }


void encode_backward(float* dwte, float* dwpe, float* dout,
                        int* inp, int B, int T, int C){
                            for (int b=0; b<B; b++){
                                for (int t=0; t<T; t++){
                                    float* dout_bt = dout + b * T * C + t * C;
                                    int ix = inp[b * T + t];
                                    float* dwte_ix = dwte + ix * C;
                                    float* dwpe_t = dwpe + t * C;
                                    for (int i=0; i<C; i++){
                                        float d = dout_bt[i];
                                        dwte_ix[i] += d;
                                        dwpe_t[i] += d;
                                    }
                                }
                            }
                        }


void layernorm_forward(float* out, float* mean, float* rstd,
                        float* inp, float* weight, float* bias,
                        int B, int T, int C){
                            //inp and out are (B, T, C)
                            //mean and rstd ar (B, T)
                            float eps = 1e-5f;
                            for (int b=0; b<B; b++){
                                for (int t=0; t<T; t++){
                                    //seek to input position
                                    float * x = inp + b * T * C + t * C;  //pointer arithmetic hence inp
                                    // calculate the mean
                                    float m = 0.0f;
                                    for (int i=0; i<C; i++){
                                        m += x[i];
                                    }
                                    m = m/C;
                                    // calculate the variance without bias
                                    float v = 0.0f;
                                    for (int i=0; i<C; i++){
                                        float xshift = x[i] - m;
                                        v += xshift * xshift;
                                    }
                                    v = v/C;
                                    //calculate the rstd
                                    float s = 1.0f/ sqrt(v + eps);
                                    // seek to the output position in out
                                    float* out_bt = out + b * T * C + t * C;
                                    for (int i=0; i<C; i++){
                                        float n = (s * (x[i] - m)); //normalize
                                        float o = n * weight[i] + bias[i]; //scale and shift
                                        out_bt[i] = o;   //write to output
                                    }
                                    //cache the meand and rstd for backward pass
                                    mean[b * T + t] = m;
                                    rstd[b * T + t] = s;

                                                                    }
                            }
                        }


void layernorm_backward(float* dinp, float*dweight, float*dbias, float* dout, float* inp,
                        float* weight, float* mean, float* rstd, int B, int T, int C){
                            for (int b=0; b<B; b++){
                                for (int t=0; t<T; t++){
                                    float* dout_bt = dout + b * T * C + t * C;   //pointer arithmetic
                                    float* inp_bt = inp + b * T * C + t * C;
                                    float* dinp_bt = dinp + b * T * C + t * C;
                                    float mean_bt = mean[b * T + t];
                                    float rstd_bt = rstd[b * T + t];

                                    // first: two reduce operations
                                    float dnorm_mean = 0.0f;
                                    float dnorm_norm_mean = 0.0f;
                                    for (int i=0; i<C; i++){
                                        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm_i = weight[i] * dout_bt[i];
                                        dnorm_mean += dnorm_i;
                                        dnorm_norm_mean += dnorm_i * norm_bti;
                                    
                                    }
                                    dnorm_mean = dnorm_mean/C;
                                    dnorm_norm_mean = dnorm_norm_mean;

                                    // iterate and accumulate gradients
                                    for ( int i = 0; i<C; i++){
                                        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                                        float dnorm_i = weight[i] * dout_bt[i];
                                        //gradient contribution to bias
                                        dbias[i] += dout_bt[i];
                                        //gradient contribution to weight
                                        dweight[i] += norm_bti * dout_bt[i];
                                        // gradient contribution to input
                                        float dval = 0.0f;
                                        dval += dnorm_i; //term 1
                                        dval -= dnorm_mean; //term 2
                                        dval -= norm_bti * dnorm_norm_mean; //term 3
                                        dval *= rstd_bt; //final scale
                                        dinp_bt[i] += dval;
                                    }
                                }
                            }
                        }