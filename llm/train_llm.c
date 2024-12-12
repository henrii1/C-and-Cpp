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
                                    float* wpe_t = wpe + t * C;
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


void matmul_forward_naive(float* out, const float* inp, const float* weight,
                            const float* bias, int B, int T, int C, int OC){
                                //naive implementation of matrix multiplication
                                #pragma omp parallel for collapse(2) // for parallelizing matrix multiplication
                                for (int b=0; b<B; b++){
                                    for (int t=0; t<T; t++){
                                        //seek to position
                                        int bt = b * T + t;
                                        for (int o=0; o<OC; o++){
                                            float val = (bias != NULL) ? bias[o] : 0.0f;
                                            for (int i = 0; i<C; i++){
                                                val += inp[bt * C + i] * weight[o*C + i];
                                            }
                                            out[bt * OC + o] = val;
                                        }

                                    }
                                }
                            }


void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC){
                        //mildly optimized implementation of matmul forward
                        
                        //default to naive if modulus isn't zero
                        const int LOOP_UNROLL = 8;
                        if (B*T % LOOP_UNROLL != 0){
                            matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
                            return;
                        }

                        //convert B and T into one strided loop
                        #pragma omp parallel for  //parellizes loops
                        for (int obt=0; obt<B*T; obt += LOOP_UNROLL){
                            for (int o=0; o<OC; o++){
                                float result[LOOP_UNROLL];

                                //initialize bias
                                for (int ibt=0; ibt<LOOP_UNROLL; ibt++){
                                    result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
                                }

                                //a bit of caching
                                for (int i=0; i<C; i++){
                                    float w = weight[i + o * C];
                                    for (int ibt=0; ibt<LOOP_UNROLL; ibt++){
                                        int bt = obt + ibt;
                                        result[ibt] += inp[bt * C + i] * w;
                                    }
                                }
                                // write back results to main memory
                                for (int ibt=0; ibt<LOOP_UNROLL; ibt++){
                                    int bt = obt + ibt;
                                    out[bt*OC + o] = result[ibt];
                                }
                            }
                        }

                    }



void matmul_backward(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const float* weight,
                        int B, int T, int C, int OC){

                            //backward into inp first
                            #pragma omp parallel for collapse(2)
                            for (int b=0; b<B; b++){
                                for (int t=0; t<T; t++){
                                    const float* dout_bt = dout + b * T * OC + t * OC;
                                    float* dinp_bt = dinp + b * T * C + t * C;
                                    for (int o=0; o<OC; o++){
                                        const float* wrow = weight + o*C;
                                        float d = dout_bt[o];
                                        for (int i=0; i<C; i++){
                                        dinp_bt[i] += wrow[i] * d;
                                        }
                                    }
                                }
                            }
                            //backward into weight/bias, parellized over output channels OC
                            #pragma omp parallel for
                            for (int o=0; o<OC; o++){
                                for (int b=0; b<B; b++){
                                    for (int t=0; t<T; t++){
                                        const float* dout_bt = dout + b*T*OC + t*OC;
                                        const float* inp_bt = inp + b * T * C + t * C;
                                        float* dwrow = dweight + o*C;
                                        float d = dout_bt[o];
                                        if (dbias != NULL) {dbias[o] += d; }
                                        for (int i=0; i<C; i++){
                                            dwrow[i] += inp_bt[i] * d;
                                        }
                                    }
                                }
                            }
                        }


void attention_forward(float* out, float* preatt, float* att, float* inp,
                       int B, int T, int C, int NH){
                        // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
                        // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
                        // that holds the pre-attention and post-attention scores (used in backward)
                        // output is (B, T, C)
                        // attention is the only layer that mixes information across time
                        // every other operation is applied at every (b,t) position independently
                        // (and of course, no layer mixes information across batch)
                        int C3 = C*3;
                        int hs = C/NH; // head size
                        float scale = 1.0/sqrtf(hs);

                        #pragma omp parallel for collapse(3)  //run parallel for 3 nested loops 
                        for (int b=0; b<B; b++){
                            for (int t=0; t<T; t++){
                                for (int h=0; h<NH; h++){
                                    //seek to positions
                                    float* query_t = inp + b*T*C3 + t*C3 + h*hs;
                                    float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                                    float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                                    //calculate query dot key and maxval
                                    float maxval = -10000.0f;
                                    for (int t2=0; t2<=t; t2++){
                                        float* key_t2 = inp + b*T*C3 + t2*C3 + h*hs + C;

                                        // dot product
                                        float val = 0.0f;
                                        for (int i=0; i<hs; i++){
                                            val += query_t[i]* key_t2[i];
                                        }
                                        val *= scale;
                                        if (val > maxval) {maxval = val;}
                                        preatt_bth[t2] = val;

                                    }
                                    //calculate esp and keep track of sum
                                    float expsum = 0.0f;
                                    for (int t2=0; t2<=t; t2++){
                                        float expv = expf(preatt_bth[t2] - maxval);
                                        expsum += expv;
                                        att_bth[t2] = expv;
                                    }

                                    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f/expsum;

                                    //normalize to get the softmax
                                    for(int t2=0; t2<T; t2++){
                                        if (t2 <= t){
                                            att_bth[t2] *= expsum_inv;
                                        } else {
                                            att_bth[t2] = 0.0f;
                                        }
                                    }

                                    // accumulate weighted values into the output of attention
                                    float* out_bth = out + b * T * C + t * C + h * hs;
                                    for (int i=0; i<hs; i++) {out_bth[i] = 0.0f; }
                                    for (int t2=0; t2<t; t2++){
                                        float* value_t2 = inp + b*T*C3 + t2*C3 + h*hs + C*2;
                                        float att_btht2 = att_bth[t2];
                                        for (int i=0; i<hs; i++){
                                            out_bth[i] += att_btht2*value_t2[i];
                                        }
                                    }


                                }
                            }
                        }

                       }


void attention_backward(float* dinp, float* dpreatt, float* datt, float* dout,
                        float* inp, float* att, int B, int T, int C, int NH){
                             // inp/dinp are (B, T, 3C) Q,K,V
                            // att/datt/dpreatt are (B, NH, T, T)
                            // dout is (B, T, C)
                            int C3 = C*3;
                            int hs = C/NH; //head size
                            float scale = 1.0f/sqrtf(hs);

                            for (int b=0; b<B; b++){
                                for (int t=0; t<T; t++){
                                    for (int h=0; h<NH; h++){
                                        float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                                        float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                                        float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                                        float* dquery_t = dinp + b*T*C3 + t*C3 + h*hs;
                                        float* query_t = inp + b*T*C3 + t*C3 + h*hs;

                                        //backward pass 4, through the value accumulation
                                        float* dout_bth = dout + b*T*C + t*C + h*hs;
                                        for (int t2=0; t2<=t; t2++){
                                            float* value_t2 = inp + b*T*C3 + t2*C3 + h*hs + C*2;
                                            float* dvalue_t2 = dinp + b*T*C3 + t2 * C3 + h*hs + C*2;
                                            for (int i=0; i<hs; i++){
                                                datt_bth[t2] += value_t2[i]*dout_bth[i];
                                                dvalue_t2[i] += att_bth[t2]*dout_bth[i];
                                            }
                                        }
                                        //backward pass 2&3, then softmax
                                        for (int t2=0; t2<=t; t2++){
                                            for (int t3=0; t3<=t; t3++){
                                                float indicator = t2 == t3 ? 1.0f : 0.0f;
                                                float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                                                dpreatt_bth[t3] += local_derivative*datt_bth[t2];
                                            }
                                        }

                                        //backward pass 1, the q@k matmul
                                        for (int t2=0; t2<=t; t2++){
                                            float* key_t2 = inp + b*T*C3 + t2*C3 + h*hs + C;
                                            float* dkey_t2 = dinp + b*T*C3 + t2*C3 + h*hs + C;
                                            for (int i=0; i<hs; i++){
                                                dquery_t[i] += key_t2[i]*dpreatt_bth[t2]*scale;
                                                dkey_t2[i] += query_t[i]*dpreatt_bth[t2]*scale;
                                            }
                                        }

                                    }
                                }
                            }
                        }



#define GELU_SCALING_FACTOR sqrtf(2.0f/M_PI)      //M_PI is pi

void gelu_forward(float* out, float* inp, int N){
    for (int i=0; i<N; i++){
        float x = inp[i];
        float cube = 0.044715f*x*x*x;
        out[i] = 0.5f*x*(1.0f + tanhf(GELU_SCALING_FACTOR*(x + cube)));
    }
}


// Ofast optimization
#pragma float_control(precise, on, push)
#if defined(__GNUC__)&& !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif

void gleu_backward(float* dinp, float* inp, float* dout, int N){
    for (int i=0; i<N; i++){
        float x = inp[i];
        float cube = 0.044715f*x*x*x;
        float tanh_arg = GELU_SCALING_FACTOR*(x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sec_out = 1.0f/(coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x*0.5f*sec_out*GELU_SCALING_FACTOR*(1.0f + 3.0f*0.044715f*x*x);
        dinp[i] += local_grad*dout[i];
    }
}

#pragma float_control(pop)   //end

void residual_forward(float* out, float* inp1, float* inp2, int N){
    for (int i=0; i<N; i++){
        out[i] = inp1[i] + inp2[i];
    }
}

void residual_backward(float* dinp1, float* dinp2, float* dout, int N){
    for (int i=0; i<N; i++){
        dinp1[i] +=dout[i];
        dinp2[i] += dout[i];
    }
}


void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp){
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #pragma omp parallel for collapes(2)
    for (int b=0; b<B; b++){
        for (int t=0; t<T; t++){
            // probs <- softma(logits)
            float* logits_bt = logits + b*T*Vp + t*Vp;
            float* probs_bt = probs + b*T*Vp + t*Vp;

            //maxval is only calculated and substracted for numerical stability
            float maxval = -10000.0f;
            for (int i=0; i<V; i++){
                if (logits_bt[i] > maxval) {maxval = logits_bt[i]; }

            }
            float sum = 0.0f;
            for (int i=0; i<V; i++){
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // loop to V, leaving the padded dim
            for (int i=0; i<V; i++){
                probs_bt[i] /= sum;
            }
            //for safety, lets force the probabilities to zero
            for (int i=V; i<Vp; i++){
                probs_bt[i] = 0.0f;
            }
        }
    }
}


void crossentropy_forward(float* losses, float* probs, int* targets,
                            int B, int T, int Vp){
                                // output: losses is (B,T) of the individual losses at each position
                                // input: probs are (B,T,Vp) of the probabilities
                                // input: targets is (B,T) of integers giving the correct index in logits
                                for(int b=0; b<B; b++){
                                    for (int t=0; t<T; t++){
                                        float* probs_bt = probs + b*T*Vp + t*Vp;
                                        int ix = targets[b*T + t];
                                        losses[b*T + t] = -logf(probs_bt[ix]);
                                    }
                                }

                            }


                            void crossenropy_softmax_backward(float* dlogits, float* dlosses, float* probs, int* targets,
                                                                int B, int T, int V, int Vp){
                                                                    for (int b=0; b<B; b++){
                                                                        for (int t=0; t<T; t++){
                                                                            float* dlogits_bt = dlogits + b*T*Vp + t*Vp;
                                                                            float* probs_bt = probs + b*T*Vp + t*Vp;
                                                                            float dloss = dlosses[b*T + t];
                                                                            int ix = targets[b*T + t];

                                                                            for (int i=0; i<V; i++){
                                                                                float p = probs_bt[i];
                                                                                float indicator = i ==ix ? 1.0f : 0.0f;
                                                                                dlogits_bt[i] += (p - indicator) * dloss;
                                                                            }
                                                                        }
                                                                    }
                                                                }


                /* GPT-2 Model Definition*/

typedef struct {
    int max_seq_len;  // max sequence length e.g 1024
    int vocab_size;  
    int padded_vocab_size;  //nice number e.g %128==0
    int num_layers;
    int num_heads;    //num attention heads  e.g 12
    int channels;  //dmodel e.g 768
} GPT2Config;

#define NUM_PARAMETER_TENSORS 16

typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C) linear shape
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w;  // (L, C)
    float* ln2b;  // (L, C)
    float* fcw;   // (L, 4*C, C) raise size 4x
    float* fcb;   //  (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw;   // (C)
    float* lnfb;   // (C)
} ParameterTensors;


void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config){
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;

    //param sizes array
    param_sizes[0] = Vp*C; //wte
    param_sizes[1] = maxT*C; //wpe
    param_sizes[2] = L*C;    //ln1w
    param_sizes[3] = L*C;    //ln1b
    param_sizes[4] = L* (3*C) *C; //qkvw
    param_sizes[5] = L * (3*C); //qkvb
    param_sizes[6] = L * C * C; //attnprojw
    param_sizes[7] = L * C;  //attnprojb
    param_sizes[8] = L * C;  //ln2w
    param_sizes[9] = L * C;  //ln2b
    param_sizes[10] = L * (4 * C) * C;  //fcw
    param_sizes[11] = L * (4 * C);  //fcb
    param_sizes[12] = L * C * (4 * C); //fcprojw
    param_sizes[13] = L * C;  //fcprojb
    param_sizes[14] = C;    //lnfw
    param_sizes[15] = C;    //lnfb

}

//allocate memories and point tensors
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i=0; i<NUM_PARAMETER_TENSORS; i++){
        num_parameters += param_sizes[i];
    }
    //malloc all params at once
    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));

    // assign all the tensors
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };  //& for linking original param and not a copy
    float* params_memory_iterator = params_memory;
    for (size_t i=0; i<NUM_PARAMETER_TENSORS; i++){
        *(ptrs[i]) = params_memory_iterator;   //assign the references to actual storage
        params_memory_iterator += param_sizes[i];  //to ensure that each points to a different, adequate memory chunk
    }
    return params_memory;   // will be same as ptrs[0]
}


#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, V)
    float* probs; // (B, T, V)
    float* losses; // (B, T)
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, int B, int T) {
    size_t C = config.channels;
    size_t NH = config.num_heads;
    size_t L = config.num_layers;
    size_t Vp = config.padded_vocab_size;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3 * C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T; // losses
}

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory = (float*)mallocCheck(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

typedef struct {
    GPT2Config config;
    //weights (params) of the model and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    //gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    //buffer for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    //activations of model
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size;
    int seq_len;
    int* inputs;
    int* targets;
    float mean_loss;
} GPT2;


void gpt2_build_from_checkpoint(GPT2* model, const char* checkpoint_path){
    // read in model from a checkpoint file
    FILE* model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326){printf("Bad magic model file \n"); exit(1);}
    if (model_header[1] != 3){
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run python train file"); exit(1);
    }
    //model struct is a class the stores the model. with pointers to memory used and static variables

    //read hyperparemeters
    size_t maxT, V, Vp, L, NH, C;  //size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2]; //all others are empty, only model_header has a valu
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);      // zu for size_t
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    //allocate space for all params and read them in
    fill_in_parameter_sizes(model->param_sizes, model->config);

    //count the number of params
    size_t num_parameters = 0;
    for (size_t i=0; i<NUM_PARAMETER_TENSORS; i++){
        num_parameters = model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    //read in all the parameters from the file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    //other inits
    model->acts_memory = NULL;
    model-> grads_memory = NULL;
    model-> m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model-> seq_len = 0;
    model->mean_loss = -1.0f;
}

void gpt2_forward(GPT2* model, int* inputs, int* targets, size_t B, size_t T){
      
}

