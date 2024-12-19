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
      //ensure model is initialized
      if (model->params_memory == NULL){printf("Error: model was not initialized properly.\n");
      exit(1);}

      //assign params
      size_t V = model->config.vocab_size;
      size_t Vp = model->config.padded_vocab_size;
      size_t L = model->config.num_layers;
      size_t NH = model->config.num_heads;
      size_t C = model->config.channels;

      //validate input, all should fall btwn 0 and V
      for (int i=0; i<B; i++){
        assert(0<=inputs[i] && inputs[i]>=V);
        if (targets != NULL){assert(0<=targets[i] && targets[i]<V); }

      }

      //allocate space for activations
      if (model->acts_memory == NULL){
        //record the current B, T
        model->batch_size = B;
        model->seq_len = T;

        //allocate memory
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        size_t num_activations = 0;
        for (size_t i=0; i<NUM_ACTIVATION_TENSORS; i++){
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        //also create memory for caching input and targets
        model->inputs = (int*)mallocCheck(B*T*sizeof(int));    //casting to int
        model->targets = (int*)mallocCheck(B*T*sizeof(int));

      } else {
        //validate B,T is consistent with previous allocation
        if (B != model->batch_size || T != model->seq_len){
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
            exit(EXIT_FAILURE);     
        }
      }
          // cache the inputs/targets
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL) {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_preatt = acts.preatt + l * B * NH * T * T;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) { mean_loss += model->acts.losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
    } else {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }

}

void gpt2_zero_grad(GPT2* model){
    if(model->grads_memory != NULL){memset(model->grads_memory, 0, model->num_parameters*sizeof(float)); }
    if(model->grads_acts_memory != NULL){memset(model->grads_acts_memory, 0, model->num_activations*sizeof(float)); }
}

void gpt2_backward(GPT2* model){
    // double check we forwarded previously

    if (model->mean_loss == -1.0f){
        printf("Error: must forward with targets before backward");
        exit(1);
    }

    //allocate memory for gradients of the weights and activations
    if (model->grads_memory == NULL){
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    //name variables
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    //for backward pass: go in reverse order of the forward pass, and call backward()
    ParameterTensors params = model->params;
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    //we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    //final loss is the mean over all losses for a batch.
    float dloss_mean = 1.0f/ (B*T);
    for (int i=0; i<B*T; i++){ grads_acts.losses[i] = dloss_mean; } //assigining the same loss value 

    crossenropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, NULL, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    float* residual = acts.residual3 + (L-1)*B*T*C;
    float* dresidual = grads_acts.residual3 + (L-1)*B*T*C;
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    for (int l=L-1; l>=0; l--){
        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        float* dl_ln1w = grads.ln1w + l * C;
        float* dl_ln1b = grads.ln1b + l * C;
        float* dl_qkvw = grads.qkvw + l * 3*C * C;
        float* dl_qkvb = grads.qkvb + l * 3*C;
        float* dl_attprojw = grads.attprojw + l * C * C;
        float* dl_attprojb = grads.attprojb + l * C;
        float* dl_ln2w = grads.ln2w + l * C;
        float* dl_ln2b = grads.ln2b + l * C;
        float* dl_fcw = grads.fcw + l * 4*C * C;
        float* dl_fcb = grads.fcb + l * 4*C;
        float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        float* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        float* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        float* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        float* dl_atty = grads_acts.atty + l * B * T * C;
        float* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        float* dl_att = grads_acts.att + l * B * NH * T * T;
        float* dl_attproj = grads_acts.attproj + l * B * T * C;
        float* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        float* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        float* dl_fch = grads_acts.fch + l * B * T * 4*C;
        float* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        float* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        float* dl_residual3 = grads_acts.residual3 + l * B * T * C;

        // backprop this layer
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);      
    }

void gpt2_update(GPT2* model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t){
    //generally an AdamW optimizer according to pytorch docs

    //allocating memoru
    if (model->m_memory == NULL){
        model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
        model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));
    }

    for (size_t i=0; i<model->num_parameters; i++){
        float param = model->params_memory[i];
        float grad = model->grads_memory[i];

        // update first moment
        float m = beta1 * model->m_memory[i] + (1.0f - beta1)*grad;
        // update the second moment
        float v = beta2 * model->v_memory[i] + (1.0f - beta2)*grad*grad;
        //bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        //update
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay*param);

    }
}

void gpt2_free(GPT2* model){
    free(model->params_memory);
    free(model->grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
    free(model->targets);
}



// this part should only run on the train.c file

#ifndef TESTING      //if testing is not defined within this file

unsigned int random_u32(uint64_t* state){  //random number generator, input is a state, output is a random number
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A

    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;

    return (*state * 0x2545F4914F6CDD1Dull) >> 32; 

}

float random_f32(uint64_t* state){  // random float btwn 0 and 1
    return (random_u32(state) >> 8) / 16777216.0f;
    }

int sample_mult(float* probabilities, int n, float coin){
    //sampling from a multinomial
    //coin is a random number between 0 and 1
    float cdf = 0.0f;
    for (int i=0; i<n; i++){
        cdf += probabilities[i];
        if (coin < cdf){
            return i;
        }
    }
    return n-1;
}


//main training loop

int main(){

    //initialize model
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_117M_model.bin");  //bin file should be generated first from train.py

    //build the DataLoader from token files
    const char* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    const char* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    int B = 4;
    int T = 64;
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B*T));
    printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B*T));
    int val_num_batches = 5;

    //build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    //some memory fot generating samples from the model
    uint64_t rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B*T*sizeof(int));
    const int genT = 64; // length of generated samples

    //train
    struct timespec start, end;
    for (int step=0; step<=40; step++){

        //estimate val loss once in a while
        if (step % 10 == 0){
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i=0; i<val_num_batches; i++){
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
        }

        //once in a while do inference
        if (step > 0 && step % 20 ==0){
            //fill gent_tokens with GPT2_EOT, which kicks off generation
            for(int i=0; i<B*T; i++){ gen_tokens[i] = tokenizer.eot_token; }

            //generate
            for (int i=1; t<genT; t++){
                gpt2_forward(&model, gen_tokens, NULL, B, T);
                
                //take only 0 position batch
                float* probs = model.acts.probs + (t-1)*model.config.padded_vocab_size;
                float coin = random_f32(&rng_state);

                int next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;

                //print using the tokenizer
                if (tokenizer.init_ok){
                    const char* token_str = tokenizer.decode(&tokenizer, new_token);
                    safe_printf(token_str);
                } else {
                    //print the token id
                    safe_printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n------\n");
        }
        //do training
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 0.00004f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
        printf("step %d, loss %f, time elapsed %f\n", step, model.mean_loss, time_elapsed_s);
    }

    //free memory
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(gen_tokens);
    return 0;





}







#endif