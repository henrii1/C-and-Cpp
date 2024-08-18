//GPT2 Decoder (token to string)

#include <stdint.h>  //for uint-8, 16 etc.
#include <ctype.h>   //char classification eg isalnum etc
#include <assert.h>  //assertion macro

#include "utils.h"

typedef struct {
    uint32_t vocab_size;
    char **token_table;   //pointer to array of pointers
    int  init_ok;
    int eot_token;  
} Tokenizer;


void safe_printf(const char* piece){
    //print out only printable tokens
    if (piece == NULL) {return;}
    if (piece[0] == "\0"){return;}
    //every token is asserted to be only 1 byte
    if (piece[1] == '\0'){
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val)|| isspace(byte_val))){
            return;
        }
    }
    printf("%s", piece);
}


void tokenizer_init(Tokenizer* tokenizer, const char* filename){
    FILE* file = fopen(filename, "rb");
    if (file == NULL){
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        return;
    }

    uint32_t header[256];   //size array
    freadCheck(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20240328);  //the freadcheck reads 256 vals into array
    int version = header[1];
    tokenizer->vocab_size = header[2];
    if (version == 1){
       // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        assert(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } else if (version == 2) {
        tokenizer->eot_token = header[3];
    } else {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
  
    }
    // read tokens
    unsigned char length;
    tokenizer->token_table = (char**)mallocCheck(tokenizer->vocab_size*sizeof(char*));
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++){    //tokenizer->vocab_size is not int hence i shouldn't be
        freadCheck(&length, sizeof(unsigned char), 1, file);  //i isn't used but file pointer points to the next
        assert(length > 0);  //every token should be at least one character
        char* token_bytes = (char*)mallocCheck(length+1);  //in c not all variables must be used
        freadCheck(token_bytes, sizeof(char), length, file);
        token_bytes[length] ='\0';  //adding null terminator
        tokenizer->token_table[i] = token_bytes;
    }
    //cleanup
    fcloseCheck(file);
    tokenizer->init_ok = 1;

}

const char* tokenizer_decode(Tokenizer* tokenizer, uint32_t token_id){
    if (tokenizer->init_ok == 0){
        return NULL;
    }
    if (token_id < tokenizer->vocab_size){
        return tokenizer->token_table[token_id];
    } else {
        printf("invalid token id %u!\n", token_id);
        return NULL;
    }
}

void tokenizer_free(Tokenizer* tokenizer){
    if (tokenizer->init_ok){
        for(uint32_t i=0; i<tokenizer->vocab_size; i++)
            free(tokenizer->token_table[i]);
        free(tokenizer->token_table);
    }
}