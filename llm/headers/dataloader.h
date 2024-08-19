/*
Implements:
- DataLoader for model training
- EvalLoader for multiple-choice evaluation. eg HellaSwag
*/
#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>        //NULL, size_t
#include <stdint.h>
#include <assert.h>
#include <string.h>

#include "utils.h"
#include "rand.h"

//windows specific
#ifndef _WIN32
#include <glob.h>
#endif

//Distributed data loader

#define HEADER_SIZE 256

typdef struct {
    int process_rank;
    int num_processes;
    //batch and token info
    size_t B;
    size_t T;
    size_t num_tokens;  //total number of tokens
    size_t shard_num_samples; //total number of shards
    glob_t glob_result;   //stores the result of glob operation for shards
    size_t current_shard_idx;
    size_t current_sample_idx;

    //File handling
    FILE* tokens_file;
    //data buffers
    uint16_t buffer;    //fread data from file to buffer
    int* inputs;    //input tokens to transformer
    int* targets;   //target tokens for transformer
    //random shuffle
    mt19937_state shuffle_rng;
    int should_shuffle;
    int* shard_indices;
    int* intra_shard_indices;

    //size in bytes
    size_t total_batch_size_bytes;      //total accross all processes
    size_t local_batch_offset_bytes;    //inner-sample offset
    size_t header_bytes;    //header size in bytes
    int64_t file_size_bytes;

} Dataloader;

int64_t dataloader_load_shard_(DataLoader *loader, int shard_index) {
    if (loader->should_shuffle) {
        shard_index = loader->shard_indices[shard_index];
    }
    // use the first glob match as the filename for now
    const char* filename = loader->glob_result.gl_pathv[shard_index];
    // open the input file for reading. also only a single file can be opened at a time
    if (loader->tokens_file != NULL) {
        fcloseCheck(loader->tokens_file);
    }
    loader->tokens_file = fopenCheck(filename, "rb");
    // validate the header
    int header[HEADER_SIZE];
    freadCheck(header, sizeof(int), HEADER_SIZE, loader->tokens_file);
    if (header[0] != 20240520) {
        printf("Bad magic in the data file\n");
        printf("---> HINT: Are you passing in a correct file?\n");
        printf("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        exit(EXIT_FAILURE);
    }
    if (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    int64_t ntok = header[2]; // number of tokens in the file
    assert(ntok > 0); // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(loader->tokens_file, 0, SEEK_END); // seek to end of file
    loader->file_size_bytes = ftell(loader->tokens_file); // read the offset, i.e. file size
    fseekCheck(loader->tokens_file, 0, SEEK_SET); // seek back to the beginning
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    int64_t expected_file_size = HEADER_SIZE * sizeof(int) + ntok * sizeof(uint16_t);
    if (loader->file_size_bytes != expected_file_size) {
        printf("Error: file size is not as expected\n");
        exit(EXIT_FAILURE);
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    loader->shard_num_samples = (ntok * sizeof(uint16_t) - sizeof(uint16_t)) / loader->total_batch_size_bytes;
    return ntok;
}

void prepare_intra_shard_indices_(DataLoader *loader) {
    // shuffle the examples inside the shards
    if (loader->intra_shard_indices != NULL) {
        // in case shards have different number of samples / sizes
        free(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (int*)mallocCheck(loader->shard_num_samples * sizeof(int));
    init_identity_permutation(loader->intra_shard_indices, (int) loader->shard_num_samples);
    random_permutation(loader->intra_shard_indices, (int) loader->shard_num_samples, &loader->shuffle_rng);
}

void dataloader_reset(DataLoader *loader) {
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    if (loader->should_shuffle) {  // shuffle the shards
        random_permutation(loader->shard_indices, (int) loader->glob_result.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_load_shard_(loader, (int) loader->current_shard_idx);

    if (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

void dataloader_advance_(DataLoader *loader) {
    if (loader->current_shard_idx == loader->glob_result.gl_pathc - 1) {
        // if we are at the last shard, we reset the loader and start a new epoch
        dataloader_reset(loader);
        return;
    }

    // advance the loader by loading the next data shard and resetting the position
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_result.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_load_shard_(loader, (int) loader->current_shard_idx);

    if (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}








#endif