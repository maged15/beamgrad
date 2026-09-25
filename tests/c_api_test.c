/* SPDX-License-Identifier: MIT
 *
 * Compiled as C: checks that both public headers are valid C and that both
 * libraries (including the CUDA stub in CPU-only builds) export their API. */
#include "dbs.h"
#include "dbs_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #cond); \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

int main(void) {
    DBSOptionsC opt;
    DBSDecoderHandle* decoder = NULL;
    DBSResultHandle* result = NULL;
    DBSBackwardHandle* backward = NULL;
    float log_probs[2 * 2 * 3] = {
        -0.1f, -1.0f, -2.0f,  -3.0f, -3.0f, -3.0f,
        -0.5f, -0.2f, -2.0f,  -0.3f, -0.9f, -2.0f,
    };
    float grad_final[2] = {1.0f, 0.0f};
    DBSCudaDecodeArgs cuda_args;
    int cuda_status;

    REQUIRE(dbs_abi_version() == DBS_ABI_VERSION);
    memset(&opt, 0, sizeof(opt));
    opt.beam_size = 2;
    opt.eos_token = -1;
    REQUIRE(dbs_create_ex(opt, &decoder) == 0);
    REQUIRE(dbs_decode(decoder, log_probs, 2, 3, &result) == 0);
    REQUIRE(dbs_result_tokens(result)[0] == 0);
    REQUIRE(dbs_backward(decoder, result, NULL, NULL, grad_final, &backward) == 0);
    REQUIRE(dbs_backward_sparse_logprob_count(backward) > 0);
    dbs_free_backward(backward);
    dbs_free_result(result);
    dbs_destroy(decoder);

    /* The CUDA library links and answers in every build. */
    memset(&cuda_args, 0, sizeof(cuda_args));
    cuda_args.batch_size = 1;
    cuda_args.steps = 2;
    cuda_args.beam_size = 2;
    cuda_args.vocab_size = 3;
    cuda_args.eos_token = -1;
    REQUIRE(dbs_cuda_status_string(DBS_CUDA_STATUS_OK) != NULL);
    REQUIRE(dbs_cuda_set_synchronization(0) == DBS_CUDA_STATUS_OK);
    cuda_status = dbs_cuda_decode(NULL, &cuda_args, NULL, NULL, 0, NULL);
    REQUIRE(cuda_status == DBS_CUDA_STATUS_INVALID_ARGUMENT || cuda_status == DBS_CUDA_STATUS_UNAVAILABLE);
    cuda_status = dbs_cuda_decode_step(NULL, &cuda_args, NULL, NULL, NULL, 0, NULL);
    REQUIRE(cuda_status == DBS_CUDA_STATUS_INVALID_ARGUMENT || cuda_status == DBS_CUDA_STATUS_UNAVAILABLE);
    REQUIRE(dbs_cuda_decode_step_workspace_size(&cuda_args, 0) < 0); /* steps must be 1 (or no CUDA) */
    printf("c_api_test passed (cuda available: %d)\n", dbs_cuda_available());
    return EXIT_SUCCESS;
}
