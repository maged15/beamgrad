/* SPDX-License-Identifier: MIT
 *
 * Decoding and backpropagating through the C ABI. Built and run by ctest. */
#include <stdio.h>
#include <string.h>

#include "dbs.h"

int main(void) {
    /* T = 2 steps, K = 2 beams, V = 3 tokens, laid out as [T, K, V]. */
    const int T = 2, V = 3;
    const float log_probs[2 * 2 * 3] = {
        -0.1f, -1.0f, -2.0f, /* t = 0, beam 0 (the only live beam at t = 0) */
        -9.0f, -9.0f, -9.0f, /* t = 0, beam 1 */
        -0.5f, -0.2f, -2.0f, /* t = 1, beam 0 */
        -0.3f, -0.9f, -2.0f, /* t = 1, beam 1 */
    };

    DBSOptionsC options;
    memset(&options, 0, sizeof(options)); /* zero fields select defaults */
    options.beam_size = 2;
    options.eos_token = -1;

    DBSDecoderHandle* decoder = NULL;
    if (dbs_create_ex(options, &decoder) != 0) {
        fprintf(stderr, "create failed: %s\n", dbs_last_global_error());
        return 1;
    }

    DBSResultHandle* result = NULL;
    if (dbs_decode(decoder, log_probs, T, V, &result) != 0) {
        fprintf(stderr, "decode failed: %s\n", dbs_last_error(decoder));
        dbs_destroy(decoder);
        return 1;
    }
    const float* scores = dbs_result_final_scores(result); /* [K] */
    const int32_t* tokens = dbs_result_tokens(result);     /* [T, K] */
    printf("best beam: tokens %d %d, score %.3f\n", tokens[0], tokens[2], scores[0]);

    /* Gradient of the best final score with respect to log_probs (sparse). */
    const float grad_final[2] = {1.0f, 0.0f};
    DBSBackwardHandle* backward = NULL;
    if (dbs_backward(decoder, result, NULL, NULL, grad_final, &backward) != 0) {
        fprintf(stderr, "backward failed: %s\n", dbs_last_error(decoder));
        dbs_free_result(result);
        dbs_destroy(decoder);
        return 1;
    }
    const int64_t n = dbs_backward_sparse_logprob_count(backward);
    const int64_t* index = dbs_backward_sparse_logprob_indices(backward); /* into [T, K, V] */
    const float* value = dbs_backward_sparse_logprob_values(backward);
    for (int64_t i = 0; i < n; ++i) printf("d score / d log_probs[%lld] = %.3f\n", (long long)index[i], value[i]);

    dbs_free_backward(backward);
    dbs_free_result(result);
    dbs_destroy(decoder);
    return 0;
}
