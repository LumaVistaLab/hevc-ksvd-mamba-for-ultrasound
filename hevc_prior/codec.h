#ifndef USR_HEVC_CODEC_H
#define USR_HEVC_CODEC_H
#include <stdint.h>
typedef struct {
    int quality_index, gradient_threshold, boost_offset, boost_limit;
    int distortion_weight_limit, rate_weight_floor, zero_multiplier, zero_shift;
    int zero_reduction_shift, structure_weight;
    int distortion_weights[5], rate_weights[5];
} HevcParameters;
int usr_hevc_write(const uint8_t *pixels, int height, int width, const char *path,
                   const HevcParameters *parameters);
#endif
