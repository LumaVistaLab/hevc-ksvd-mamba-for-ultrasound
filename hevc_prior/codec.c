/* Structure-sensitive HEVC intra coding.
 * Copyright (c) 2025 https://github.com/WangXuan95
 * License: licenses/third_party.txt
 */
#include "codec.h"
#include "coding_tables.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int size;
    int x;
    int y;
} PartitionRecord;
typedef struct {
    PartitionRecord *blocks;
    int count;
    int capacity;
} PartitionList;
typedef struct {
    PartitionList ctu;
    PartitionList cu;
    PartitionList pu;
    PartitionList tu;
} PartitionCollection;
static PartitionCollection *partitions;
static const HevcParameters *settings;
static int collection_failed;
static void append_partition(PartitionList *, int, int, int);
typedef enum { SCAN_TYPE_DIAG = 0, SCAN_TYPE_HOR = 1, SCAN_TYPE_VER = 2 } ScanType;
typedef enum { CH_Y = 0, CH_U = 1, CH_V = 2 } ChannelType;
uint8_t contains_coefficients(const int32_t block_size, const int32_t source[][32]) {
    int32_t i;
    int32_t j;
    for (i = 0; i < block_size; i++)
        for (j = 0; j < block_size; j++)
            if (source[i][j] != 0)
                return 1;

    return 0;
}

static int32_t gradient_energy(const int32_t block_size, const uint8_t blk[][32]) {
    int32_t i;
    int32_t j;
    int32_t gx;
    int32_t gy;
    int32_t energy = 0;
    for (i = 1; i < (block_size - 1); i++) {
        for (j = 1; j < (block_size - 1); j++) {
            gx = ((int32_t)blk[i][j + 1]) - ((int32_t)blk[i][j - 1]);
            gy = ((int32_t)blk[i + 1][j]) - ((int32_t)blk[i - 1][j]);
            energy += ((gx < 0) ? (-gx) : (gx)) + ((gy < 0) ? (-gy) : (gy));
        }
    }

    return energy / (block_size * block_size);
}

static int32_t partition_cost(int32_t quality_index, int32_t dist, int32_t bits, int32_t gradient) {
    int32_t distortion_weight = settings->distortion_weights[quality_index];
    int32_t rate_weight = settings->rate_weights[quality_index];
    if (gradient > settings->gradient_threshold) {
        int32_t boost = ((gradient - settings->boost_offset) < settings->boost_limit)
                            ? (gradient - settings->boost_offset)
                            : (settings->boost_limit);
        distortion_weight = ((distortion_weight + boost) < settings->distortion_weight_limit)
                                ? (distortion_weight + boost)
                                : (settings->distortion_weight_limit);
        rate_weight = ((rate_weight - boost) < settings->rate_weight_floor) ? (settings->rate_weight_floor)
                                                                            : (rate_weight - boost);
    }
    int32_t distortion_cost = ((((int32_t)0x7fffffff) / distortion_weight) <= dist)
                                  ? ((int32_t)0x7fffffff)
                                  : (distortion_weight * dist);
    int32_t rate_cost =
        ((((int32_t)0x7fffffff) / rate_weight) <= bits) ? ((int32_t)0x7fffffff) : (rate_weight * bits);
    return ((((int32_t)0x7fffffff) - distortion_cost) <= rate_cost) ? ((int32_t)0x7fffffff)
                                                                    : (distortion_cost + rate_cost);
}

static void prediction_border(const int32_t block_size, const uint8_t bll_exist, const uint8_t blb_exist,
                              const uint8_t baa_exist, const uint8_t bar_exist,
                              const uint8_t reconstructed_block[][1 + (32 * 2)], uint8_t ubla[1],
                              uint8_t ublb[32 * 2], uint8_t ubar[32 * 2], uint8_t fbla[1],
                              uint8_t fblb[32 * 2], uint8_t fbar[32 * 2]) {
    int32_t i;
    if (bll_exist && baa_exist)
        ubla[0] = reconstructed_block[-1][-1];
    else if (bll_exist)
        ubla[0] = reconstructed_block[0][-1];
    else if (baa_exist)
        ubla[0] = reconstructed_block[-1][0];
    else
        ubla[0] = (uint8_t)128;
    for (i = 0; i < block_size; i++)
        if (bll_exist)
            ublb[i] = reconstructed_block[i][-1];
        else
            ublb[i] = ubla[0];

    for (i = block_size; i < (block_size * 2); i++)
        if (blb_exist)
            ublb[i] = reconstructed_block[i][-1];
        else
            ublb[i] = ublb[block_size - 1];

    for (i = 0; i < block_size; i++)
        if (baa_exist)
            ubar[i] = reconstructed_block[-1][i];
        else
            ubar[i] = ubla[0];

    for (i = block_size; i < (block_size * 2); i++)
        if (bar_exist)
            ubar[i] = reconstructed_block[-1][i];
        else
            ubar[i] = ubar[block_size - 1];

    fbla[0] = (((2 + ublb[0]) + ubar[0]) + (2 * ubla[0])) >> 2;
    fblb[0] = (((2 + (2 * ublb[0])) + ublb[1]) + ubla[0]) >> 2;
    fbar[0] = (((2 + (2 * ubar[0])) + ubar[1]) + ubla[0]) >> 2;
    for (i = 1; i < ((block_size * 2) - 1); i++) {
        fblb[i] = (((2 + (2 * ublb[i])) + ublb[i - 1]) + ublb[i + 1]) >> 2;
        fbar[i] = (((2 + (2 * ubar[i])) + ubar[i - 1]) + ubar[i + 1]) >> 2;
    }

    fblb[(block_size * 2) - 1] = ublb[(block_size * 2) - 1];
    fbar[(block_size * 2) - 1] = ubar[(block_size * 2) - 1];
}

static void intra_prediction(const int32_t block_size, const ChannelType ch, const int32_t prediction_mode,
                             const uint8_t ubla, const uint8_t ublb[32 * 2], const uint8_t ubar[32 * 2],
                             const uint8_t fbla, const uint8_t fblb[32 * 2], const uint8_t fbar[32 * 2],
                             uint8_t destination[32][32]) {
    const uint8_t whether_filter_edge = (ch == CH_Y) && (block_size <= 16);
    const uint8_t whether_filter_border =
        (ch == CH_Y) && intra_prediction_whether_filter_border_for_y_table[block_size / 8][prediction_mode];
    const uint8_t bla = (whether_filter_border) ? (fbla) : (ubla);
    const uint8_t *blb = (whether_filter_border) ? (fblb) : (ublb);
    const uint8_t *bar = (whether_filter_border) ? (fbar) : (ubar);
    int32_t i;
    int32_t j;
    if (prediction_mode == 0) {
        for (i = 0; i < block_size; i++) {
            for (j = 0; j < block_size; j++) {
                const int32_t hor_pred = (((block_size - j) - 1) * blb[i]) + ((j + 1) * bar[block_size]);
                const int32_t ver_pred = (((block_size - i) - 1) * bar[j]) + ((i + 1) * blb[block_size]);
                destination[i][j] = (uint8_t)(((block_size + hor_pred) + ver_pred) / (block_size * 2));
            }
        }

    } else if (prediction_mode == 1) {
        int32_t dc_pix = block_size;
        for (i = 0; i < block_size; i++)
            dc_pix += blb[i] + bar[i];

        dc_pix /= block_size * 2;
        for (i = 0; i < block_size; i++)
            for (j = 0; j < block_size; j++)
                destination[i][j] = (uint8_t)dc_pix;

        if (whether_filter_edge) {
            destination[0][0] = (uint8_t)((((2 + (2 * dc_pix)) + blb[0]) + bar[0]) >> 2);
            for (i = 1; i < block_size; i++) {
                destination[0][i] = (uint8_t)(((2 + (3 * dc_pix)) + bar[i]) >> 2);
                destination[i][0] = (uint8_t)(((2 + (3 * dc_pix)) + blb[i]) >> 2);
            }
        }
    } else if (prediction_mode == 10) {
        for (i = 0; i < block_size; i++)
            for (j = 0; j < block_size; j++)
                destination[i][j] = blb[i];

        if (whether_filter_edge)
            for (j = 0; j < block_size; j++) {
                const int32_t bias = (bar[j] - bla) >> 1;
                destination[0][j] =
                    (uint8_t)(((((bias + destination[0][j]) < ((uint8_t)0))
                                    ? ((uint8_t)0)
                                    : (bias + destination[0][j])) < ((uint8_t)255))
                                  ? (((bias + destination[0][j]) < ((uint8_t)0)) ? ((uint8_t)0)
                                                                                 : (bias + destination[0][j]))
                                  : ((uint8_t)255));
            }

    } else if (prediction_mode == 26) {
        for (i = 0; i < block_size; i++)
            for (j = 0; j < block_size; j++)
                destination[i][j] = bar[j];

        if (whether_filter_edge)
            for (i = 0; i < block_size; i++) {
                const int32_t bias = (blb[i] - bla) >> 1;
                destination[i][0] =
                    (uint8_t)(((((bias + destination[i][0]) < ((uint8_t)0))
                                    ? ((uint8_t)0)
                                    : (bias + destination[i][0])) < ((uint8_t)255))
                                  ? (((bias + destination[i][0]) < ((uint8_t)0)) ? ((uint8_t)0)
                                                                                 : (bias + destination[i][0]))
                                  : ((uint8_t)255));
            }

    } else {
        const uint8_t is_horizontal = prediction_mode < 18;
        const int32_t angle = intra_prediction_angle_table[prediction_mode];
        const int32_t abs_inv_angle = intra_prediction_abs_inv_angle_table[prediction_mode];
        const uint8_t *bmain = (is_horizontal) ? (blb) : (bar);
        const uint8_t *bside = (is_horizontal) ? (bar) : (blb);
        uint8_t ref_buff0[(32 * 4) + 1];
        uint8_t *ref_buff = ref_buff0 + (32 * 2);
        ref_buff[0] = bla;
        for (i = 0; i < (block_size * 2); i++)
            ref_buff[1 + i] = bside[i];

        for (i = -1; i > ((block_size * angle) >> 5); i--) {
            j = (128 - (abs_inv_angle * i)) >> 8;
            ref_buff[i] = ref_buff[j];
        }

        for (i = 0; i < (block_size * 2); i++)
            ref_buff[1 + i] = bmain[i];

        for (i = 0; i < block_size; i++) {
            const int32_t offset = angle * (i + 1);
            const int32_t offset_i = offset >> 5;
            const int32_t offset_f = offset & 0x1f;
            for (j = 0; j < block_size; j++) {
                const uint8_t pix1 = ref_buff[(offset_i + j) + 1];
                const uint8_t pix2 = ref_buff[(offset_i + j) + 2];
                const uint8_t pix = (uint8_t)(((((32 - offset_f) * pix1) + (offset_f * pix2)) + 16) >> 5);
                if (is_horizontal)
                    destination[j][i] = pix;
                else
                    destination[i][j] = pix;
            }
        }
    }
}

const int32_t DST4_MAT[][32] = {{29, 55, 74, 84}, {74, 74, 0, -74}, {84, -29, -74, 55}, {55, -84, 74, -29}};
const int32_t DCT8_MAT[][32] = {{64, 64, 64, 64, 64, 64, 64, 64},     {89, 75, 50, 18, -18, -50, -75, -89},
                                {83, 36, -36, -83, -83, -36, 36, 83}, {75, -18, -89, -50, 50, 89, 18, -75},
                                {64, -64, -64, 64, 64, -64, -64, 64}, {50, -89, 18, 75, -75, -18, 89, -50},
                                {36, -83, 83, -36, -36, 83, -83, 36}, {18, -50, 75, -89, 89, -75, 50, -18}};
const int32_t DCT16_MAT[][32] = {{64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64},
                                 {90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90},
                                 {89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89},
                                 {87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87},
                                 {83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83},
                                 {80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80},
                                 {75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75},
                                 {70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70},
                                 {64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64},
                                 {57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57},
                                 {50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50},
                                 {43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43},
                                 {36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36},
                                 {25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25},
                                 {18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18},
                                 {9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9}};
const int32_t DCT32_MAT[][32] = {
    {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
     64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64},
    {90, 90,  88,  85,  82,  78,  73,  67,  61,  54,  46,  38,  31,  22,  13,  4,
     -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90},
    {90,  87,  80,  70,  57,  43,  25,  9,  -9, -25, -43, -57, -70, -80, -87, -90,
     -90, -87, -80, -70, -57, -43, -25, -9, 9,  25,  43,  57,  70,  80,  87,  90},
    {90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13,
     13, 38, 61, 78, 88, 90, 85,  73,  54,  31,  4,   -22, -46, -67, -82, -90},
    {89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
     89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89},
    {88,  67,  31,  -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85,  61,  22,
     -22, -61, -85, -90, -73, -38, 4,   46,  78,  90, 82, 54, 13, -31, -67, -88},
    {87,  57,  9,  -43, -80, -90, -70, -25, 25,  70,  90,  80,  43,  -9, -57, -87,
     -87, -57, -9, 43,  80,  90,  70,  25,  -25, -70, -90, -80, -43, 9,  57,  87},
    {85, 46, -13, -67, -90, -73, -22, 38,  82,  88, 54, -4, -61, -90, -78, -31,
     31, 78, 90,  61,  4,   -54, -88, -82, -38, 22, 73, 90, 67,  13,  -46, -85},
    {83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83,
     83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83},
    {82,  22,  -54, -90, -61, 13, 78, 85,  31,  -46, -90, -67, 4,  73, 88,  38,
     -38, -88, -73, -4,  67,  90, 46, -31, -85, -78, -13, 61,  90, 54, -22, -82},
    {80,  9,  -70, -87, -25, 57,  90,  43,  -43, -90, -57, 25,  87,  70,  -9, -80,
     -80, -9, 70,  87,  25,  -57, -90, -43, 43,  90,  57,  -25, -87, -70, 9,  80},
    {78, -4, -82, -73, 13,  85,  67, -22, -88, -61, 31,  90,  54, -38, -90, -46,
     46, 90, 38,  -54, -90, -31, 61, 88,  22,  -67, -85, -13, 73, 82,  4,   -78},
    {75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
     75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75},
    {73,  -31, -90, -22, 78, 67,  -38, -90, -13, 82, 61,  -46, -88, -4, 85, 54,
     -54, -85, 4,   88,  46, -61, -82, 13,  90,  38, -67, -78, 22,  90, 31, -73},
    {70,  -43, -87, 9,  90,  25,  -80, -57, 57,  80,  -25, -90, -9, 87,  43,  -70,
     -70, 43,  87,  -9, -90, -25, 80,  57,  -57, -80, 25,  90,  9,  -87, -43, 70},
    {67, -54, -78, 38,  85, -22, -90, 4,   90, 13, -88, -31, 82,  46, -73, -61,
     61, 73,  -46, -82, 31, 88,  -13, -90, -4, 90, 22,  -85, -38, 78, 54,  -67},
    {64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
     64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64},
    {61,  -73, -46, 82, 31,  -88, -13, 90, -4,  -90, 22, 85,  -38, -78, 54, 67,
     -67, -54, 78,  38, -85, -22, 90,  4,  -90, 13,  88, -31, -82, 46,  73, -61},
    {57,  -80, -25, 90,  -9, -87, 43,  70,  -70, -43, 87,  9,  -90, 25,  80,  -57,
     -57, 80,  25,  -90, 9,  87,  -43, -70, 70,  43,  -87, -9, 90,  -25, -80, 57},
    {54, -85, -4,  88, -46, -61, 82,  13, -90, 38,  67, -78, -22, 90, -31, -73,
     73, 31,  -90, 22, 78,  -67, -38, 90, -13, -82, 61, 46,  -88, 4,  85,  -54},
    {50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
     50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50},
    {46,  -90, 38, 54,  -90, 31, 61,  -88, 22, 67,  -85, 13, 73,  -82, 4,  78,
     -78, -4,  82, -73, -13, 85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46},
    {43,  -90, 57,  25,  -87, 70,  9,  -80, 80,  -9, -70, 87,  -25, -57, 90,  -43,
     -43, 90,  -57, -25, 87,  -70, -9, 80,  -80, 9,  70,  -87, 25,  57,  -90, 43},
    {38, -88, 73,  -4, -67, 90,  -46, -31, 85, -78, 13,  61, -90, 54,  22, -82,
     82, -22, -54, 90, -61, -13, 78,  -85, 31, 46,  -90, 67, 4,   -73, 88, -38},
    {36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36,
     36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36},
    {31,  -78, 90, -61, 4,  54,  -88, 82, -38, -22, 73,  -90, 67, -13, -46, 85,
     -85, 46,  13, -67, 90, -73, 22,  38, -82, 88,  -54, -4,  61, -90, 78,  -31},
    {25,  -70, 90,  -80, 43,  9,  -57, 87,  -87, 57,  -9, -43, 80,  -90, 70,  -25,
     -25, 70,  -90, 80,  -43, -9, 57,  -87, 87,  -57, 9,  43,  -80, 90,  -70, 25},
    {22, -61, 85, -90, 73,  -38, -4,  46, -78, 90, -82, 54,  -13, -31, 67, -88,
     88, -67, 31, 13,  -54, 82,  -90, 78, -46, 4,  38,  -73, 90,  -85, 61, -22},
    {18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
     18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18},
    {13,  -38, 61,  -78, 88,  -90, 85, -73, 54, -31, 4,  22,  -46, 67,  -82, 90,
     -90, 82,  -67, 46,  -22, -4,  31, -54, 73, -85, 90, -88, 78,  -61, 38,  -13},
    {9,  -25, 43,  -57, 70,  -80, 87,  -90, 90,  -87, 80,  -70, 57,  -43, 25,  -9,
     -9, 25,  -43, 57,  -70, 80,  -87, 90,  -90, 87,  -80, 70,  -57, 43,  -25, 9},
    {4,  -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90,
     90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4}};
static void transform_product(const int32_t block_size, const uint8_t src1_transpose,
                              const uint8_t src2_transpose, const int32_t dst_sft, const uint8_t dst_clip,
                              const int32_t src1[][32], const int32_t src2[][32], int32_t destination[][32]) {
    const int32_t dst_add = (1 << dst_sft) >> 1;
    int32_t i;
    int32_t j;
    int32_t k;
    int32_t s;
    for (i = 0; i < block_size; i++) {
        for (j = 0; j < block_size; j++) {
            s = dst_add;
            for (k = 0; k < block_size; k++)
                s += ((src1_transpose) ? (src1[k][i]) : (src1[i][k])) *
                     ((src2_transpose) ? (src2[j][k]) : (src2[k][j]));

            s >>= dst_sft;
            if (dst_clip)
                s = (int32_t)((((s < ((int32_t)(-32768))) ? ((int32_t)(-32768)) : (s)) < ((int32_t)32767))
                                  ? ((s < ((int32_t)(-32768))) ? ((int32_t)(-32768)) : (s))
                                  : ((int32_t)32767));
            destination[i][j] = s;
        }
    }
}

static void integer_transform(const int32_t block_size, const uint8_t inverse, const int32_t source[][32],
                              int32_t destination[][32]) {
    static const int32_t (*TABLE_TRANSFORM_MAT[5])[32] = {DST4_MAT, DCT8_MAT, DCT16_MAT, 0, DCT32_MAT};
    int32_t tmp[32][32];
    const int32_t (*mat)[32] = TABLE_TRANSFORM_MAT[block_size / 8];
    const int32_t a = (inverse) ? (7) : (integer_transform_table_a_for_transform[block_size / 8]);
    const int32_t b = (inverse) ? (12) : (a + 7);
    transform_product(block_size, inverse, 0, a, inverse, mat, source, tmp);
    transform_product(block_size, 0, !inverse, b, inverse, tmp, mat, destination);
}

static int32_t coefficient_rate(int32_t level) {
    int32_t i;
    if (level < 6)
        return coefficient_rate_level_rate_table[level];
    level -= 6;
    for (i = 0; (1 << i) <= level; i++)
        level -= 1 << i;

    return 92000 + (((3 + (i * 2)) + 1) << 15);
}

static void quantize_coefficients(const int32_t quality_index, const int32_t block_size,
                                  const int32_t gradient, const int32_t source[][32],
                                  int32_t destination[][32]) {
    const int32_t dist_sft = quantize_coefficients_dist_shift_table[block_size / 8];
    const int32_t sft = quantize_coefficients_level_shift_table[block_size / 8] + quality_index;
    const int32_t add = (1 << sft) >> 1;
    const int32_t max_dlevel = ((int32_t)0x7fffffff) - add;
    const int32_t cg_dlevel_threshold_base = (settings->zero_multiplier << sft) >> settings->zero_shift;
    const int32_t cg_dlevel_threshold =
        (gradient > settings->gradient_threshold)
            ? (((cg_dlevel_threshold_base >> settings->zero_reduction_shift) < 1)
                   ? (1)
                   : (cg_dlevel_threshold_base >> settings->zero_reduction_shift))
            : (cg_dlevel_threshold_base);
    int32_t yc;
    int32_t xc;
    int32_t y;
    int32_t x;
    for (yc = 0; yc < block_size; yc += 4) {
        for (xc = 0; xc < block_size; xc += 4) {
            int32_t cg_sum_dlevel = 0;
            for (y = yc; y < (yc + 4); y++) {
                for (x = xc; x < (xc + 4); x++) {
                    int32_t absval = (source[y][x] < 0) ? (-source[y][x]) : (source[y][x]);
                    int32_t dlevel = (absval > 0x1ffff) ? (max_dlevel)
                                                        : ((((absval & 0x1ffff) << 14) < max_dlevel)
                                                               ? ((absval & 0x1ffff) << 14)
                                                               : (max_dlevel));
                    int32_t level = (int32_t)((((((dlevel + add) >> sft) < ((int32_t)(-32768)))
                                                    ? ((int32_t)(-32768))
                                                    : ((dlevel + add) >> sft)) < ((int32_t)32767))
                                                  ? ((((dlevel + add) >> sft) < ((int32_t)(-32768)))
                                                         ? ((int32_t)(-32768))
                                                         : ((dlevel + add) >> sft))
                                                  : ((int32_t)32767));
                    int32_t min_level = (0 < (level - 2)) ? (level - 2) : (0);
                    int32_t best_cost = (int32_t)0x7fffffff;
                    for (; level >= min_level; level--) {
                        int32_t dist1 = (((dlevel - (level << sft)) < 0) ? (-(dlevel - (level << sft)))
                                                                         : (dlevel - (level << sft))) >>
                                        dist_sft;
                        int32_t dist = ((dist1 < 46340) ? (dist1 * dist1) : ((int32_t)0x7fffffff)) >> 7;
                        int32_t cost = partition_cost(quality_index, dist, coefficient_rate(level), gradient);
                        if (cost < best_cost) {
                            best_cost = cost;
                            destination[y][x] = level;
                        }
                    }

                    if (source[y][x] < 0)
                        destination[y][x] *= -1;
                    cg_sum_dlevel += (dlevel < cg_dlevel_threshold) ? (dlevel) : (cg_dlevel_threshold);
                }
            }

            if (cg_sum_dlevel < cg_dlevel_threshold)
                for (y = yc; y < (yc + 4); y++)
                    for (x = xc; x < (xc + 4); x++)
                        destination[y][x] = 0;
        }
    }
}

static void inverse_quantize(const int32_t quality_index, const int32_t block_size,
                             const int32_t source[][32], int32_t destination[][32]) {
    const int32_t q_sft = inverse_quantize_q_shift_table[block_size / 8] + quality_index;
    int32_t i;
    int32_t j;
    for (i = 0; i < block_size; i++)
        for (j = 0; j < block_size; j++)
            destination[i][j] =
                (int32_t)(((((source[i][j] << q_sft) < ((int32_t)(-32768)))
                                ? ((int32_t)(-32768))
                                : (source[i][j] << q_sft)) < ((int32_t)32767))
                              ? (((source[i][j] << q_sft) < ((int32_t)(-32768))) ? ((int32_t)(-32768))
                                                                                 : (source[i][j] << q_sft))
                              : ((int32_t)32767));
}

static void write_bits(uint8_t **stream_cursor, int32_t *bit_position, int32_t bits, int32_t len) {
    for (len--; len >= 0; len--) {
        uint8_t bit = (bits >> len) & 1;
        if (bit)
            *(*stream_cursor) |= 1 << (*bit_position);
        else
            *(*stream_cursor) &= ~(1 << (*bit_position));
        if ((*bit_position) > 0) {
            (*bit_position)--;
        } else {
            *bit_position = 7;
            (*stream_cursor)++;
        }
    }
}

static void write_unsigned_exp_golomb(uint8_t **stream_cursor, int32_t *bit_position, int32_t val) {
    int32_t tmp;
    int32_t len = 1;
    val++;
    for (tmp = val + 1; tmp != 1; tmp >>= 1)
        len += 2;

    write_bits(stream_cursor, bit_position, val & ((1 << ((len + 1) >> 1)) - 1),
               (len >> 1) + ((len + 1) >> 1));
}

static void align_bitstream(uint8_t **stream_cursor, int32_t *bit_position) {
    if ((*bit_position) < 7)
        *((*stream_cursor)++) &= 0xfe << (*bit_position);
    *bit_position = 7;
}

static void write_bytes(uint8_t **stream_cursor, const uint8_t *bytes, int32_t len) {
    int32_t i;
    for (i = 0; i < len; i++)
        *((*stream_cursor)++) = bytes[i];
}

static void write_picture_header(uint8_t **stream_cursor, const int32_t quality_index, const int32_t height,
                                 const int32_t width) {
    int32_t bit_position = 7;
    write_bytes(stream_cursor, write_picture_header_vps, sizeof(write_picture_header_vps));
    write_bytes(stream_cursor, write_picture_header_sps, sizeof(write_picture_header_sps));
    write_bits(stream_cursor, &bit_position, 0x0A, 4);
    write_unsigned_exp_golomb(stream_cursor, &bit_position, width);
    write_unsigned_exp_golomb(stream_cursor, &bit_position, height);
    write_bits(stream_cursor, &bit_position, 0x197EE4, 22);
    write_bits(stream_cursor, &bit_position, 0x681ED1, 24);
    align_bitstream(stream_cursor, &bit_position);
    write_bytes(stream_cursor, write_picture_header_pps, sizeof(write_picture_header_pps));
    write_bytes(stream_cursor, write_picture_header_slice_headers[quality_index],
                sizeof(write_picture_header_slice_headers[quality_index]));
}

const uint8_t CONTEXT_NEXT_STATE_MPS[] = {
    2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,
    21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
    40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,
    59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
    78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
    97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
    116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 124, 125, 126, 127};
const uint8_t CONTEXT_NEXT_STATE_LPS[] = {
    1,  0,  0,  1,  2,  3,  4,  5,  4,  5,  8,  9,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,  19, 18, 19,
    22, 23, 22, 23, 24, 25, 26, 27, 26, 27, 30, 31, 30, 31, 32, 33, 32, 33, 36, 37, 36, 37, 38,  39, 38, 39,
    42, 43, 42, 43, 44, 45, 44, 45, 46, 47, 48, 49, 48, 49, 50, 51, 52, 53, 52, 53, 54, 55, 54,  55, 56, 57,
    58, 59, 58, 59, 60, 61, 60, 61, 60, 61, 62, 63, 64, 65, 64, 65, 66, 67, 66, 67, 66, 67, 68,  69, 68, 69,
    70, 71, 70, 71, 70, 71, 72, 73, 72, 73, 72, 73, 74, 75, 74, 75, 74, 75, 76, 77, 76, 77, 126, 127};
const uint8_t CABAC_LPS_TABLE[][4] = {
    {128, 176, 208, 240}, {128, 167, 197, 227}, {128, 158, 187, 216}, {123, 150, 178, 205},
    {116, 142, 169, 195}, {111, 135, 160, 185}, {105, 128, 152, 175}, {100, 122, 144, 166},
    {95, 116, 137, 158},  {90, 110, 130, 150},  {85, 104, 123, 142},  {81, 99, 117, 135},
    {77, 94, 111, 128},   {73, 89, 105, 122},   {69, 85, 100, 116},   {66, 80, 95, 110},
    {62, 76, 90, 104},    {59, 72, 86, 99},     {56, 69, 81, 94},     {53, 65, 77, 89},
    {51, 62, 73, 85},     {48, 59, 69, 80},     {46, 56, 66, 76},     {43, 53, 63, 72},
    {41, 50, 59, 69},     {39, 48, 56, 65},     {37, 45, 54, 62},     {35, 43, 51, 59},
    {33, 41, 48, 56},     {32, 39, 46, 53},     {30, 37, 43, 50},     {29, 35, 41, 48},
    {27, 33, 39, 45},     {26, 31, 37, 43},     {24, 30, 35, 41},     {23, 28, 33, 39},
    {22, 27, 32, 37},     {21, 26, 30, 35},     {20, 24, 29, 33},     {19, 23, 27, 31},
    {18, 22, 26, 30},     {17, 21, 25, 28},     {16, 20, 23, 27},     {15, 19, 22, 25},
    {14, 18, 21, 24},     {14, 17, 20, 23},     {13, 16, 19, 22},     {12, 15, 18, 21},
    {12, 14, 17, 20},     {11, 14, 16, 19},     {11, 13, 15, 18},     {10, 12, 15, 17},
    {10, 12, 14, 16},     {9, 11, 13, 15},      {9, 11, 12, 14},      {8, 10, 12, 14},
    {8, 9, 11, 13},       {7, 9, 11, 12},       {7, 9, 10, 12},       {7, 8, 10, 11},
    {6, 8, 9, 11},        {6, 7, 9, 10},        {6, 7, 8, 9},         {2, 2, 2, 2}};
const uint8_t CABAC_RENORM_TABLE[] = {6, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2,
                                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static uint8_t initial_context(uint8_t init_val, int32_t quality_index) {
    int32_t qp = (quality_index * 6) + 4;
    int32_t init_state = ((((((init_val >> 4) * 5) - 45) * qp) >> 4) + ((init_val & 15) << 3)) - 16;
    init_state =
        (((init_state < 1) ? (1) : (init_state)) < 126) ? ((init_state < 1) ? (1) : (init_state)) : (126);
    if (init_state >= 64)
        return (uint8_t)(((init_state - 64) << 1) | 1);
    else
        return (uint8_t)((63 - init_state) << 1);
}

typedef struct {
    uint8_t split_cu_flag[3];
    uint8_t partsize;
    uint8_t Y_pmode;
    uint8_t UV_pmode;
    uint8_t split_tu_flag[3];
    uint8_t Y_qt_cbf[2];
    uint8_t UV_qt_cbf[5];
    uint8_t last_x[5][5];
    uint8_t last_y[5][5];
    uint8_t sig_map[2];
    uint8_t sig_sc[44];
    uint8_t one_sc[24];
    uint8_t abs_sc[6];
} EntropyContexts;
static EntropyContexts initialize_contexts(int32_t quality_index) {
    EntropyContexts tCtxs = {{139, 141, 157},
                             184,
                             184,
                             63,
                             {153, 138, 138},
                             {111, 141},
                             {94, 138, 182, 154, 154},
                             {{110, 110, 124},
                              {125, 140, 153},
                              {125, 127, 140, 109},
                              {111, 143, 127, 111, 79},
                              {108, 123, 63, 154}},
                             {{110, 110, 124},
                              {125, 140, 153},
                              {125, 127, 140, 109},
                              {111, 143, 127, 111, 79},
                              {108, 123, 63, 154}},
                             {91, 171},
                             {111, 111, 125, 110, 110, 94,  124, 108, 124, 107, 125, 141, 179, 153, 125,
                              107, 125, 141, 179, 153, 125, 107, 125, 141, 179, 153, 125, 141, 140, 139,
                              182, 182, 152, 136, 152, 136, 153, 136, 139, 111, 136, 139, 111, 111},
                             {140, 92,  137, 138, 140, 152, 138, 139, 153, 74,  149, 92,
                              139, 107, 122, 152, 140, 179, 166, 182, 140, 227, 122, 197},
                             {138, 153, 136, 167, 152, 152}};
    uint8_t *ptr = (uint8_t *)(&tCtxs);
    uint8_t *endptr = ptr + (sizeof(tCtxs));
    for (; ptr < endptr; ptr++)
        *ptr = initial_context(*ptr, quality_index);

    return tCtxs;
}

typedef struct {
    uint8_t tmpbuf[((32 * 32) * 3) + 128];
    int32_t tmpcnt;
    int32_t count00;
    int32_t range;
    int32_t low;
    int32_t nbits;
    int32_t nbytes;
    int32_t bufbyte;
} ArithmeticEncoder;
static ArithmeticEncoder initialize_arithmetic(void) {
    ArithmeticEncoder tCABAC = {{0x00}, 0, 0, 510, 0, 23, 0, 0xFF};
    return tCABAC;
}

static void submit_arithmetic(ArithmeticEncoder *p, uint8_t **stream_cursor) {
    write_bytes(stream_cursor, p->tmpbuf, p->tmpcnt);
    p->tmpcnt = 0;
}

static void arithmetic_byte(ArithmeticEncoder *p, int32_t byte) {
    if ((p->count00 >= 2) && (((uint8_t)byte) <= 0x03)) {
        p->tmpbuf[p->tmpcnt++] = 0x03;
        p->count00 = 0;
    }
    p->tmpbuf[p->tmpcnt++] = (uint8_t)byte;
    if (((uint8_t)byte) == 0x00)
        p->count00++;
    else
        p->count00 = 0;
}

static int32_t arithmetic_length(const ArithmeticEncoder *p) {
    return ((8 * (p->tmpcnt + p->nbytes)) + 23) - p->nbits;
}

static void finish_arithmetic(ArithmeticEncoder *p) {
    int32_t tmp = 0x00;
    if ((p->low >> (32 - p->nbits)) > 0) {
        arithmetic_byte(p, p->bufbyte + 1);
        p->low -= 1 << (32 - p->nbits);
    } else {
        if (p->nbytes > 0)
            arithmetic_byte(p, p->bufbyte);
        tmp = 0xff;
    }
    for (; p->nbytes > 1; p->nbytes--)
        arithmetic_byte(p, tmp);

    tmp = (p->low >> 8) << p->nbits;
    arithmetic_byte(p, tmp >> 16);
    arithmetic_byte(p, tmp >> 8);
    arithmetic_byte(p, tmp);
}

static void update_arithmetic(ArithmeticEncoder *p) {
    if (p->nbits < 12) {
        int32_t lead_byte = p->low >> (24 - p->nbits);
        p->nbits += 8;
        p->low &= 0xFFFFFFFF >> p->nbits;
        if (lead_byte == 0xFF) {
            p->nbytes++;
        } else if (p->nbytes > 0) {
            int32_t carry = lead_byte >> 8;
            int32_t byte = carry + p->bufbyte;
            p->bufbyte = lead_byte & 0xFF;
            arithmetic_byte(p, byte);
            byte = (0xFF + carry) & 0xFF;
            for (; p->nbytes > 1; p->nbytes--)
                arithmetic_byte(p, byte);

        } else {
            p->nbytes = 1;
            p->bufbyte = lead_byte;
        }
    }
}

static void write_termination(ArithmeticEncoder *p, uint8_t bin) {
    bin = !(!bin);
    p->range -= 2;
    if (bin) {
        p->low += p->range;
        p->low <<= 7;
        p->range = 2 << 7;
        p->nbits -= 7;
    } else if (p->range < 256) {
        p->low <<= 1;
        p->range <<= 1;
        p->nbits--;
    }
    update_arithmetic(p);
}

static void write_bypass_bins(ArithmeticEncoder *p, int32_t bins, int32_t len) {
    bins &= (1 << len) - 1;
    while (len > 0) {
        const int32_t len_curr = (len < 8) ? (len) : (8);
        int32_t bins_curr;
        len -= len_curr;
        bins_curr = (bins >> len) & ((1 << len_curr) - 1);
        p->low <<= len_curr;
        p->low += p->range * bins_curr;
        p->nbits -= len_curr;
        update_arithmetic(p);
    }
}

static void write_context_bin(ArithmeticEncoder *p, uint8_t bin, uint8_t *pCtx) {
    int32_t lps = CABAC_LPS_TABLE[(*pCtx) >> 1][(p->range >> 6) & 3];
    int32_t nbit = CABAC_RENORM_TABLE[lps >> 3];
    bin = !(!bin);
    p->range -= lps;
    if (bin != ((*pCtx) & 1)) {
        {
            *pCtx = CONTEXT_NEXT_STATE_LPS[*pCtx];
        };
        p->low = (p->low + p->range) << nbit;
        p->range = lps << nbit;
        p->nbits -= nbit;
    } else {
        {
            *pCtx = CONTEXT_NEXT_STATE_MPS[*pCtx];
        };
        if (p->range < 256) {
            p->low <<= 1;
            p->range <<= 1;
            p->nbits--;
        }
    }
    update_arithmetic(p);
}

static void write_cu_split(ArithmeticEncoder *arithmetic, EntropyContexts *contexts, const int32_t block_size,
                           const uint8_t split_cu_flag, const uint8_t larger_than_left_cu,
                           const uint8_t larger_than_above_cu) {
    const uint8_t ctx_idx = (!(!larger_than_left_cu)) + (!(!larger_than_above_cu));
    if (block_size >= 16)
        write_context_bin(arithmetic, split_cu_flag, &contexts->split_cu_flag[ctx_idx]);
}

static void write_prediction_partition(ArithmeticEncoder *arithmetic, EntropyContexts *contexts,
                                       const int32_t block_size, const uint8_t partNxN) {
    if (block_size == 8)
        write_context_bin(arithmetic, !partNxN, &contexts->partsize);
}

static void probable_modes(const int32_t pmode_left, const int32_t pmode_above, int32_t pmodes[3]) {
    if (pmode_left != pmode_above) {
        pmodes[0] = pmode_left;
        pmodes[1] = pmode_above;
        if ((pmode_left != 0) && (pmode_above != 0))
            pmodes[2] = 0;
        else if ((pmode_left + pmode_above) < 2)
            pmodes[2] = 26;
        else
            pmodes[2] = 1;
    } else if (pmode_left > 1) {
        pmodes[0] = pmode_left;
        pmodes[1] = ((pmode_left + 29) % 32) + 2;
        pmodes[2] = ((pmode_left - 1) % 32) + 2;
    } else {
        pmodes[0] = 0;
        pmodes[1] = 1;
        pmodes[2] = 26;
    }
}

static void write_luma_mode(ArithmeticEncoder *arithmetic, EntropyContexts *contexts, const uint8_t partNxN,
                            const int32_t prediction_mode[], const int32_t pmode_left[],
                            const int32_t pmode_above[]) {
    const int32_t part_count = (partNxN) ? (4) : (1);
    int32_t probable_pmodes[4][3];
    int32_t hit_index[4] = {-1, -1, -1, -1};
    int32_t i;
    int32_t j;
    int32_t tmp;
    for (i = 0; i < part_count; i++) {
        probable_modes(pmode_left[i], pmode_above[i], probable_pmodes[i]);
        for (j = 0; j < 3; j++)
            if (probable_pmodes[i][j] == prediction_mode[i])
                hit_index[i] = j;

        write_context_bin(arithmetic, hit_index[i] >= 0, &contexts->Y_pmode);
    }

    for (i = 0; i < part_count; i++) {
        j = hit_index[i];
        if (j >= 0) {
            write_bypass_bins(arithmetic, j > 0, 1);
            if (j > 0)
                write_bypass_bins(arithmetic, j - 1, 1);
        } else {
            if (probable_pmodes[i][0] < probable_pmodes[i][1]) {
                tmp = probable_pmodes[i][0];
                probable_pmodes[i][0] = probable_pmodes[i][1];
                probable_pmodes[i][1] = tmp;
            }
            if (probable_pmodes[i][1] < probable_pmodes[i][2]) {
                tmp = probable_pmodes[i][1];
                probable_pmodes[i][1] = probable_pmodes[i][2];
                probable_pmodes[i][2] = tmp;
            }
            if (probable_pmodes[i][0] < probable_pmodes[i][1]) {
                tmp = probable_pmodes[i][0];
                probable_pmodes[i][0] = probable_pmodes[i][1];
                probable_pmodes[i][1] = tmp;
            }
            tmp = prediction_mode[i];
            for (j = 0; j < 3; j++)
                if (tmp > probable_pmodes[i][j])
                    tmp--;

            write_bypass_bins(arithmetic, tmp, 5);
        }
    }
}

static void write_chroma_mode(ArithmeticEncoder *arithmetic, EntropyContexts *contexts) {
    write_context_bin(arithmetic, 0, &contexts->UV_pmode);
}

static void write_tu_split(ArithmeticEncoder *arithmetic, EntropyContexts *contexts, const int32_t block_size,
                           const uint8_t split_tu_flag) {
    if (block_size == 32)
        write_context_bin(arithmetic, split_tu_flag, &contexts->split_tu_flag[0]);
    else if (block_size == 16)
        write_context_bin(arithmetic, split_tu_flag, &contexts->split_tu_flag[1]);
    else if (block_size == 8)
        write_context_bin(arithmetic, split_tu_flag, &contexts->split_tu_flag[2]);
}

static void write_coded_block(ArithmeticEncoder *arithmetic, EntropyContexts *contexts,
                              const int32_t tu_depth_in_cu, const ChannelType ch, const uint8_t cbf) {
    if (ch == CH_Y)
        write_context_bin(arithmetic, cbf, &contexts->Y_qt_cbf[!tu_depth_in_cu]);
    else
        write_context_bin(arithmetic, cbf, &contexts->UV_qt_cbf[tu_depth_in_cu]);
}

static void write_last_coefficient(ArithmeticEncoder *arithmetic, EntropyContexts *contexts,
                                   const int32_t block_size, const ChannelType ch, const ScanType scan_type,
                                   const int32_t y, const int32_t x) {
    const int32_t addr = write_last_coefficient_addr_table[ch != CH_Y][block_size / 8];
    const int32_t sft = write_last_coefficient_sft_table[ch != CH_Y][block_size / 8];
    int32_t ty = (scan_type == SCAN_TYPE_VER) ? (x) : (y);
    int32_t tx = (scan_type == SCAN_TYPE_VER) ? (y) : (x);
    int32_t gy = write_last_coefficient_group_index_table[ty];
    int32_t gx = write_last_coefficient_group_index_table[tx];
    int32_t i;
    for (i = 0; i < gx; i++)
        write_context_bin(arithmetic, 1, &contexts->last_x[addr][i >> sft]);

    if (gx < write_last_coefficient_group_index_table[block_size - 1])
        write_context_bin(arithmetic, 0, &contexts->last_x[addr][gx >> sft]);
    for (i = 0; i < gy; i++)
        write_context_bin(arithmetic, 1, &contexts->last_y[addr][i >> sft]);

    if (gy < write_last_coefficient_group_index_table[block_size - 1])
        write_context_bin(arithmetic, 0, &contexts->last_y[addr][gy >> sft]);
    if (gx > 3) {
        tx -= write_last_coefficient_min_in_group_table[gx];
        for (i = ((gx - 2) >> 1) - 1; i >= 0; i--)
            write_bypass_bins(arithmetic, (tx >> i) & 1, 1);
    }
    if (gy > 3) {
        ty -= write_last_coefficient_min_in_group_table[gy];
        for (i = ((gy - 2) >> 1) - 1; i >= 0; i--)
            write_bypass_bins(arithmetic, (ty >> i) & 1, 1);
    }
}

static int32_t significance_context(const int32_t block_size, const ChannelType ch, const ScanType scan_type,
                                    const int32_t y, const int32_t x, const int32_t sig_ctx) {
    int32_t ctx_idx = (ch == CH_Y) ? (0) : (28);
    if ((y == 0) && (x == 0))
        return ctx_idx;
    if (block_size == 4)
        return ctx_idx + significance_context_ctx_offset_4x4_table[y][x];
    ctx_idx += 9;
    if (ch == CH_Y) {
        if (block_size >= 16)
            ctx_idx += 12;
        if ((block_size == 8) && (scan_type != SCAN_TYPE_DIAG))
            ctx_idx += 6;
        if (!(((y / 4) == 0) && ((x / 4) == 0)))
            ctx_idx += 3;
    } else if (block_size >= 16)
        ctx_idx += 3;
    switch (sig_ctx) {
    case 0:
        return ctx_idx + significance_context_ctx_offset_position[(y % 4) + (x % 4)];

    case 1:
        return ctx_idx + significance_context_ctx_offset_position[(y % 4) << 1];

    case 2:
        return ctx_idx + significance_context_ctx_offset_position[(x % 4) << 1];

    default:
        return ctx_idx + 2;
    }
}

static ScanType coefficient_scan(const int32_t block_size, const int32_t prediction_mode,
                                 const uint8_t (**scan)[2]) {
    if (block_size <= 8) {
        if ((((prediction_mode - 26) < 0) ? (-(prediction_mode - 26)) : (prediction_mode - 26)) <= 4) {
            *scan = coefficient_scan_scan_hor_8x8;
            return SCAN_TYPE_HOR;
        } else if ((((prediction_mode - 10) < 0) ? (-(prediction_mode - 10)) : (prediction_mode - 10)) <= 4) {
            *scan = coefficient_scan_scan_ver_8x8;
            return SCAN_TYPE_VER;
        }
    }
    switch (block_size) {
    case 4:
        *scan = coefficient_scan_scan_diag_8x8;
        break;

    case 8:
        *scan = coefficient_scan_scan_diag_8x8;
        break;

    case 16:
        *scan = coefficient_scan_scan_diag_16x16;
        break;

    case 32:
        *scan = coefficient_scan_scan_diag_32x32;
        break;
    }

    return SCAN_TYPE_DIAG;
}

static void write_coefficient_tail(ArithmeticEncoder *arithmetic, int32_t value, int32_t rparam) {
    int32_t len;
    int32_t tmp;
    if (value < (3 << rparam)) {
        len = value >> rparam;
        write_bypass_bins(arithmetic, (1 << (len + 1)) - 2, len + 1);
        write_bypass_bins(arithmetic, value % (1 << rparam), rparam);
    } else {
        len = rparam;
        value -= 3 << rparam;
        for (; value >= (1 << len); len++)
            value -= 1 << len;

        tmp = (4 + len) - rparam;
        write_bypass_bins(arithmetic, (1 << tmp) - 2, tmp);
        write_bypass_bins(arithmetic, value, len);
    }
}

static void write_coefficients(ArithmeticEncoder *arithmetic, EntropyContexts *contexts,
                               const int32_t block_size, const ChannelType ch, const int32_t prediction_mode,
                               const int32_t blk[][32]) {
    const uint8_t (*scan)[2] = 0;
    const ScanType scan_type = coefficient_scan(block_size, prediction_mode, &scan);
    int32_t i;
    int32_t i_last = 0;
    int32_t j_nz = 0;
    int32_t signs = 0;
    int32_t sig_ctx = 0;
    int32_t c1 = 1;
    int32_t arr_abs_nz[4 * 4];
    uint8_t sig_map[32 / 4][32 / 4];
    {
        int32_t i;
        int32_t j;
        for (i = 0; i < (block_size / 4); i++)
            for (j = 0; j < (block_size / 4); j++)
                sig_map[i][j] = 0;
    };
    for (i = 0; i < (block_size * block_size); i++) {
        const int32_t y = scan[i][0];
        const int32_t x = scan[i][1];
        if (blk[y][x] != 0) {
            sig_map[y / 4][x / 4] = 1;
            i_last = i;
        }
    }

    write_last_coefficient(arithmetic, contexts, block_size, ch, scan_type, scan[i_last][0], scan[i_last][1]);
    for (i = i_last; i >= 0; i--) {
        const int32_t y = scan[i][0];
        const int32_t x = scan[i][1];
        const int32_t y_cg = y / 4;
        const int32_t x_cg = x / 4;
        const uint8_t sig_cg = sig_map[y_cg][x_cg];
        const uint8_t sig = blk[y][x] != 0;
        const uint8_t sign = blk[y][x] < 0;
        const uint8_t is_final = i == i_last;
        const uint8_t is_first_cg = (y_cg == 0) && (x_cg == 0);
        const uint8_t is_first_in_cg = (i % (4 * 4)) == 0;
        const uint8_t is_final_in_cg = ((i % (4 * 4)) == ((4 * 4) - 1)) || is_final;
        if (is_final_in_cg) {
            const uint8_t sig_cg_right = (x_cg < ((block_size / 4) - 1)) && sig_map[y_cg][x_cg + 1];
            const uint8_t sig_cg_below = (y_cg < ((block_size / 4) - 1)) && sig_map[y_cg + 1][x_cg];
            sig_ctx = (sig_cg_below << 1) | sig_cg_right;
            j_nz = 0;
            signs = 0;
            if ((!is_first_cg) && (!is_final))
                write_context_bin(arithmetic, sig_cg, &contexts->sig_map[!(!sig_ctx)]);
        }
        if ((!is_final) && (is_first_cg || (sig_cg && ((!is_first_in_cg) || (j_nz > 0))))) {
            const int32_t ctx_idx = significance_context(block_size, ch, scan_type, y, x, sig_ctx);
            write_context_bin(arithmetic, sig, &contexts->sig_sc[ctx_idx]);
        }
        if (sig) {
            arr_abs_nz[j_nz++] = (blk[y][x] < 0) ? (-blk[y][x]) : (blk[y][x]);
            signs = (signs << 1) | sign;
        }
        if (is_first_in_cg && (j_nz > 0)) {
            const int32_t ctx_set =
                (((ch == CH_Y) ? (0) : (4)) + (((ch == CH_Y) && (!is_first_cg)) ? (2) : (0))) +
                ((c1 == 0) ? (1) : (0));
            uint8_t escape_flag = j_nz > 8;
            int32_t j;
            int32_t c2_flag = -1;
            c1 = 1;
            for (j = 0; (j < 8) && (j < j_nz); j++) {
                write_context_bin(arithmetic, arr_abs_nz[j] > 1, &contexts->one_sc[(4 * ctx_set) + c1]);
                if (arr_abs_nz[j] > 1) {
                    c1 = 0;
                    if (c2_flag < 0)
                        c2_flag = arr_abs_nz[j] > 2;
                    else
                        escape_flag = 1;
                } else if ((c1 > 0) && (c1 < 3))
                    c1++;
            }

            if ((c1 == 0) && (c2_flag >= 0)) {
                write_context_bin(arithmetic, (uint8_t)c2_flag, &contexts->abs_sc[ctx_set]);
                escape_flag |= c2_flag;
            }
            write_bypass_bins(arithmetic, signs, j_nz);
            if (escape_flag) {
                int32_t first_coeff2 = 3;
                int32_t gorice_param = 0;
                for (j = 0; j < j_nz; j++) {
                    const int32_t escape_value = arr_abs_nz[j] - ((j < 8) ? (first_coeff2) : (1));
                    if (escape_value >= 0) {
                        write_coefficient_tail(arithmetic, escape_value, gorice_param);
                        if (arr_abs_nz[j] > (3 << gorice_param))
                            gorice_param = ((gorice_param + 1) < 4) ? (gorice_param + 1) : (4);
                    }
                    if (arr_abs_nz[j] >= 2)
                        first_coeff2 = 2;
                }
            }
        }
    }
}

static void write_unsplit_transform_unit(ArithmeticEncoder *arithmetic, EntropyContexts *contexts,
                                         const int32_t block_size, const int32_t prediction_mode,
                                         const int32_t pmode_left, const int32_t pmode_above,
                                         const int32_t blk[][32]) {
    uint8_t Ycbf = contains_coefficients(block_size, blk);
    write_prediction_partition(arithmetic, contexts, block_size, 0);
    write_luma_mode(arithmetic, contexts, 0, &prediction_mode, &pmode_left, &pmode_above);
    write_chroma_mode(arithmetic, contexts);
    write_tu_split(arithmetic, contexts, block_size, 0);
    write_coded_block(arithmetic, contexts, 0, CH_U, 0);
    write_coded_block(arithmetic, contexts, 0, CH_V, 0);
    write_coded_block(arithmetic, contexts, 0, CH_Y, Ycbf);
    if (Ycbf)
        write_coefficients(arithmetic, contexts, block_size, CH_Y, prediction_mode, blk);
}

static void write_split_transform_units(ArithmeticEncoder *arithmetic, EntropyContexts *contexts,
                                        const int32_t block_size, const int32_t prediction_mode,
                                        const int32_t pmode_left, const int32_t pmode_above,
                                        int32_t sub_blk[4][32 / 2][32]) {
    int32_t isub;
    write_prediction_partition(arithmetic, contexts, block_size, 0);
    write_luma_mode(arithmetic, contexts, 0, &prediction_mode, &pmode_left, &pmode_above);
    write_chroma_mode(arithmetic, contexts);
    write_tu_split(arithmetic, contexts, block_size, 1);
    write_coded_block(arithmetic, contexts, 0, CH_U, 0);
    write_coded_block(arithmetic, contexts, 0, CH_V, 0);
    for (isub = 0; isub < 4; isub++) {
        uint8_t Ycbf = contains_coefficients(block_size / 2, sub_blk[isub]);
        write_coded_block(arithmetic, contexts, 1, CH_Y, Ycbf);
        if (Ycbf)
            write_coefficients(arithmetic, contexts, block_size / 2, CH_Y, prediction_mode, sub_blk[isub]);
    }
}

static void write_split_prediction_units(ArithmeticEncoder *arithmetic, EntropyContexts *contexts,
                                         const int32_t block_size, const int32_t pmodes[4],
                                         const int32_t pmodes_left[4], const int32_t pmodes_above[4],
                                         int32_t sub_blk[4][32 / 2][32]) {
    int32_t isub;
    write_prediction_partition(arithmetic, contexts, block_size, 1);
    write_luma_mode(arithmetic, contexts, 1, pmodes, pmodes_left, pmodes_above);
    write_chroma_mode(arithmetic, contexts);
    write_coded_block(arithmetic, contexts, 0, CH_U, 0);
    write_coded_block(arithmetic, contexts, 0, CH_V, 0);
    for (isub = 0; isub < 4; isub++) {
        uint8_t Ycbf = contains_coefficients(block_size / 2, sub_blk[isub]);
        write_coded_block(arithmetic, contexts, 1, CH_Y, Ycbf);
        if (Ycbf)
            write_coefficients(arithmetic, contexts, block_size / 2, CH_Y, pmodes[isub], sub_blk[isub]);
    }
}

static void search_coding_unit(const int32_t quality_index, ArithmeticEncoder *arithmetic,
                               EntropyContexts *contexts, uint8_t original_block[][32],
                               uint8_t reconstructed_block[][1 + (32 * 2)],
                               uint8_t map_cu_sz[][1 + (8192 / 4)], uint8_t map_pmode[][1 + (8192 / 4)],
                               const int32_t block_size, const uint8_t bll_exist, const uint8_t blb_exist,
                               const uint8_t baa_exist, const uint8_t bar_exist, const int32_t abs_y,
                               const int32_t abs_x) {
    const ArithmeticEncoder oCABAC = *arithmetic;
    const EntropyContexts oCtxs = *contexts;
    const int32_t nTU = block_size / 4;
    const uint8_t larger_than_left_cu = block_size > map_cu_sz[0][-1];
    const uint8_t larger_than_above_cu = block_size > map_cu_sz[-1][0];
    const int32_t pmode_left = map_pmode[0][-1];
    const int32_t pmode_above = map_pmode[-1][0];
    const uint8_t sub_bll_exist[4] = {bll_exist, 1, bll_exist, 1};
    const uint8_t sub_blb_exist[4] = {bll_exist, 0, blb_exist, 0};
    const uint8_t sub_baa_exist[4] = {baa_exist, baa_exist, 1, 1};
    const uint8_t sub_bar_exist[4] = {baa_exist, bar_exist, 1, 0};
    uint8_t (*sub_blk_orig[4])[32] = {(uint8_t (*)[32])(&original_block[0][0]),
                                      (uint8_t (*)[32])(&original_block[0][block_size / 2]),
                                      (uint8_t (*)[32])(&original_block[block_size / 2][0]),
                                      (uint8_t (*)[32])(&original_block[block_size / 2][block_size / 2])};
    uint8_t (*sub_blk_rcon[4])[1 + (32 * 2)] = {
        (uint8_t (*)[1 + (32 * 2)])(&reconstructed_block[0][0]),
        (uint8_t (*)[1 + (32 * 2)])(&reconstructed_block[0][block_size / 2]),
        (uint8_t (*)[1 + (32 * 2)])(&reconstructed_block[block_size / 2][0]),
        (uint8_t (*)[1 + (32 * 2)])(&reconstructed_block[block_size / 2][block_size / 2])};
    uint8_t (*sub_map_cu_sz[4])[1 + (8192 / 4)] = {
        (uint8_t (*)[1 + (8192 / 4)])(&map_cu_sz[0][0]),
        (uint8_t (*)[1 + (8192 / 4)])(&map_cu_sz[0][nTU / 2]),
        (uint8_t (*)[1 + (8192 / 4)])(&map_cu_sz[nTU / 2][0]),
        (uint8_t (*)[1 + (8192 / 4)])(&map_cu_sz[nTU / 2][nTU / 2])};
    uint8_t (*sub_map_pmode[4])[1 + (8192 / 4)] = {
        (uint8_t (*)[1 + (8192 / 4)])(&map_pmode[0][0]),
        (uint8_t (*)[1 + (8192 / 4)])(&map_pmode[0][nTU / 2]),
        (uint8_t (*)[1 + (8192 / 4)])(&map_pmode[nTU / 2][0]),
        (uint8_t (*)[1 + (8192 / 4)])(&map_pmode[nTU / 2][nTU / 2])};
    uint8_t ubla;
    uint8_t ublb[32 * 2];
    uint8_t ubar[32 * 2];
    uint8_t fbla;
    uint8_t fblb[32 * 2];
    uint8_t fbar[32 * 2];
    uint8_t blk_tmp1[32][32];
    int32_t blk_tmp2[32][32];
    int32_t quantized_block[32][32];
    int32_t sub_blk_quat[4][32 / 2][32];
    uint8_t best_rcon[32][32];
    int32_t isub;
    int32_t prediction_mode;
    int32_t distortion;
    int32_t rate_distortion;
    int32_t rdcost_best = (int32_t)0x7fffffff;
    int32_t best_step = 0;
    int32_t cu_checkpoint = 0;
    int32_t pu_checkpoint = 0;
    int32_t tu_checkpoint = 0;
    int32_t gradient;
    if (partitions) {
        cu_checkpoint = partitions->cu.count;
        pu_checkpoint = partitions->pu.count;
        tu_checkpoint = partitions->tu.count;
    }
    gradient = gradient_energy(block_size, original_block);
    if (block_size > 8) {
        write_cu_split(arithmetic, contexts, block_size, 1, larger_than_left_cu, larger_than_above_cu);
        for (isub = 0; isub < 4; isub++)
            search_coding_unit(quality_index, arithmetic, contexts, sub_blk_orig[isub], sub_blk_rcon[isub],
                               sub_map_cu_sz[isub], sub_map_pmode[isub], block_size / 2, sub_bll_exist[isub],
                               sub_blb_exist[isub], sub_baa_exist[isub], sub_bar_exist[isub],
                               abs_y + ((isub / 2) * (block_size / 2)),
                               abs_x + ((isub % 2) * (block_size / 2)));

        {
            int32_t _sd_i;
            int32_t _sd_j;
            int32_t _sd_diff;
            int32_t _gxo;
            int32_t _gyo;
            int32_t _gxr;
            int32_t _gyr;
            distortion = 0;
            for (_sd_i = 0; _sd_i < block_size; _sd_i++) {
                for (_sd_j = 0; _sd_j < block_size; _sd_j++) {
                    _sd_diff =
                        ((((int32_t)original_block[_sd_i][_sd_j]) - reconstructed_block[_sd_i][_sd_j]) < 0)
                            ? (-(((int32_t)original_block[_sd_i][_sd_j]) - reconstructed_block[_sd_i][_sd_j]))
                            : (((int32_t)original_block[_sd_i][_sd_j]) - reconstructed_block[_sd_i][_sd_j]);
                    distortion += _sd_diff * _sd_diff;
                }
            }

            if (block_size >= 4) {
                for (_sd_i = 1; _sd_i < (block_size - 1); _sd_i++) {
                    for (_sd_j = 1; _sd_j < (block_size - 1); _sd_j++) {
                        _gxo = ((int32_t)original_block[_sd_i][_sd_j + 1]) -
                               ((int32_t)original_block[_sd_i][_sd_j - 1]);
                        _gyo = ((int32_t)original_block[_sd_i + 1][_sd_j]) -
                               ((int32_t)original_block[_sd_i - 1][_sd_j]);
                        _gxr = ((int32_t)reconstructed_block[_sd_i][_sd_j + 1]) -
                               ((int32_t)reconstructed_block[_sd_i][_sd_j - 1]);
                        _gyr = ((int32_t)reconstructed_block[_sd_i + 1][_sd_j]) -
                               ((int32_t)reconstructed_block[_sd_i - 1][_sd_j]);
                        distortion += settings->structure_weight *
                                      ((((_gxo - _gxr) < 0) ? (-(_gxo - _gxr)) : (_gxo - _gxr)) +
                                       (((_gyo - _gyr) < 0) ? (-(_gyo - _gyr)) : (_gyo - _gyr)));
                    }
                }
            }
        };
        rdcost_best = partition_cost(quality_index, distortion,
                                     arithmetic_length(arithmetic) - arithmetic_length(&oCABAC), gradient);
        best_step = 1;
        {
            int32_t i;
            int32_t j;
            for (i = 0; i < block_size; i++)
                for (j = 0; j < block_size; j++)
                    best_rcon[i][j] = reconstructed_block[i][j];
        };
    }
    prediction_border(block_size, bll_exist, blb_exist, baa_exist, bar_exist, reconstructed_block, &ubla,
                      ublb, ubar, &fbla, fblb, fbar);
    for (prediction_mode = 0; prediction_mode <= 1; prediction_mode++) {
        ArithmeticEncoder tCABAC = oCABAC;
        EntropyContexts tCtxs = oCtxs;
        intra_prediction(block_size, CH_Y, prediction_mode, ubla, ublb, ubar, fbla, fblb, fbar, blk_tmp1);
        {
            int32_t i;
            int32_t j;
            for (i = 0; i < block_size; i++)
                for (j = 0; j < block_size; j++)
                    blk_tmp2[i][j] = ((int32_t)original_block[i][j]) - blk_tmp1[i][j];
        };
        integer_transform(block_size, 0, blk_tmp2, blk_tmp2);
        quantize_coefficients(quality_index, block_size, gradient, blk_tmp2, quantized_block);
        inverse_quantize(quality_index, block_size, quantized_block, blk_tmp2);
        integer_transform(block_size, 1, blk_tmp2, blk_tmp2);
        {
            int32_t i;
            int32_t j;
            for (i = 0; i < block_size; i++)
                for (j = 0; j < block_size; j++)
                    blk_tmp1[i][j] = (uint8_t)(((((blk_tmp2[i][j] + blk_tmp1[i][j]) < ((uint8_t)0))
                                                     ? ((uint8_t)0)
                                                     : (blk_tmp2[i][j] + blk_tmp1[i][j])) < ((uint8_t)255))
                                                   ? (((blk_tmp2[i][j] + blk_tmp1[i][j]) < ((uint8_t)0))
                                                          ? ((uint8_t)0)
                                                          : (blk_tmp2[i][j] + blk_tmp1[i][j]))
                                                   : ((uint8_t)255));
        };
        write_cu_split(&tCABAC, &tCtxs, block_size, 0, larger_than_left_cu, larger_than_above_cu);
        write_unsplit_transform_unit(&tCABAC, &tCtxs, block_size, prediction_mode, pmode_left, pmode_above,
                                     quantized_block);
        {
            int32_t _sd_i;
            int32_t _sd_j;
            int32_t _sd_diff;
            int32_t _gxo;
            int32_t _gyo;
            int32_t _gxr;
            int32_t _gyr;
            distortion = 0;
            for (_sd_i = 0; _sd_i < block_size; _sd_i++) {
                for (_sd_j = 0; _sd_j < block_size; _sd_j++) {
                    _sd_diff = ((((int32_t)original_block[_sd_i][_sd_j]) - blk_tmp1[_sd_i][_sd_j]) < 0)
                                   ? (-(((int32_t)original_block[_sd_i][_sd_j]) - blk_tmp1[_sd_i][_sd_j]))
                                   : (((int32_t)original_block[_sd_i][_sd_j]) - blk_tmp1[_sd_i][_sd_j]);
                    distortion += _sd_diff * _sd_diff;
                }
            }

            if (block_size >= 4) {
                for (_sd_i = 1; _sd_i < (block_size - 1); _sd_i++) {
                    for (_sd_j = 1; _sd_j < (block_size - 1); _sd_j++) {
                        _gxo = ((int32_t)original_block[_sd_i][_sd_j + 1]) -
                               ((int32_t)original_block[_sd_i][_sd_j - 1]);
                        _gyo = ((int32_t)original_block[_sd_i + 1][_sd_j]) -
                               ((int32_t)original_block[_sd_i - 1][_sd_j]);
                        _gxr = ((int32_t)blk_tmp1[_sd_i][_sd_j + 1]) - ((int32_t)blk_tmp1[_sd_i][_sd_j - 1]);
                        _gyr = ((int32_t)blk_tmp1[_sd_i + 1][_sd_j]) - ((int32_t)blk_tmp1[_sd_i - 1][_sd_j]);
                        distortion += settings->structure_weight *
                                      ((((_gxo - _gxr) < 0) ? (-(_gxo - _gxr)) : (_gxo - _gxr)) +
                                       (((_gyo - _gyr) < 0) ? (-(_gyo - _gyr)) : (_gyo - _gyr)));
                    }
                }
            }
        };
        rate_distortion = partition_cost(quality_index, distortion,
                                         arithmetic_length(&tCABAC) - arithmetic_length(&oCABAC), gradient);
        if (rdcost_best >= rate_distortion) {
            rdcost_best = rate_distortion;
            best_step = 2;
            *arithmetic = tCABAC;
            *contexts = tCtxs;
            {
                int32_t i;
                int32_t j;
                for (i = 0; i < block_size; i++)
                    for (j = 0; j < block_size; j++)
                        best_rcon[i][j] = blk_tmp1[i][j];
            };
            {
                int32_t i;
                int32_t j;
                for (i = 0; i < nTU; i++)
                    for (j = 0; j < nTU; j++)
                        map_cu_sz[i][j] = (uint8_t)block_size;
            };
            {
                int32_t i;
                int32_t j;
                for (i = 0; i < nTU; i++)
                    for (j = 0; j < nTU; j++)
                        map_pmode[i][j] = (uint8_t)prediction_mode;
            };
        }
    }

    for (prediction_mode = 0; prediction_mode <= 1; prediction_mode++) {
        ArithmeticEncoder tCABAC = oCABAC;
        EntropyContexts tCtxs = oCtxs;
        for (isub = 0; isub < 4; isub++) {
            prediction_border(block_size / 2, sub_bll_exist[isub], sub_blb_exist[isub], sub_baa_exist[isub],
                              sub_bar_exist[isub], sub_blk_rcon[isub], &ubla, ublb, ubar, &fbla, fblb, fbar);
            intra_prediction(block_size / 2, CH_Y, prediction_mode, ubla, ublb, ubar, fbla, fblb, fbar,
                             blk_tmp1);
            {
                int32_t i;
                int32_t j;
                for (i = 0; i < (block_size / 2); i++)
                    for (j = 0; j < (block_size / 2); j++)
                        blk_tmp2[i][j] = ((int32_t)sub_blk_orig[isub][i][j]) - blk_tmp1[i][j];
            };
            integer_transform(block_size / 2, 0, blk_tmp2, blk_tmp2);
            quantize_coefficients(quality_index, block_size / 2, gradient, blk_tmp2, sub_blk_quat[isub]);
            inverse_quantize(quality_index, block_size / 2, sub_blk_quat[isub], blk_tmp2);
            integer_transform(block_size / 2, 1, blk_tmp2, blk_tmp2);
            {
                int32_t i;
                int32_t j;
                for (i = 0; i < (block_size / 2); i++)
                    for (j = 0; j < (block_size / 2); j++)
                        sub_blk_rcon[isub][i][j] =
                            (uint8_t)(((((blk_tmp2[i][j] + blk_tmp1[i][j]) < ((uint8_t)0))
                                            ? ((uint8_t)0)
                                            : (blk_tmp2[i][j] + blk_tmp1[i][j])) < ((uint8_t)255))
                                          ? (((blk_tmp2[i][j] + blk_tmp1[i][j]) < ((uint8_t)0))
                                                 ? ((uint8_t)0)
                                                 : (blk_tmp2[i][j] + blk_tmp1[i][j]))
                                          : ((uint8_t)255));
            };
        }

        write_cu_split(&tCABAC, &tCtxs, block_size, 0, larger_than_left_cu, larger_than_above_cu);
        write_split_transform_units(&tCABAC, &tCtxs, block_size, prediction_mode, pmode_left, pmode_above,
                                    sub_blk_quat);
        {
            int32_t _sd_i;
            int32_t _sd_j;
            int32_t _sd_diff;
            int32_t _gxo;
            int32_t _gyo;
            int32_t _gxr;
            int32_t _gyr;
            distortion = 0;
            for (_sd_i = 0; _sd_i < block_size; _sd_i++) {
                for (_sd_j = 0; _sd_j < block_size; _sd_j++) {
                    _sd_diff =
                        ((((int32_t)original_block[_sd_i][_sd_j]) - reconstructed_block[_sd_i][_sd_j]) < 0)
                            ? (-(((int32_t)original_block[_sd_i][_sd_j]) - reconstructed_block[_sd_i][_sd_j]))
                            : (((int32_t)original_block[_sd_i][_sd_j]) - reconstructed_block[_sd_i][_sd_j]);
                    distortion += _sd_diff * _sd_diff;
                }
            }

            if (block_size >= 4) {
                for (_sd_i = 1; _sd_i < (block_size - 1); _sd_i++) {
                    for (_sd_j = 1; _sd_j < (block_size - 1); _sd_j++) {
                        _gxo = ((int32_t)original_block[_sd_i][_sd_j + 1]) -
                               ((int32_t)original_block[_sd_i][_sd_j - 1]);
                        _gyo = ((int32_t)original_block[_sd_i + 1][_sd_j]) -
                               ((int32_t)original_block[_sd_i - 1][_sd_j]);
                        _gxr = ((int32_t)reconstructed_block[_sd_i][_sd_j + 1]) -
                               ((int32_t)reconstructed_block[_sd_i][_sd_j - 1]);
                        _gyr = ((int32_t)reconstructed_block[_sd_i + 1][_sd_j]) -
                               ((int32_t)reconstructed_block[_sd_i - 1][_sd_j]);
                        distortion += settings->structure_weight *
                                      ((((_gxo - _gxr) < 0) ? (-(_gxo - _gxr)) : (_gxo - _gxr)) +
                                       (((_gyo - _gyr) < 0) ? (-(_gyo - _gyr)) : (_gyo - _gyr)));
                    }
                }
            }
        };
        rate_distortion = partition_cost(quality_index, distortion,
                                         arithmetic_length(&tCABAC) - arithmetic_length(&oCABAC), gradient);
        if (rdcost_best >= rate_distortion) {
            rdcost_best = rate_distortion;
            best_step = 3;
            *arithmetic = tCABAC;
            *contexts = tCtxs;
            {
                int32_t i;
                int32_t j;
                for (i = 0; i < block_size; i++)
                    for (j = 0; j < block_size; j++)
                        best_rcon[i][j] = reconstructed_block[i][j];
            };
            {
                int32_t i;
                int32_t j;
                for (i = 0; i < nTU; i++)
                    for (j = 0; j < nTU; j++)
                        map_cu_sz[i][j] = (uint8_t)block_size;
            };
            {
                int32_t i;
                int32_t j;
                for (i = 0; i < nTU; i++)
                    for (j = 0; j < nTU; j++)
                        map_pmode[i][j] = (uint8_t)prediction_mode;
            };
        }
    }

    if (block_size == 8) {
        ArithmeticEncoder tCABAC = oCABAC;
        EntropyContexts tCtxs = oCtxs;
        int32_t sub_pmodes[4] = {-1, -1, -1, -1};
        int32_t sub_pmodes_left[4] = {-1, -1, -1, -1};
        int32_t sub_pmodes_above[4] = {-1, -1, -1, -1};
        for (isub = 0; isub < 4; isub++) {
            int32_t rdcost_subpart_best = (int32_t)0x7fffffff;
            prediction_border(block_size / 2, sub_bll_exist[isub], sub_blb_exist[isub], sub_baa_exist[isub],
                              sub_bar_exist[isub], sub_blk_rcon[isub], &ubla, ublb, ubar, &fbla, fblb, fbar);
            for (prediction_mode = 0; prediction_mode <= 1; prediction_mode++) {
                ArithmeticEncoder nCABAC = initialize_arithmetic();
                EntropyContexts nCtxs = initialize_contexts(quality_index);
                intra_prediction(block_size / 2, CH_Y, prediction_mode, ubla, ublb, ubar, fbla, fblb, fbar,
                                 blk_tmp1);
                {
                    int32_t i;
                    int32_t j;
                    for (i = 0; i < (block_size / 2); i++)
                        for (j = 0; j < (block_size / 2); j++)
                            blk_tmp2[i][j] = ((int32_t)sub_blk_orig[isub][i][j]) - blk_tmp1[i][j];
                };
                integer_transform(block_size / 2, 0, blk_tmp2, blk_tmp2);
                quantize_coefficients(quality_index, block_size / 2, gradient, blk_tmp2, quantized_block);
                inverse_quantize(quality_index, block_size / 2, quantized_block, blk_tmp2);
                integer_transform(block_size / 2, 1, blk_tmp2, blk_tmp2);
                {
                    int32_t i;
                    int32_t j;
                    for (i = 0; i < (block_size / 2); i++)
                        for (j = 0; j < (block_size / 2); j++)
                            blk_tmp1[i][j] =
                                (uint8_t)(((((blk_tmp2[i][j] + blk_tmp1[i][j]) < ((uint8_t)0))
                                                ? ((uint8_t)0)
                                                : (blk_tmp2[i][j] + blk_tmp1[i][j])) < ((uint8_t)255))
                                              ? (((blk_tmp2[i][j] + blk_tmp1[i][j]) < ((uint8_t)0))
                                                     ? ((uint8_t)0)
                                                     : (blk_tmp2[i][j] + blk_tmp1[i][j]))
                                              : ((uint8_t)255));
                };
                write_coefficients(&nCABAC, &nCtxs, block_size / 2, CH_Y, prediction_mode, quantized_block);
                {
                    int32_t _sd_i;
                    int32_t _sd_j;
                    int32_t _sd_diff;
                    int32_t _gxo;
                    int32_t _gyo;
                    int32_t _gxr;
                    int32_t _gyr;
                    distortion = 0;
                    for (_sd_i = 0; _sd_i < (block_size / 2); _sd_i++) {
                        for (_sd_j = 0; _sd_j < (block_size / 2); _sd_j++) {
                            _sd_diff =
                                ((((int32_t)sub_blk_orig[isub][_sd_i][_sd_j]) - blk_tmp1[_sd_i][_sd_j]) < 0)
                                    ? (-(((int32_t)sub_blk_orig[isub][_sd_i][_sd_j]) -
                                         blk_tmp1[_sd_i][_sd_j]))
                                    : (((int32_t)sub_blk_orig[isub][_sd_i][_sd_j]) - blk_tmp1[_sd_i][_sd_j]);
                            distortion += _sd_diff * _sd_diff;
                        }
                    }

                    if ((block_size / 2) >= 4) {
                        for (_sd_i = 1; _sd_i < ((block_size / 2) - 1); _sd_i++) {
                            for (_sd_j = 1; _sd_j < ((block_size / 2) - 1); _sd_j++) {
                                _gxo = ((int32_t)sub_blk_orig[isub][_sd_i][_sd_j + 1]) -
                                       ((int32_t)sub_blk_orig[isub][_sd_i][_sd_j - 1]);
                                _gyo = ((int32_t)sub_blk_orig[isub][_sd_i + 1][_sd_j]) -
                                       ((int32_t)sub_blk_orig[isub][_sd_i - 1][_sd_j]);
                                _gxr = ((int32_t)blk_tmp1[_sd_i][_sd_j + 1]) -
                                       ((int32_t)blk_tmp1[_sd_i][_sd_j - 1]);
                                _gyr = ((int32_t)blk_tmp1[_sd_i + 1][_sd_j]) -
                                       ((int32_t)blk_tmp1[_sd_i - 1][_sd_j]);
                                distortion += settings->structure_weight *
                                              ((((_gxo - _gxr) < 0) ? (-(_gxo - _gxr)) : (_gxo - _gxr)) +
                                               (((_gyo - _gyr) < 0) ? (-(_gyo - _gyr)) : (_gyo - _gyr)));
                            }
                        }
                    }
                };
                rate_distortion =
                    partition_cost(quality_index, distortion, arithmetic_length(&nCABAC), gradient);
                if (rdcost_subpart_best >= rate_distortion) {
                    rdcost_subpart_best = rate_distortion;
                    sub_pmodes[isub] = prediction_mode;
                    {
                        int32_t i;
                        int32_t j;
                        for (i = 0; i < (block_size / 2); i++)
                            for (j = 0; j < (block_size / 2); j++)
                                sub_blk_quat[isub][i][j] = quantized_block[i][j];
                    };
                    {
                        int32_t i;
                        int32_t j;
                        for (i = 0; i < (block_size / 2); i++)
                            for (j = 0; j < (block_size / 2); j++)
                                sub_blk_rcon[isub][i][j] = blk_tmp1[i][j];
                    };
                }
            }
        }

        sub_pmodes_left[0] = pmode_left;
        sub_pmodes_above[0] = pmode_above;
        sub_pmodes_left[1] = sub_pmodes[0];
        sub_pmodes_above[1] = sub_map_pmode[1][-1][0];
        sub_pmodes_left[2] = sub_map_pmode[2][0][-1];
        sub_pmodes_above[2] = sub_pmodes[0];
        sub_pmodes_left[3] = sub_pmodes[2];
        sub_pmodes_above[3] = sub_pmodes[1];
        write_cu_split(&tCABAC, &tCtxs, block_size, 0, larger_than_left_cu, larger_than_above_cu);
        write_split_prediction_units(&tCABAC, &tCtxs, block_size, sub_pmodes, sub_pmodes_left,
                                     sub_pmodes_above, sub_blk_quat);
        {
            int32_t _sd_i;
            int32_t _sd_j;
            int32_t _sd_diff;
            int32_t _gxo;
            int32_t _gyo;
            int32_t _gxr;
            int32_t _gyr;
            distortion = 0;
            for (_sd_i = 0; _sd_i < block_size; _sd_i++) {
                for (_sd_j = 0; _sd_j < block_size; _sd_j++) {
                    _sd_diff =
                        ((((int32_t)original_block[_sd_i][_sd_j]) - reconstructed_block[_sd_i][_sd_j]) < 0)
                            ? (-(((int32_t)original_block[_sd_i][_sd_j]) - reconstructed_block[_sd_i][_sd_j]))
                            : (((int32_t)original_block[_sd_i][_sd_j]) - reconstructed_block[_sd_i][_sd_j]);
                    distortion += _sd_diff * _sd_diff;
                }
            }

            if (block_size >= 4) {
                for (_sd_i = 1; _sd_i < (block_size - 1); _sd_i++) {
                    for (_sd_j = 1; _sd_j < (block_size - 1); _sd_j++) {
                        _gxo = ((int32_t)original_block[_sd_i][_sd_j + 1]) -
                               ((int32_t)original_block[_sd_i][_sd_j - 1]);
                        _gyo = ((int32_t)original_block[_sd_i + 1][_sd_j]) -
                               ((int32_t)original_block[_sd_i - 1][_sd_j]);
                        _gxr = ((int32_t)reconstructed_block[_sd_i][_sd_j + 1]) -
                               ((int32_t)reconstructed_block[_sd_i][_sd_j - 1]);
                        _gyr = ((int32_t)reconstructed_block[_sd_i + 1][_sd_j]) -
                               ((int32_t)reconstructed_block[_sd_i - 1][_sd_j]);
                        distortion += settings->structure_weight *
                                      ((((_gxo - _gxr) < 0) ? (-(_gxo - _gxr)) : (_gxo - _gxr)) +
                                       (((_gyo - _gyr) < 0) ? (-(_gyo - _gyr)) : (_gyo - _gyr)));
                    }
                }
            }
        };
        rate_distortion = partition_cost(quality_index, distortion,
                                         arithmetic_length(&tCABAC) - arithmetic_length(&oCABAC), gradient);
        if (rdcost_best >= rate_distortion) {
            rdcost_best = rate_distortion;
            *arithmetic = tCABAC;
            *contexts = tCtxs;
            {
                int32_t i;
                int32_t j;
                for (i = 0; i < nTU; i++)
                    for (j = 0; j < nTU; j++)
                        map_cu_sz[i][j] = (uint8_t)block_size;
            };
            {
                int32_t i;
                int32_t j;
                for (i = 0; i < (nTU / 2); i++)
                    for (j = 0; j < (nTU / 2); j++)
                        sub_map_pmode[0][i][j] = (uint8_t)sub_pmodes[0];
            };
            {
                int32_t i;
                int32_t j;
                for (i = 0; i < (nTU / 2); i++)
                    for (j = 0; j < (nTU / 2); j++)
                        sub_map_pmode[1][i][j] = (uint8_t)sub_pmodes[1];
            };
            {
                int32_t i;
                int32_t j;
                for (i = 0; i < (nTU / 2); i++)
                    for (j = 0; j < (nTU / 2); j++)
                        sub_map_pmode[2][i][j] = (uint8_t)sub_pmodes[2];
            };
            {
                int32_t i;
                int32_t j;
                for (i = 0; i < (nTU / 2); i++)
                    for (j = 0; j < (nTU / 2); j++)
                        sub_map_pmode[3][i][j] = (uint8_t)sub_pmodes[3];
            };
            if (partitions) {
                partitions->cu.count = cu_checkpoint;
                partitions->pu.count = pu_checkpoint;
                partitions->tu.count = tu_checkpoint;
                append_partition(&partitions->cu, block_size, abs_x, abs_y);
                for (isub = 0; isub < 4; isub++) {
                    append_partition(&partitions->pu, block_size / 2, abs_x + ((isub % 2) * (block_size / 2)),
                                     abs_y + ((isub / 2) * (block_size / 2)));
                    append_partition(&partitions->tu, block_size / 2, abs_x + ((isub % 2) * (block_size / 2)),
                                     abs_y + ((isub / 2) * (block_size / 2)));
                }
            }
            return;
        }
    }
    if (partitions && (best_step >= 2)) {
        partitions->cu.count = cu_checkpoint;
        partitions->pu.count = pu_checkpoint;
        partitions->tu.count = tu_checkpoint;
        append_partition(&partitions->cu, block_size, abs_x, abs_y);
        append_partition(&partitions->pu, block_size, abs_x, abs_y);
        if (best_step == 2) {
            append_partition(&partitions->tu, block_size, abs_x, abs_y);
        } else {
            for (isub = 0; isub < 4; isub++)
                append_partition(&partitions->tu, block_size / 2, abs_x + ((isub % 2) * (block_size / 2)),
                                 abs_y + ((isub / 2) * (block_size / 2)));
        }
    }
    {
        int32_t i;
        int32_t j;
        for (i = 0; i < block_size; i++)
            for (j = 0; j < block_size; j++)
                reconstructed_block[i][j] = best_rcon[i][j];
    };
}

int32_t encode_intra_picture(uint8_t *stream, const uint8_t *img, uint8_t *reconstructed, int32_t *height,
                             int32_t *width, const int32_t quality_index) {
    ArithmeticEncoder tCABAC = initialize_arithmetic();
    EntropyContexts tCtxs = initialize_contexts(quality_index);
    const int32_t yszn = ((((((*height) < 8192) ? (*height) : (8192)) + 32) - 1) / 32) * 32;
    const int32_t xszn = ((((((*width) < 8192) ? (*width) : (8192)) + 32) - 1) / 32) * 32;
    uint8_t *pbuf = stream;
    int32_t y;
    int32_t x;
    int32_t i;
    int32_t j;
    uint8_t ctu_orig[32][32];
    uint8_t ctu_rcon_0[1 + 32][1 + (32 * 2)];
    uint8_t (*ctu_rcon)[1 + (32 * 2)] = (uint8_t (*)[1 + (32 * 2)])(&ctu_rcon_0[1][1]);
    uint8_t map_cu_sz_0[1 + (32 / 4)][1 + (8192 / 4)];
    uint8_t map_pmode_0[1 + (32 / 4)][1 + (8192 / 4)];
    for (i = 0; i <= (32 / 4); i++) {
        for (j = 0; j <= (8192 / 4); j++) {
            map_cu_sz_0[i][j] = 32;
            map_pmode_0[i][j] = 1;
        }
    }

    write_picture_header(&pbuf, quality_index, yszn, xszn);
    for (y = 0; y < yszn; y += 32) {
        for (x = 0; x < xszn; x += 32) {
            const uint8_t bll_exist = x > 0;
            const uint8_t blb_exist = 0;
            const uint8_t baa_exist = y > 0;
            const uint8_t bar_exist = baa_exist && ((x + 32) < xszn);
            uint8_t (*map_cu_sz)[1 + (8192 / 4)] =
                (uint8_t (*)[1 + (8192 / 4)])(&map_cu_sz_0[1][1 + (x / 4)]);
            uint8_t (*map_pmode)[1 + (8192 / 4)] =
                (uint8_t (*)[1 + (8192 / 4)])(&map_pmode_0[1][1 + (x / 4)]);
            for (i = 0; i < 32; i++)
                ctu_rcon[i][-1] =
                    *((reconstructed + (xszn * (((((y + i) < 0) ? (0) : (y + i)) < (yszn - 1))
                                                    ? (((y + i) < 0) ? (0) : (y + i))
                                                    : (yszn - 1)))) +
                      (((((x - 1) < 0) ? (0) : (x - 1)) < (xszn - 1)) ? (((x - 1) < 0) ? (0) : (x - 1))
                                                                      : (xszn - 1)));

            for (j = -1; j < (32 * 2); j++)
                ctu_rcon[-1][j] =
                    *((reconstructed + (xszn * (((((y - 1) < 0) ? (0) : (y - 1)) < (yszn - 1))
                                                    ? (((y - 1) < 0) ? (0) : (y - 1))
                                                    : (yszn - 1)))) +
                      (((((x + j) < 0) ? (0) : (x + j)) < (xszn - 1)) ? (((x + j) < 0) ? (0) : (x + j))
                                                                      : (xszn - 1)));

            for (i = 0; i < 32; i++)
                for (j = 0; j < 32; j++)
                    ctu_orig[i][j] = *((img + ((*width) * (((((y + i) < 0) ? (0) : (y + i)) < ((*height) - 1))
                                                               ? (((y + i) < 0) ? (0) : (y + i))
                                                               : ((*height) - 1)))) +
                                       (((((x + j) < 0) ? (0) : (x + j)) < ((*width) - 1))
                                            ? (((x + j) < 0) ? (0) : (x + j))
                                            : ((*width) - 1)));

            if (partitions)
                append_partition(&partitions->ctu, 32, x, y);
            search_coding_unit(quality_index, &tCABAC, &tCtxs, ctu_orig, ctu_rcon, map_cu_sz, map_pmode, 32,
                               bll_exist, blb_exist, baa_exist, bar_exist, y, x);
            for (i = 0; i < 32; i++)
                for (j = 0; j < 32; j++)
                    *((reconstructed + (xszn * (((((y + i) < 0) ? (0) : (y + i)) < (yszn - 1))
                                                    ? (((y + i) < 0) ? (0) : (y + i))
                                                    : (yszn - 1)))) +
                      (((((x + j) < 0) ? (0) : (x + j)) < (xszn - 1)) ? (((x + j) < 0) ? (0) : (x + j))
                                                                      : (xszn - 1))) = ctu_rcon[i][j];

            write_termination(&tCABAC, ((y + 32) >= yszn) && ((x + 32) >= xszn));
            submit_arithmetic(&tCABAC, &pbuf);
        }

        for (j = 1; j <= (8192 / 4); j++) {
            map_cu_sz_0[0][j] = map_cu_sz_0[32 / 4][j];
        }
    }

    finish_arithmetic(&tCABAC);
    submit_arithmetic(&tCABAC, &pbuf);
    *height = yszn;
    *width = xszn;
    return (int32_t)(pbuf - stream);
}

static void append_partition(PartitionList *list, int size, int x, int y) {
    if (list->count == list->capacity) {
        int capacity = (list->capacity) ? (list->capacity * 2) : (256);
        PartitionRecord *records =
            (PartitionRecord *)realloc(list->blocks, ((size_t)capacity) * (sizeof(*records)));
        if (!records) {
            collection_failed = 1;
            return;
        }
        list->blocks = records;
        list->capacity = capacity;
    }
    list->blocks[list->count++] = (PartitionRecord){size, x, y};
}

static int export_partitions(const char *path, const char *kind, const PartitionList *list) {
    char name[4096];
    const char *extension = strrchr(path, '.');
    size_t length = (extension) ? ((size_t)(extension - path)) : (strlen(path));
    if (((length + strlen(kind)) + 6) >= (sizeof(name)))
        return 1;
    memcpy(name, path, length);
    name[length] = 0;
    snprintf(name + length, (sizeof(name)) - length, "_%s.csv", kind);
    FILE *file = fopen(name, "wb");
    if (!file)
        return 1;
    int failed = 0;
    for (int index = 0; index < list->count; index++) {
        PartitionRecord record = list->blocks[index];
        if (fprintf(file, "%d,%d,%d\n", record.size, record.x, record.y) < 0)
            failed = 1;
    }

    return (fclose(file) != 0) || failed;
}

int usr_hevc_write(const uint8_t *pixels, int height, int width, const char *path,
                   const HevcParameters *parameters) {
    if ((((((height < 32) || (width < 32)) || (height > 8192)) || (width > 8192)) || (height % 32)) ||
        (width % 32))
        return 1;
    size_t area = ((size_t)(width + 32)) * (height + 32);
    uint8_t *memory = (uint8_t *)malloc(((2 * area) + 65536) + ((area + 1048576) * 2));
    if (!memory)
        return 1;
    uint8_t *original = (memory + (2 * area)) + 65536;
    uint8_t *reconstruction = (original + area) + 1048576;
    memcpy(original, pixels, ((size_t)height) * width);
    PartitionCollection collection = {0};
    partitions = &collection;
    settings = parameters;
    collection_failed = 0;
    int bytes =
        encode_intra_picture(memory, original, reconstruction, &height, &width, parameters->quality_index);
    partitions = NULL;
    int failed = (bytes <= 0) || collection_failed;
    if (!failed)
        failed =
            export_partitions(path, "tu", &collection.tu) || export_partitions(path, "pu", &collection.pu);
    if (!failed) {
        FILE *file = fopen(path, "wb");
        if (!file)
            failed = 1;
        else {
            failed = fwrite(memory, 1, (size_t)bytes, file) != ((size_t)bytes);
            failed |= fclose(file) != 0;
        }
    }
    free(collection.ctu.blocks);
    free(collection.cu.blocks);
    free(collection.tu.blocks);
    free(collection.pu.blocks);
    free(memory);
    return failed;
}
