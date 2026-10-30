/* Grayscale PGM input and explicit search parameters. */
#include "codec.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

static int next_number(FILE *stream) {
    int character;
    do {
        character = fgetc(stream);
        if (character == '#')
            while (character != '\n' && character != EOF)
                character = fgetc(stream);
    } while (character != EOF && isspace(character));
    if (character == EOF || !isdigit(character))
        return -1;
    int value = 0;
    do {
        if (value > 100000000)
            return -1;
        value = value * 10 + character - '0';
        character = fgetc(stream);
    } while (isdigit(character));
    if (character == '\r') {
        character = fgetc(stream);
        if (character != '\n')
            ungetc(character, stream);
    }
    return value;
}

static uint8_t *read_pgm(const char *path, int *height, int *width) {
    FILE *stream = fopen(path, "rb");
    if (!stream)
        return NULL;
    int magic = fgetc(stream), type = fgetc(stream);
    *width = next_number(stream);
    *height = next_number(stream);
    int maximum = next_number(stream);
    if (magic != 'P' || type != '5' || maximum != 255 || *width < 32 || *height < 32 || *width > 8192 ||
        *height > 8192 || *width % 32 || *height % 32) {
        fclose(stream);
        return NULL;
    }
    size_t length = (size_t)*width * *height;
    uint8_t *pixels = (uint8_t *)malloc(length);
    if (pixels && fread(pixels, 1, length, stream) != length) {
        free(pixels);
        pixels = NULL;
    }
    fclose(stream);
    return pixels;
}

static int read_parameters(const char *path, HevcParameters *parameters) {
    FILE *stream = fopen(path, "r");
    if (!stream)
        return 1;
    int *fields[] = {&parameters->quality_index,
                     &parameters->gradient_threshold,
                     &parameters->boost_offset,
                     &parameters->boost_limit,
                     &parameters->distortion_weight_limit,
                     &parameters->rate_weight_floor,
                     &parameters->zero_multiplier,
                     &parameters->zero_shift,
                     &parameters->zero_reduction_shift,
                     &parameters->structure_weight};
    int failed = 0;
    for (int index = 0; index < 10; index++)
        failed |= fscanf(stream, "%d", fields[index]) != 1;
    for (int index = 0; index < 5; index++)
        failed |= fscanf(stream, "%d", &parameters->distortion_weights[index]) != 1;
    for (int index = 0; index < 5; index++)
        failed |= fscanf(stream, "%d", &parameters->rate_weights[index]) != 1;
    fclose(stream);
    if (failed || parameters->quality_index < 0 || parameters->quality_index > 4 ||
        parameters->boost_limit < 0 || parameters->rate_weight_floor < 1 || parameters->zero_multiplier < 1 ||
        parameters->distortion_weight_limit < 1 || parameters->zero_shift < 0 ||
        parameters->zero_shift > 16 || parameters->zero_reduction_shift < 0 ||
        parameters->zero_reduction_shift > 16 || parameters->structure_weight < 0)
        return 1;
    for (int index = 0; index < 5; index++)
        if (parameters->distortion_weights[index] < 1 || parameters->rate_weights[index] < 1)
            return 1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: hevc_prior input.pgm output.hevc search_parameters.txt\n");
        return 2;
    }
    HevcParameters parameters;
    if (read_parameters(argv[3], &parameters)) {
        fprintf(stderr, "Invalid search parameters\n");
        return 2;
    }
    FILE *existing = fopen(argv[2], "rb");
    if (existing) {
        fclose(existing);
        fprintf(stderr, "Output already exists\n");
        return 2;
    }
    int height, width;
    uint8_t *pixels = read_pgm(argv[1], &height, &width);
    if (!pixels) {
        fprintf(stderr, "Expected P5 grayscale with complete 32-pixel CTUs\n");
        return 2;
    }
    int failed = usr_hevc_write(pixels, height, width, argv[2], &parameters);
    free(pixels);
    return failed;
}
