#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

int cal_pixel(double creal, double cimag) {
    double z_real = 0.0, z_imag = 0.0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2.0 * z_real * z_imag + cimag;
        z_real = z_real2 - z_imag2 + creal;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while (iter < MAX_ITER && lengthsq < 4.0);
    return iter;
}

void fill_row(int row, int *rowbuf) {
    double cimag = (row - HEIGHT / 2.0) * 4.0 / HEIGHT;
    for (int j = 0; j < WIDTH; j++) {
        double creal = (j - WIDTH / 2.0) * 4.0 / WIDTH;
        rowbuf[j] = cal_pixel(creal, cimag);
    }
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE *pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n%d %d\n255\n", WIDTH, HEIGHT);
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(pgmimg, "%d ", image[i][j]);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int (*image)[WIDTH] = NULL;
    if (rank == 0)
        image = malloc(sizeof(int) * WIDTH * HEIGHT);

    double start = MPI_Wtime();

    const int TAG_WORK = 1, TAG_RESULT = 2;

    if (rank == 0) {
        int next_row = 0, active = 0;
        for (int p = 1; p < size; p++) {
            if (next_row < HEIGHT) {
                MPI_Send(&next_row, 1, MPI_INT, p, TAG_WORK, MPI_COMM_WORLD);
                next_row++;
                active++;
            }
        }

        while (active > 0) {
            int row;
            MPI_Status st;
            MPI_Recv(&row, 1, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &st);
            MPI_Recv(&image[row][0], WIDTH, MPI_INT, st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (next_row < HEIGHT) {
                MPI_Send(&next_row, 1, MPI_INT, st.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
                next_row++;
            } else {
                int stop = -1;
                MPI_Send(&stop, 1, MPI_INT, st.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
                active--;
            }
        }
    } else {
        int row;
        int *buf = malloc(sizeof(int) * WIDTH);
        while (1) {
            MPI_Recv(&row, 1, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (row < 0) break;
            fill_row(row, buf);
            MPI_Send(&row, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
            MPI_Send(buf, WIDTH, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
        free(buf);
    }

    double end = MPI_Wtime();
    double elapsed = end - start;
    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("[DYNAMIC] Execution time: %.6f seconds\n", max_elapsed);
        save_pgm("mandelbrot_dynamic.pgm", image);
        free(image);
    }

    MPI_Finalize();
    return 0;
}
