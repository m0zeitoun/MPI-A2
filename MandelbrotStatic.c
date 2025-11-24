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

    int rows_per_rank = HEIGHT / size;
    int remainder = HEIGHT % size;
    int my_rows = rows_per_rank + (rank < remainder ? 1 : 0);
    int my_start = rank * rows_per_rank + (rank < remainder ? rank : remainder);

    int *local = malloc(sizeof(int) * my_rows * WIDTH);
    for (int r = 0; r < my_rows; r++)
        fill_row(my_start + r, &local[r * WIDTH]);

    int *recvcounts = NULL, *displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(sizeof(int) * size);
        displs = malloc(sizeof(int) * size);
        for (int p = 0; p < size; p++) {
            int rows = rows_per_rank + (p < remainder ? 1 : 0);
            recvcounts[p] = rows * WIDTH;
        }
        displs[0] = 0;
        for (int p = 1; p < size; p++)
            displs[p] = displs[p - 1] + recvcounts[p - 1];
    }

    MPI_Gatherv(local, my_rows * WIDTH, MPI_INT,
                rank == 0 ? &image[0][0] : NULL,
                recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    double end = MPI_Wtime();
    double elapsed = end - start;
    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("[STATIC] Execution time: %.6f seconds\n", max_elapsed);
        save_pgm("mandelbrot_static.pgm", image);
        free(image);
        free(recvcounts);
        free(displs);
    }

    free(local);
    MPI_Finalize();
    return 0;
}
